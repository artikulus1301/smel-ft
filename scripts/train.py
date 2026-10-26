"""Stage 6/7: layer-major LoRA training of the selected experts (+ shared experts) on top of the cached input of L.

Every optimizer step loads each block layer twice (forward, then backward with recompute) regardless of the number
of microbatches. Logs loss / val loss / tok/s / VRAM / RAM / load vs compute / GPU temperature and monitors the
share of domain gate mass routed to the selected experts.
Usage: python scripts/train.py [--max-steps 50] [--resume] [--set train.lr=2e-4]
"""
import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import yaml
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from smel.data import make_steps  # noqa: E402
from smel.layer_runner import LayerMajorRunner, make_ctx  # noqa: E402
from smel.lora import BlockLoRA  # noqa: E402
from smel.loss import LMHead  # noqa: E402
from smel.model_io import Checkpoint  # noqa: E402
from smel.pack import pack_layer  # noqa: E402
from smel.pipeline import (ROOT, HiddenStore, JsonlLog, TokenSet, append_progress, gpu_stats,  # noqa: E402
                           load_config, mb_rows, microbatches, parse_overrides, rss_gb)
from smel.stream import DictSource, PackedSource, build_layer  # noqa: E402


class Batches:
    def __init__(self, ts, store, d, device="cuda"):
        self.ts, self.store, self.d, self.device = ts, store, d, device

    def make(self, mbs):
        hs, ctxs, tg = [], [], []
        for mb in mbs:
            rows = mb_rows(self.ts, mb)
            hs.append(self.store.read(rows).to(self.device, non_blocking=True))
            ctxs.append(make_ctx([self.ts.lengths[i] for i in mb], self.d, self.device, torch.bfloat16))
            tg.append(torch.from_numpy(self.ts.tgt[rows].astype(np.int64)).to(self.device))
        denom = sum(int((t != -100).sum()) for t in tg)
        return hs, ctxs, tg, denom


def plan_steps(ts, idx, cfg, epochs, seed):
    tc = cfg["train"]
    plan = []
    for ep in range(epochs):
        rng = random.Random(seed + ep)
        order = list(idx)
        rng.shuffle(order)
        mbs = microbatches(ts, order, tc["mb_tokens"], seed=seed + 1000 + ep)
        plan += [(ep, s) for s in make_steps(mbs, ts.lengths, tc["tokens_per_step"])]
    return plan


def build_source(cfg, sel, d, ck, layers, mode, log):
    L = sel["L"]
    experts = {int(k): v for k, v in sel["experts"].items()}
    packed = os.path.join(cfg["paths"]["work_dir"], f"packed_L{L}")
    meta_p = os.path.join(packed, "meta.json")
    have_packed = os.path.exists(meta_p) and all(os.path.exists(os.path.join(packed, f"layer_{l:02d}.safetensors"))
                                                 for l in layers)
    # the packed int4 bank holds every expert; a different expert set only needs its bf16 copies from the original
    override = None
    if have_packed and json.load(open(meta_p)).get("experts") != {str(k): v for k, v in experts.items()}:
        override = experts
        log(f"packed block was made for another expert set -> bf16 copies of {sum(map(len, experts.values()))} "
            f"experts read from the original checkpoint")
    group = cfg["model"]["expert_group"]
    preloaded = None
    if not have_packed:
        log("no packed block on disk -> quantizing layers into RAM")
        preloaded = {}
        for l in layers:
            t = {k: v.cuda() for k, v in ck.layer(l).items()}
            preloaded[l] = {k: v.cpu() for k, v in pack_layer(t, experts.get(l), cfg["model"]["quant_group"]).items()}
            del t
    if mode == "resident":
        src = PackedSource(packed, d, layers, "cuda", group=group, preloaded=preloaded, selection=override, ckpt=ck)
        res = {}
        for l in layers:
            res[l] = src.get(l)
            src.cache.clear()
        src.close()
        log(f"block resident on GPU: {torch.cuda.memory_allocated()/2**30:.2f} GB allocated")
        return DictSource(res)
    pin = mode == "stream_pinned" and cfg["train"]["pin"]
    src = PackedSource(packed, d, layers, "cuda", pin=pin, group=group, pin_budget_gb=cfg["select"]["pin_budget_gb"],
                       preloaded=preloaded, selection=override, ckpt=ck)
    log(f"streaming block, pinned {src.pinned_gb:.2f} GB")
    return src


def load_head(cfg, sel, ck):
    packed = os.path.join(cfg["paths"]["work_dir"], f"packed_L{sel['L']}", "head.safetensors")
    if os.path.exists(packed):
        t = load_file(packed)
        norm = t["norm.weight"].cuda()
        if "lm_head.q" in t:
            return norm, LMHead(int8={"q": t["lm_head.q"].cuda(), "scale": t["lm_head.scale"].cuda()})
        return norm, LMHead(weight=t["lm_head.weight"].cuda())
    return ck.get(ck.norm_name()).cuda(), LMHead.from_weight(ck.get(ck.lm_head_name()).cuda(),
                                                             int8=cfg["model"]["lm_head_int8"])


def monitor_share(runner, batch, experts):
    """Share of top-k gate mass that goes to the selected experts, per block layer."""
    hs, ctxs, _, _ = batch
    acc = {}
    sel = {l: torch.tensor(v, device="cuda") for l, v in experts.items() if len(v)}

    def cb(l, mb, probs, topv, topi):
        if l not in sel:
            return
        m = torch.isin(topi, sel[l])
        a = acc.setdefault(l, [0.0, 0.0])
        a[0] += float((topv.float() * m).sum())
        a[1] += float(topv.float().sum())

    runner.forward([h.clone() for h in hs], ctxs, stats_cb=cb)
    return {l: a[0] / max(a[1], 1e-9) for l, a in sorted(acc.items())}


def save_adapter(lora, sel, cfg, path, extra=None):
    os.makedirs(path, exist_ok=True)
    save_file({k: v.detach().float().cpu().contiguous() for k, v in lora.state_dict().items()},
              os.path.join(path, "adapter.safetensors"))
    meta = {"base_model": "Qwen/Qwen3.6-35B-A3B", "L": sel["L"], "experts": sel["experts"], "r": cfg["train"]["r"],
            "alpha": cfg["train"]["alpha"], "shared_lora": cfg["train"]["shared_lora"],
            "format": "smel-v1: layers.{l}.experts.{A,B}_{gate,up,down}[slot], layers.{l}.shared.{A,B}_*; "
                      "delta = alpha/r * B @ A"}
    meta.update(extra or {})
    json.dump(meta, open(os.path.join(path, "adapter_config.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--max-steps", type=int, default=None, help="stop after N steps (smoke test)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--selection", default=os.path.join(ROOT, "configs", "selection.yaml"))
    args = ap.parse_args()
    cfg = load_config(args.config, parse_overrides(args.set))
    tc = cfg["train"]
    assert tc["dropout"] == 0.0, "LoRA dropout is not implemented"
    sel = yaml.safe_load(open(args.selection))
    L = sel["L"]
    experts = {int(k): v for k, v in sel["experts"].items()}
    tag = args.tag or f"L{L}"
    run_dir = os.path.join(cfg["paths"]["work_dir"], f"train_{tag}")
    os.makedirs(run_dir, exist_ok=True)
    logj = JsonlLog(os.path.join(cfg["paths"]["logs"], f"train_{tag}.jsonl"))

    def log(msg, **kw):
        print(msg, flush=True)
        logj(msg=msg, **kw)

    torch.manual_seed(tc["seed"])
    ck = Checkpoint(cfg["paths"]["model_dir"])
    d = ck.dims
    layers = list(range(L, d.n_layers))
    mode = sel["mode"] if tc["resident"] == "auto" else ("resident" if tc["resident"] else "stream_pinned")
    src = build_source(cfg, sel, d, ck, layers, mode, log)
    norm_w, head = load_head(cfg, sel, ck)
    lora = BlockLoRA.build(experts, d.hidden, d.moe_inter, d.shared_inter, tc["r"], tc["alpha"],
                           shared=tc["shared_lora"]).cuda()
    n_params = sum(p.numel() for p in lora.parameters())
    runner = LayerMajorRunner(d, src, layers, lora, norm_w, head,
                              ckpt_device=None if tc["ckpt_device"] == "cuda" else "cpu")

    ts = TokenSet.load(os.path.join(cfg["paths"]["work_dir"], "tokens_domain.npz"))
    store = HiddenStore(os.path.join(cfg["paths"]["work_dir"], f"cache_L{L}", "h_domain.npy"))
    B = Batches(ts, store, d)
    train_idx, val_idx = ts.indices("train"), ts.indices("val")
    plan = plan_steps(ts, train_idx, cfg, tc["epochs"], tc["seed"])
    total = len(plan)
    val_mbs = microbatches(ts, val_idx, tc["mb_tokens"])
    mon_idx, n = [], 0
    for i in val_idx:
        if n >= tc["monitor_tokens"]:
            break
        mon_idx.append(i)
        n += ts.lengths[i]
    mon_batch = B.make(microbatches(ts, mon_idx, tc["mb_tokens"]))

    opt = torch.optim.AdamW(lora.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"], betas=(0.9, 0.999))
    warm = max(1, int(round(tc["warmup_frac"] * total)))

    def lr_lambda(s):
        if s < warm:
            return (s + 1) / warm
        p = (s - warm) / max(1, total - warm)
        return tc["min_lr_frac"] + (1 - tc["min_lr_frac"]) * 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    start, mon0, hot = 0, None, 0
    ck_path = os.path.join(run_dir, "ckpt.pt")
    if args.resume and os.path.exists(ck_path):
        st = torch.load(ck_path, map_location="cuda", weights_only=False)
        lora.load_state_dict(st["lora"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        start, mon0 = st["step"], st.get("mon0")
        log(f"resumed at step {start}/{total}")

    def val_loss():
        vb = B.make(val_mbs)
        return runner.eval_loss(*vb)

    log(f"train L={L}: {len(layers)} layers, {sum(len(v) for v in experts.values())} experts + shared, "
        f"LoRA {n_params/1e6:.2f}M params; {len(train_idx)} train seqs, {total} steps "
        f"({tc['epochs']} epochs), warmup {warm}, mode {mode}")
    if start == 0:
        v0 = val_loss()
        mon0 = monitor_share(runner, mon_batch, experts)
        log(f"step 0: val {v0:.4f} (ppl {math.exp(v0):.3f}); gate share on selected "
            f"{np.mean(list(mon0.values())):.3f}", step=0, val=v0, mon=mon0)
    end = total if args.max_steps is None else min(total, start + args.max_steps)
    t_run = time.perf_counter()
    tok_run = 0
    for s in range(start, end):
        ep, mbs = plan[s]
        torch.cuda.reset_peak_memory_stats()
        st0 = dict(src.stats)
        t0 = time.perf_counter()
        hs, ctxs, tg, denom = B.make(mbs)
        ntok = sum(h.shape[0] for h in hs)
        loss = runner.train_step(hs, ctxs, tg, denom)
        gn = float(torch.nn.utils.clip_grad_norm_(lora.parameters(), tc["grad_clip"]))
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        tok_run += ntok
        rec = {"step": s + 1, "epoch": ep, "loss": loss, "grad_norm": gn, "lr": sched.get_last_lr()[0],
               "tokens": ntok, "sup_tokens": denom, "tok_s": ntok / dt, "step_s": dt,
               "load_s": src.stats["load_s"] - st0["load_s"], "wait_s": src.stats["wait_s"] - st0["wait_s"],
               "vram_peak_gb": torch.cuda.max_memory_allocated() / 2**30, "rss_gb": rss_gb()}
        if (s + 1) % tc["log_every"] == 0 or s + 1 == end:
            rec.update(gpu_stats())
            if rec.get("gpu_temp", 0) > 85:
                hot += 1
                if hot >= 3:
                    log("WARNING: GPU > 85 C for 3 logs in a row; consider a lower power limit")
            else:
                hot = 0
        if (s + 1) % tc["val_every"] == 0 or s + 1 == end:
            rec["val"] = val_loss()
        if (s + 1) % tc["monitor_every"] == 0 or s + 1 == end:
            mon = monitor_share(runner, mon_batch, experts)
            rec["mon"] = mon
            drops = {l: 1 - mon[l] / max(mon0[l], 1e-9) for l in mon}
            bad = {l: round(v, 3) for l, v in drops.items() if v > tc["monitor_drop_warn"]}
            if bad:
                w = f"WARNING step {s+1}: gate share on selected experts dropped >{tc['monitor_drop_warn']:.0%}: {bad}"
                log(w)
                append_progress(cfg, f"- {w}")
        log(f"step {s+1}/{total} ep {ep} loss {loss:.4f} gn {gn:.2f} lr {rec['lr']:.2e} | {rec['tok_s']:.0f} tok/s "
            f"{dt:.1f}s (load {rec['load_s']:.1f}s, wait {rec['wait_s']:.1f}s) | VRAM {rec['vram_peak_gb']:.2f} GB "
            f"RSS {rec['rss_gb']:.1f} GB" + (f" | {rec['gpu_temp']:.0f}C" if "gpu_temp" in rec else "") +
            (f" | val {rec['val']:.4f}" if "val" in rec else ""), **rec)
        if (s + 1) % tc["ckpt_every"] == 0 or s + 1 == end:
            torch.save({"lora": lora.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "step": s + 1, "mon0": mon0}, ck_path + ".tmp")
            os.replace(ck_path + ".tmp", ck_path)
    el = time.perf_counter() - t_run
    out = os.path.join(cfg["paths"]["artifacts"], f"adapter_{tag}")
    save_adapter(lora, sel, cfg, out, {"steps": end, "total_steps": total})
    msg = (f"### train {tag}: steps {start}->{end}/{total} in {el/60:.1f} min ({tok_run/max(el,1e-9):.0f} tok/s); "
           f"adapter -> {out}")
    log(msg)
    append_progress(cfg, msg)
    src.close()


if __name__ == "__main__":
    main()
