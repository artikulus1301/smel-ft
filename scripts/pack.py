"""Stage 5: pack the trained block (layers L..N-1, final norm, lm_head) and verify it against the bf16 original.

Frozen experts -> int4 (group 64, bf16 scale/zero); selected experts and dense parts -> bf16; lm_head -> int8.
Output: work_dir/packed_L{L}/layer_XX.safetensors + head.safetensors; verification in reports/pack.md.
Usage: python scripts/pack.py [--verify-n 50] [--no-write]
"""
import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
import torch
import yaml
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from smel.layer_runner import LayerMajorRunner, make_ctx, rmsnorm  # noqa: E402
from smel.loss import LMHead  # noqa: E402
from smel.model_io import Checkpoint  # noqa: E402
from smel.pack import pack_layer  # noqa: E402
from smel.pipeline import (ROOT, HiddenStore, TokenSet, append_progress, load_config, mb_rows,  # noqa: E402
                           microbatches, parse_overrides)
from smel.quant import quantize_int8_rows, rel_err  # noqa: E402
from smel.stream import CheckpointSource, PackedSource  # noqa: E402


def head_tensors(ck, int8=True):
    w = ck.get(ck.lm_head_name())
    out = {"norm.weight": ck.get(ck.norm_name())}
    if int8:
        q = quantize_int8_rows(w.cuda())
        out["lm_head.q"], out["lm_head.scale"] = q["q"].cpu(), q["scale"].cpu()
    else:
        out["lm_head.weight"] = w
    return out


def load_head(path, device="cuda"):
    from safetensors import safe_open
    with safe_open(path, "pt") as f:
        t = {k: f.get_tensor(k) for k in f.keys()}
    norm = t["norm.weight"].to(device)
    if "lm_head.q" in t:
        head = LMHead(int8={"q": t["lm_head.q"].to(device), "scale": t["lm_head.scale"].to(device)})
    else:
        head = LMHead(weight=t["lm_head.weight"].to(device))
    return norm, head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--verify-n", type=int, default=50)
    ap.add_argument("--selection", default=os.path.join(ROOT, "configs", "selection.yaml"))
    ap.add_argument("--no-write", action="store_true", help="only verify (quantizing on the fly)")
    args = ap.parse_args()
    cfg = load_config(args.config, parse_overrides(args.set))
    sel = yaml.safe_load(open(args.selection))
    L = sel["L"]
    experts = {int(k): v for k, v in sel["experts"].items()}
    ck = Checkpoint(cfg["paths"]["model_dir"])
    d = ck.dims
    layers = list(range(L, d.n_layers))
    out = os.path.join(cfg["paths"]["work_dir"], f"packed_L{L}")
    t0 = time.perf_counter()
    if not args.no_write:
        os.makedirs(out, exist_ok=True)
        g = cfg["model"]["quant_group"]
        per_layer = d.n_experts * 3 * d.hidden * d.moe_inter * (0.5 + 4 / g) + 40 * d.hidden ** 2  # int4 bank + dense
        need = len(layers) * per_layer + d.vocab * d.hidden
        free = shutil.disk_usage(out).free
        if free < need * 1.1:
            raise SystemExit(f"not enough disk for the packed block: need ~{need/2**30:.1f} GB, free {free/2**30:.1f}"
                             f" GB. Use --no-write (training can quantize at start) or free space.")
        want = {"L": L, "experts": {str(k): v for k, v in experts.items()}, "quant_group": cfg["model"]["quant_group"]}
        meta_p = os.path.join(out, "meta.json")
        fresh = not (os.path.exists(meta_p) and json.load(open(meta_p)) == want)
        for l in layers:
            p = os.path.join(out, f"layer_{l:02d}.safetensors")
            if os.path.exists(p) and not fresh:
                continue
            t = {k: v.cuda() for k, v in ck.layer(l).items()}
            pk = {k: v.cpu() for k, v in pack_layer(t, experts.get(l), cfg["model"]["quant_group"]).items()}
            save_file(pk, p + ".tmp")
            os.replace(p + ".tmp", p)
            print(f"packed layer {l}: {os.path.getsize(p)/2**30:.3f} GB", flush=True)
        hp = os.path.join(out, "head.safetensors")
        if not os.path.exists(hp) or fresh:
            save_file(head_tensors(ck, cfg["model"]["lm_head_int8"]), hp)
        json.dump(want, open(meta_p, "w"))
    t_pack = time.perf_counter() - t0

    # verification on the first N domain eval sequences from the cache
    cache = os.path.join(cfg["paths"]["work_dir"], f"cache_L{L}")
    ts = TokenSet.load(os.path.join(cfg["paths"]["work_dir"], "tokens_domain.npz"))
    idx = (ts.indices("eval") + ts.indices("val") + ts.indices("train"))[: args.verify_n]
    store = HiddenStore(os.path.join(cache, "h_domain.npy"))
    mbs = microbatches(ts, idx, cfg["train"]["mb_tokens"])
    res = {}
    norm_w = ck.get(ck.norm_name()).cuda()
    for name in ("bf16", "packed"):
        if name == "bf16":
            src = CheckpointSource(ck, "cuda")
            head = LMHead.from_weight(ck.get(ck.lm_head_name()).cuda())
        elif args.no_write:
            src = CheckpointSource(ck, "cuda", selection=experts, frozen="int4", quant_group=cfg["model"]["quant_group"])
            head = LMHead.from_weight(ck.get(ck.lm_head_name()).cuda(), int8=cfg["model"]["lm_head_int8"])
        else:
            src = PackedSource(out, d, layers, "cuda")
            norm_w, head = load_head(os.path.join(out, "head.safetensors"))
        run = LayerMajorRunner(d, src, layers, None, norm_w, head)
        hs = [store.read(mb_rows(ts, mb)).cuda() for mb in mbs]
        ctxs = [make_ctx([ts.lengths[i] for i in mb], d, "cuda", torch.bfloat16) for mb in mbs]
        tg = [torch.from_numpy(ts.tgt[mb_rows(ts, mb)].astype(np.int64)).cuda() for mb in mbs]
        denom = sum(int((t != -100).sum()) for t in tg)
        loss = run.eval_loss([h.clone() for h in hs], ctxs, tg, denom)
        hs, _ = run.forward(hs, ctxs)
        res[name] = {"loss": loss, "h": torch.cat([rmsnorm(h, norm_w, d.eps) for h in hs]).float().cpu()}
        src.close()
    e_h = rel_err(res["packed"]["h"], res["bf16"]["h"])
    lb, lp = res["bf16"]["loss"], res["packed"]["loss"]
    msg = (f"### pack L={L}\n- {len(layers)} layers packed in {t_pack:.0f}s -> {out}\n"
           f"- verify on {len(idx)} domain seqs: final hidden rel err {e_h:.4f}; loss bf16 {lb:.4f} -> packed {lp:.4f}"
           f" (ppl {np.exp(lb):.3f} -> {np.exp(lp):.3f}, {100*(np.exp(lp)/np.exp(lb)-1):+.2f}%)")
    print(msg)
    open(os.path.join(cfg["paths"]["reports"], "pack.md"), "w", encoding="utf-8").write(msg.replace("### ", "# ") +
                                                                                       "\n")
    append_progress(cfg, msg)


if __name__ == "__main__":
    main()
