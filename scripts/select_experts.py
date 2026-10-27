"""Stage 3: choose the cut layer L and the trainable experts per layer; estimate VRAM / RAM / time.

Output: paths.selection (configs/selection.yaml) and reports/select.md.
Usage: python scripts/select_experts.py [--set select.L_override=30 select.k_max=12]
"""
import os
import sys

# this file is named like the stdlib module `select`; keep scripts/ off sys.path so `import select` stays stdlib
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != os.path.dirname(os.path.abspath(__file__))]
import argparse  # noqa: E402
import json  # noqa: E402

import torch  # noqa: E402
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from smel.model_io import Dims  # noqa: E402
from smel.pipeline import ROOT, append_progress, load_config, parse_overrides  # noqa: E402

GB = 2**30


def divergences(stats, view):
    eps = 1e-12
    P, GM = {}, {}
    for name in ("domain", "general"):
        s = stats[name]
        v = view if s["ntok"][view].sum() > 0 else "all"
        p = (s["d"][v]["prob_sum"] / s["ntok"][v][:, None].clamp_min(1)).clamp_min(eps)
        P[name] = p / p.sum(-1, keepdim=True)
        gm = s["d"][v]["gate_mass"]
        GM[name] = gm / gm.sum(-1, keepdim=True).clamp_min(eps)
    p, q = P["domain"], P["general"]
    kl = (p * (p / q).log()).sum(-1)
    m = 0.5 * (p + q)
    js = 0.5 * (p * (p / m).log()).sum(-1) + 0.5 * (q * (q / m).log()).sum(-1)
    return kl, js, GM


def choose_L(D, coverage):
    """Largest L such that layers >= L hold at least `coverage` of the total divergence (smallest block)."""
    tail = D.flip(0).cumsum(0).flip(0) / D.sum()
    ok = [l for l in range(len(D)) if tail[l] >= coverage]
    return max(ok), tail


def choose_experts(gm_l, mass_p, k_max):
    srt = gm_l.sort(descending=True)
    k = int((srt.values.cumsum(0) < mass_p).sum()) + 1
    k = min(k, k_max)
    return sorted(srt.indices[:k].tolist()), float(srt.values[:k].sum())


def estimate(d: Dims, L, sel, cfg, tokens_per_step, mb_tokens, n_train_tokens, bench=None, dense_bytes=None):
    """Memory/time estimate from tensor shapes (bytes) and the measured layer benchmark. dense_bytes[l] = bf16
    bytes of everything in layer l except the routed experts (read from the checkpoint)."""
    H, I, E, g = d.hidden, d.moe_inter, d.n_experts, cfg["model"]["quant_group"]
    r = cfg["train"]["r"]
    n_block = d.n_layers - L
    int4_bank = E * (3 * H * I) * 0.5 + E * (3 * H * I) / g * 2 * 2  # nibbles + bf16 scale & zero
    layer_bytes, sel_bytes, lora_params = {}, 0, 0
    for l in range(L, d.n_layers):
        k = len(sel[l])
        sb = k * 3 * H * I * 2
        sel_bytes += sb
        layer_bytes[l] = dense_bytes[l] + int4_bank + sb
        lora_params += k * 3 * r * (H + I) + (3 * r * (H + d.shared_inter) if cfg["train"]["shared_lora"] else 0)
    block = sum(layer_bytes.values())
    head = d.vocab * H * (1 if cfg["model"]["lm_head_int8"] else 2) + d.vocab * 2 + H * 2
    lora_state = lora_params * 4 * 4  # fp32 param + grad + 2 Adam moments
    act = (bench or {}).get("act_gb_per_1k", 0.4) * mb_tokens / 1000 * GB
    ce_tile = 1024 * 32768 * 4 * 3  # token chunk x vocab chunk fp32 logits/probs
    ckpt = n_block * tokens_per_step * H * 2
    ctx = 0.35 * GB  # CUDA context / allocator slack
    max_layer = max(layer_bytes.values())
    vram_resident = block + head + lora_state + act + ce_tile + ckpt + ctx
    vram_stream = 2 * max_layer + head + lora_state + act + ce_tile + ckpt + ctx
    # time: measured per-token cost of fwd + recompute/bwd per layer, plus two loads per layer per step
    ms_tok = (bench or {}).get("ms_per_token_layer", 0.08)
    steps = max(1, round(n_train_tokens / tokens_per_step))
    load_s = (bench or {}).get("h2d_s_per_layer", 0.035)
    t_step_res = tokens_per_step * n_block * ms_tok / 1e3
    t_step_stream = t_step_res + 2 * n_block * load_s
    return {"L": L, "n_block_layers": n_block, "n_trainable_experts": sum(len(v) for v in sel.values()),
            "lora_params_M": lora_params / 1e6, "block_GB": block / GB, "selected_bf16_GB": sel_bytes / GB,
            "int4_bank_per_layer_GB": int4_bank / GB, "lm_head_GB": head / GB, "act_GB": act / GB,
            "ckpt_GB": ckpt / GB, "vram_resident_GB": vram_resident / GB, "vram_stream_GB": vram_stream / GB,
            "pinned_GB_if_stream": block / GB, "steps_per_epoch": steps,
            "epoch_min_resident": steps * t_step_res / 60, "epoch_min_stream": steps * t_step_stream / 60}


def load_bench():
    p = os.path.join(ROOT, "logs", "bench_layer_int4.json")
    if not os.path.exists(p):
        return None
    b = json.load(open(p))
    keys = [k for k in b if k.endswith("T4092")]
    if not keys:
        return None
    ms = sum(b[k]["fwd_ms"] + b[k]["bwd_ms"] for k in keys) / len(keys)
    act = max(b[k]["act_peak_GB"] for k in keys)
    return {"ms_per_token_layer": ms / 4092, "act_gb_per_1k": act / 4.092, "h2d_s_per_layer": b["h2d_bank_ms"] / 1e3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--stats", default=None)
    ap.add_argument("--out", default=None, help="default: paths.selection")
    args = ap.parse_args()
    cfg = load_config(args.config, parse_overrides(args.set))
    sc = cfg["select"]
    args.out = args.out or cfg["paths"]["selection"]
    raw = torch.load(args.stats or os.path.join(cfg["paths"]["artifacts"], "routing_stats.pt"))
    d = Dims(**raw["dims"])
    kl, js, GM = divergences(raw, sc["stats_view"])
    D = kl if sc["divergence"] == "kl" else js
    L_auto, tail = choose_L(D, sc["coverage"])
    L = int(sc["L_override"]) if sc.get("L_override") is not None else L_auto
    tok_path = os.path.join(cfg["paths"]["work_dir"], "tokens_domain.npz")
    import numpy as np
    z = np.load(tok_path)
    split = z["split"]
    lens = np.diff(z["offsets"])
    n_train = int(lens[split == "train"].sum())
    bench = load_bench() if d.arch == "qwen3_5_moe" else None   # the layer benchmark is Qwen-specific
    from smel.model_io import Checkpoint
    ck = Checkpoint(cfg["paths"]["model_dir"])
    dense_b = {}
    for l in range(d.n_layers):
        names = [n for n in ck.layer_names(l) if "experts." not in n]
        dense_b[l] = sum(2 * int(torch.tensor(ck.shape(n)).prod()) for n in names)
    k_max = sc["k_max"]
    notes = []
    while True:
        sel, cov = {}, {}
        for l in range(L, d.n_layers):
            sel[l], cov[l] = choose_experts(GM["domain"][l], sc["mass_p"], k_max)
        est = estimate(d, L, sel, cfg, cfg["train"]["tokens_per_step"], cfg["train"]["mb_tokens"], n_train, bench, dense_b)
        fits_res = est["vram_resident_GB"] <= sc["vram_budget_gb"]
        fits_stream = est["vram_stream_GB"] <= sc["vram_budget_gb"] and est["pinned_GB_if_stream"] <= sc["pin_budget_gb"]
        fits_disk_stream = est["vram_stream_GB"] <= sc["vram_budget_gb"]
        if fits_res or fits_stream or fits_disk_stream:
            break
        if k_max > 4:
            k_max -= 2
            notes.append(f"VRAM estimate over budget -> k_max={k_max}")
        elif L < d.n_layers - 1:
            L += 1
            notes.append(f"still over budget -> L={L}")
        else:
            break
    mode = "resident" if fits_res else ("stream_pinned" if fits_stream else "stream_disk")
    out = {"L": L, "L_auto": L_auto, "mode": mode, "k_max": k_max, "mass_p": sc["mass_p"],
           "divergence": sc["divergence"], "experts": {int(l): v for l, v in sel.items()},
           "domain_mass_covered": {int(l): round(c, 4) for l, c in cov.items()},
           "estimate": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in est.items()}}
    with open(args.out, "w", encoding="utf-8") as f:
        yaml.safe_dump(out, f, sort_keys=False, allow_unicode=True)
    md = ["# Select: cut layer and experts\n",
          f"Divergence: {sc['divergence']} on {sc['stats_view']} tokens, coverage threshold {sc['coverage']}.",
          f"Rule: the largest L such that layers >= L hold >= {sc['coverage']:.0%} of sum D_l (the smallest block "
          f"that keeps most of the domain-specific routing).\n",
          f"- L from the data: **{L_auto}** (share of D in layers >= L: {tail[L_auto]:.2f})",
          f"- final L: **{L}**" + (" (set manually)" if sc.get("L_override") is not None else ""),
          f"- mode: **{mode}**, k_max={k_max}, mass_p={sc['mass_p']}",
          f"- trainable experts: **{est['n_trainable_experts']}** + {est['n_block_layers']} shared; "
          f"LoRA params {est['lora_params_M']:.1f}M",
          f"- block (int4 + selected bf16 + dense bf16): {est['block_GB']:.2f} GB; lm_head {est['lm_head_GB']:.2f} GB",
          f"- VRAM: resident {est['vram_resident_GB']:.2f} GB, streaming {est['vram_stream_GB']:.2f} GB "
          f"(budget {sc['vram_budget_gb']})",
          f"- steps per epoch {est['steps_per_epoch']}, epoch ~ {est['epoch_min_resident']:.1f} min (resident) / "
          f"{est['epoch_min_stream']:.1f} min (streaming from pinned RAM)"]
    md += [f"- {n}" for n in notes]
    md += ["\n| layer | D_l (KL) | JS | share of D in layers >= l | experts | domain mass covered | expert ids |",
           "|---|---|---|---|---|---|---|"]
    for l in range(d.n_layers):
        s = sel.get(l)
        md.append(f"| {l} | {kl[l]:.4f} | {js[l]:.4f} | {tail[l]:.2f} | {len(s) if s else '-'} | "
                  f"{cov[l]:.2f} | {s if s else ''} |" if s else
                  f"| {l} | {kl[l]:.4f} | {js[l]:.4f} | {tail[l]:.2f} | - | - | |")
    open(os.path.join(cfg["paths"]["reports"], "select.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md[:12]))
    append_progress(cfg, f"### select done\n- L_auto={L_auto}, L={L}, mode={mode}, experts={est['n_trainable_experts']},"
                         f" VRAM res/stream {est['vram_resident_GB']:.2f}/{est['vram_stream_GB']:.2f} GB, epoch ≈ "
                         f"{est['epoch_min_resident']:.1f}/{est['epoch_min_stream']:.1f} min")


if __name__ == "__main__":
    main()
