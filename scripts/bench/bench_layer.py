"""Benchmark one decoder layer at the real Qwen3.6-35B-A3B size with random weights: frozen experts int4 (or bf16),
k selected bf16 experts with LoRA, shared-expert LoRA. Measures forward (no grad) and recompute+backward time and
peak VRAM for a packed microbatch, plus H2D time of the packed layer."""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from smel.layer_runner import LayerWeights, layer_forward, make_ctx  # noqa: E402
from smel.lora import BlockLoRA  # noqa: E402
from smel.model_io import Dims  # noqa: E402
from smel.moe import BF16Experts, Int4Experts, LayerExperts  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--config", default=None, help="model config.json (default: paths.model_dir/config.json)")
p.add_argument("--tokens", type=int, nargs="+", default=[2048, 4096])
p.add_argument("--seq-len", type=int, default=132)
p.add_argument("--k-sel", type=int, default=12)
p.add_argument("--group", type=int, default=32)
p.add_argument("--frozen", default="int4")
args = p.parse_args()

if args.config is None:
    from smel.pipeline import load_config
    args.config = os.path.join(load_config()["paths"]["model_dir"], "config.json")
d = Dims.from_config(json.load(open(args.config)))
dev, dt = "cuda", torch.bfloat16
torch.manual_seed(0)
H, I, E = d.hidden, d.moe_inter, d.n_experts


def rnd(*s, std=0.02):
    return (torch.randn(*s, device=dev) * std).to(dt)


def dense(ltype):
    W = {"input_layernorm.weight": rnd(H), "post_attention_layernorm.weight": rnd(H),
         "mlp.gate.weight": rnd(E, H, std=0.05), "mlp.shared_expert.gate_proj.weight": rnd(d.shared_inter, H),
         "mlp.shared_expert.up_proj.weight": rnd(d.shared_inter, H),
         "mlp.shared_expert.down_proj.weight": rnd(H, d.shared_inter), "mlp.shared_expert_gate.weight": rnd(1, H)}
    if ltype == "linear_attention":
        kd, vd = d.lin_k_heads * d.lin_k_dim, d.lin_v_heads * d.lin_v_dim
        W.update({"linear_attn.in_proj_qkv.weight": rnd(2 * kd + vd, H), "linear_attn.in_proj_z.weight": rnd(vd, H),
                  "linear_attn.in_proj_b.weight": rnd(d.lin_v_heads, H),
                  "linear_attn.in_proj_a.weight": rnd(d.lin_v_heads, H),
                  "linear_attn.conv1d.weight": rnd(2 * kd + vd, 1, d.conv_k, std=0.3),
                  "linear_attn.A_log": torch.zeros(d.lin_v_heads, device=dev),
                  "linear_attn.dt_bias": torch.ones(d.lin_v_heads, device=dev, dtype=dt),
                  "linear_attn.norm.weight": torch.ones(d.lin_v_dim, device=dev, dtype=dt),
                  "linear_attn.out_proj.weight": rnd(H, vd)})
    else:
        W.update({"self_attn.q_proj.weight": rnd(2 * d.n_heads * d.head_dim, H),
                  "self_attn.k_proj.weight": rnd(d.n_kv * d.head_dim, H),
                  "self_attn.v_proj.weight": rnd(d.n_kv * d.head_dim, H),
                  "self_attn.o_proj.weight": rnd(H, d.n_heads * d.head_dim),
                  "self_attn.q_norm.weight": rnd(d.head_dim), "self_attn.k_norm.weight": rnd(d.head_dim)})
    return W


gu = rnd(E, 2 * I, H)
dn = rnd(E, H, I)
bank = Int4Experts.from_bf16(gu, dn) if args.frozen == "int4" else BF16Experts(gu, dn)
sel = list(range(0, E, E // args.k_sel))[: args.k_sel]
ex = LayerExperts(bank, sel, gu[sel].clone(), dn[sel].clone(), group=args.group)
bank_gb = bank.nbytes() / 2**30
del gu, dn
torch.cuda.empty_cache()
lora = BlockLoRA.build({0: sel}, H, I, d.shared_inter, 16, 32).to(dev)
res = {"frozen_bank_GB": bank_gb}
print(f"frozen bank ({args.frozen}) {bank_gb:.3f} GB, selected {len(sel)} experts")

for ltype in ("linear_attention", "full_attention"):
    LW = LayerWeights(0, ltype, dense(ltype), ex)
    for T in args.tokens:
        n = max(1, T // args.seq_len)
        lens = [args.seq_len] * n
        ctx = make_ctx(lens, d, dev, dt)
        x = torch.randn(sum(lens), H, device=dev, dtype=dt)
        g = torch.randn_like(x)
        for _ in range(2):  # warmup (Triton compile)
            with torch.no_grad():
                layer_forward(x, LW, d, ctx, lora.get(0))
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        t0 = time.perf_counter()
        for _ in range(3):
            with torch.no_grad():
                layer_forward(x, LW, d, ctx, lora.get(0))
        torch.cuda.synchronize()
        tf = (time.perf_counter() - t0) / 3
        t0 = time.perf_counter()
        for _ in range(3):
            xx = x.detach().requires_grad_(True)
            layer_forward(xx, LW, d, ctx, lora.get(0)).backward(g)
        torch.cuda.synchronize()
        tb = (time.perf_counter() - t0) / 3
        peak = (torch.cuda.max_memory_allocated() - base) / 2**30
        print(f"{ltype:17s} T={sum(lens):5d} ({n}x{args.seq_len}): fwd {tf*1e3:7.1f} ms, recompute+bwd {tb*1e3:7.1f} ms,"
              f" activation peak {peak:.2f} GB")
        res[f"{ltype}_T{sum(lens)}"] = {"fwd_ms": tf * 1e3, "bwd_ms": tb * 1e3, "act_peak_GB": peak}

# H2D of a packed layer from pinned memory
cpu = {k: v.cpu().pin_memory() for k, v in bank.tensors().items()}
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(3):
    g_ = {k: v.to(dev, non_blocking=True) for k, v in cpu.items()}
torch.cuda.synchronize()
res["h2d_bank_ms"] = (time.perf_counter() - t0) / 3 * 1e3
print(f"H2D of frozen bank from pinned: {res['h2d_bank_ms']:.1f} ms")
out = os.path.join(os.path.dirname(__file__), "..", "..", "logs", f"bench_layer_{args.frozen}.json")
json.dump(res, open(out, "w"), indent=1)
