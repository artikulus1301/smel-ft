"""Stage-0 micro-benchmarks: pinned memory, H2D bandwidth, matmul, small experts, DeltaNet kernels, disk read."""
import argparse, os, sys, time, json
import torch
import torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument("--pinned-gb", type=float, nargs="+", default=[1, 4, 8])
p.add_argument("--disk-file", default=None)
p.add_argument("--out", default=None)
args = p.parse_args()
res = {}
dev = "cuda"


def sync_time(fn, iters=10, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters


# ---------------- pinned memory + H2D
free, total = torch.cuda.mem_get_info()
res["vram_free_gb_at_start"] = free / 2**30
res["vram_total_gb"] = total / 2**30
print(f"VRAM free {free/2**30:.2f} / {total/2**30:.2f} GB")

pinned_ok = {}
for gb in args.pinned_gb:
    try:
        t = time.perf_counter()
        buf = torch.empty(int(gb * 2**30), dtype=torch.uint8, pin_memory=True)
        pinned_ok[gb] = round(time.perf_counter() - t, 2)
        del buf
    except Exception as e:
        pinned_ok[gb] = f"FAIL {type(e).__name__}: {str(e)[:100]}"
    print("pinned", gb, "GB ->", pinned_ok[gb])
res["pinned_alloc_s"] = pinned_ok

h2d = {}
for mb in [256, 512, 1024]:
    n = mb * 2**20
    src = torch.empty(n, dtype=torch.uint8, pin_memory=True)
    dst = torch.empty(n, dtype=torch.uint8, device=dev)
    dt = sync_time(lambda: dst.copy_(src, non_blocking=True), iters=10)
    h2d[mb] = n / dt / 1e9
    # pageable
    srcp = torch.empty(n, dtype=torch.uint8)
    dtp = sync_time(lambda: dst.copy_(srcp), iters=5, warmup=1)
    h2d[f"{mb}_pageable"] = n / dtp / 1e9
    del src, dst, srcp
    torch.cuda.empty_cache()
print("H2D GB/s", {k: round(v, 2) for k, v in h2d.items()})
res["h2d_GBps"] = h2d

# ---------------- dense bf16 matmul
mm = {}
for n in [2048, 4096]:
    a = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
    b = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
    dt = sync_time(lambda: a @ b, iters=20)
    mm[n] = 2 * n**3 / dt / 1e12
print("bf16 matmul TFLOPS", {k: round(v, 1) for k, v in mm.items()})
res["bf16_matmul_TFLOPS"] = mm

# ---------------- small experts: tokens x 2048 -> 512 (gate_up = 1024 out), down 512 -> 2048
se = {}
H, I = 2048, 512
wgu = torch.randn(2 * I, H, device=dev, dtype=torch.bfloat16) * 0.02
wd = torch.randn(H, I, device=dev, dtype=torch.bfloat16) * 0.02
for ntok in [16, 64, 256, 1024, 4096]:
    x = torch.randn(ntok, H, device=dev, dtype=torch.bfloat16)

    def f():
        g, u = F.linear(x, wgu).chunk(2, -1)
        return F.linear(F.silu(g) * u, wd)

    dt = sync_time(f, iters=50)
    flops = 2 * ntok * H * 3 * I
    se[ntok] = {"us": dt * 1e6, "TFLOPS": flops / dt / 1e12}
print("small expert", {k: (round(v["us"], 1), round(v["TFLOPS"], 2)) for k, v in se.items()})
res["small_expert"] = se

# full MoE layer emulation: 256 experts, top-8, T tokens, looped (like HF eager)
E, K = 256, 8
W_gu = torch.randn(E, 2 * I, H, device=dev, dtype=torch.bfloat16) * 0.02
W_d = torch.randn(E, H, I, device=dev, dtype=torch.bfloat16) * 0.02
for T in [2048, 4096]:
    x = torch.randn(T, H, device=dev, dtype=torch.bfloat16)
    idx = torch.randint(0, E, (T, K), device=dev)
    w = torch.rand(T, K, device=dev, dtype=torch.bfloat16)

    def moe():
        out = torch.zeros_like(x)
        flat = idx.reshape(-1)
        order = flat.argsort()
        counts = torch.bincount(flat, minlength=E).tolist()
        tok = order // K
        xs = x[tok]
        ws = w.reshape(-1)[order]
        start = 0
        outs = []
        for e in range(E):
            c = counts[e]
            if c == 0:
                continue
            xe = xs[start:start + c]
            g, u = F.linear(xe, W_gu[e]).chunk(2, -1)
            outs.append(F.linear(F.silu(g) * u, W_d[e]))
            start += c
        y = torch.cat(outs) * ws[:, None]
        out.index_add_(0, tok, y)
        return out

    dt = sync_time(moe, iters=5, warmup=2)
    res.setdefault("moe_layer_bf16_ms", {})[T] = dt * 1e3
    print(f"MoE layer (256e top8, sorted loop) T={T}: {dt*1e3:.1f} ms")
del W_gu, W_d
torch.cuda.empty_cache()

# ---------------- DeltaNet: FLA vs torch fallback
try:
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import torch_chunk_gated_delta_rule
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    B, T, Hh, D = 1, 2048, 32, 128
    torch.manual_seed(0)
    q = torch.randn(B, T, Hh, D, device=dev, dtype=torch.bfloat16)
    k = torch.randn(B, T, Hh, D, device=dev, dtype=torch.bfloat16)
    v = torch.randn(B, T, Hh, D, device=dev, dtype=torch.bfloat16)
    g = -F.softplus(torch.randn(B, T, Hh, device=dev)) * 0.5
    beta = torch.rand(B, T, Hh, device=dev, dtype=torch.bfloat16)
    t0 = time.perf_counter()
    o_f, _ = chunk_gated_delta_rule(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True)
    torch.cuda.synchronize()
    res["fla_first_call_s"] = time.perf_counter() - t0
    o_t, _ = torch_chunk_gated_delta_rule(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True)
    rel = ((o_f.float() - o_t.float()).norm() / o_t.float().norm()).item()
    res["fla_vs_torch_rel_err"] = rel
    print(f"FLA first call {res['fla_first_call_s']:.1f}s, rel err vs torch {rel:.2e}")
    for name, fn in [("fla", chunk_gated_delta_rule), ("torch", torch_chunk_gated_delta_rule)]:
        qq, kk, vv = [t.clone().requires_grad_() for t in (q, k, v)]

        def fb():
            o, _ = fn(qq, kk, vv, g=g, beta=beta, use_qk_l2norm_in_kernel=True)
            o.float().sum().backward()

        torch.cuda.reset_peak_memory_stats()
        dt = sync_time(fb, iters=5, warmup=2)
        res[f"deltanet_{name}_fwdbwd_ms_T2048"] = dt * 1e3
        res[f"deltanet_{name}_peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
        print(f"DeltaNet {name} fwd+bwd T=2048: {dt*1e3:.1f} ms, peak {res[f'deltanet_{name}_peak_mem_gb']:.2f} GB")
except Exception as e:
    res["deltanet_error"] = repr(e)
    print("DeltaNet bench failed:", repr(e))

# ---------------- disk read
if args.disk_file and os.path.exists(args.disk_file):
    sz = os.path.getsize(args.disk_file)
    t = time.perf_counter()
    n = 0
    with open(args.disk_file, "rb", buffering=0) as f:
        while True:
            b = f.read(64 * 2**20)
            if not b:
                break
            n += len(b)
    dt = time.perf_counter() - t
    res["disk_read_GBps"] = n / dt / 1e9
    res["disk_read_file_gb"] = sz / 1e9
    print(f"disk seq read {n/dt/1e9:.2f} GB/s over {sz/1e9:.1f} GB (may be partly OS-cached)")

if args.out:
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1, default=str)
