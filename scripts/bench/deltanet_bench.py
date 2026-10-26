"""Compare FLA chunk_gated_delta_rule against the pure-torch reference (the original undecorated function)."""
import time, inspect, torch, torch.nn.functional as F
import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as M
from fla.ops.gated_delta_rule import chunk_gated_delta_rule

hf_fn = M.torch_chunk_gated_delta_rule
print("HF symbol resolves to:", getattr(hf_fn, "__module__", "?"), getattr(hf_fn, "__name__", "?"))
ref = getattr(hf_fn, "__wrapped__", None) or getattr(hf_fn, "fallback", None)
if ref is None:
    # re-exec the function source without decorators to get the pure torch version
    src = inspect.getsource(M).split("def torch_chunk_gated_delta_rule(")[1].split("\n@use_kernel_func_from_hub")[0]
    ns = dict(M.__dict__)
    exec("def torch_chunk_gated_delta_rule(" + src, ns)
    ref = ns["torch_chunk_gated_delta_rule"]
print("reference:", ref.__module__, ref.__name__)

dev = "cuda"
torch.manual_seed(0)
for T in [512, 2048, 4096]:
    B, Hh, D = 1, 32, 128
    q, k, v = [torch.randn(B, T, Hh, D, device=dev, dtype=torch.bfloat16) for _ in range(3)]
    g = -F.softplus(torch.randn(B, T, Hh, device=dev)) * 0.5
    beta = torch.rand(B, T, Hh, device=dev, dtype=torch.bfloat16)
    o_f, _ = chunk_gated_delta_rule(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True)
    o_t, _ = ref(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True)
    rel = ((o_f.float() - o_t.float()).norm() / o_t.float().norm()).item()
    row = [f"T={T} rel_err={rel:.2e}"]
    for name, fn in [("fla", chunk_gated_delta_rule), ("torch", ref)]:
        qq, kk, vv = [t.clone().requires_grad_() for t in (q, k, v)]
        def fb():
            o, _ = fn(qq, kk, vv, g=g, beta=beta, use_qk_l2norm_in_kernel=True)
            o.float().sum().backward()
        fb(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        for _ in range(3): fb()
        torch.cuda.synchronize()
        row.append(f"{name} {(time.perf_counter()-t)/3*1e3:.1f}ms peak {torch.cuda.max_memory_allocated()/2**30:.2f}GB")
    print(" | ".join(row))

# varlen: two packed sequences must equal separate runs
T1, T2 = 300, 500
q, k, v = [torch.randn(1, T1 + T2, 32, 128, device=dev, dtype=torch.bfloat16) for _ in range(3)]
g = -F.softplus(torch.randn(1, T1 + T2, 32, device=dev)) * 0.5
beta = torch.rand(1, T1 + T2, 32, device=dev, dtype=torch.bfloat16)
cu = torch.tensor([0, T1, T1 + T2], device=dev, dtype=torch.int32)
o_p, _ = chunk_gated_delta_rule(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True, cu_seqlens=cu)
o_a, _ = chunk_gated_delta_rule(q[:, :T1], k[:, :T1], v[:, :T1], g=g[:, :T1], beta=beta[:, :T1], use_qk_l2norm_in_kernel=True)
o_b, _ = chunk_gated_delta_rule(q[:, T1:], k[:, T1:], v[:, T1:], g=g[:, T1:], beta=beta[:, T1:], use_qk_l2norm_in_kernel=True)
sep = torch.cat([o_a, o_b], 1)
print("varlen packed vs separate rel err:", ((o_p.float() - sep.float()).norm() / sep.float().norm()).item())
