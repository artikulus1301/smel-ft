"""Does FLA chunk_gated_delta_rule (varlen) recompile / re-autotune for every new total length T or number of
sequences? Times fwd+bwd for a series of distinct shapes, twice."""
import time, torch, torch.nn.functional as F
from fla.ops.gated_delta_rule import chunk_gated_delta_rule

dev = "cuda"
H, D = 32, 128
shapes = [[132] * 15, [132] * 31, [100, 900, 77], [500] * 7, [131] * 16, [133] * 30, [64] * 3, [954, 12, 300, 77, 1500]]
for rnd in range(2):
    for lens in shapes:
        T = sum(lens)
        q, k, v = [torch.randn(1, T, H, D, device=dev, dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
        g = (-F.softplus(torch.randn(1, T, H, device=dev)) * 0.5)
        beta = torch.rand(1, T, H, device=dev, dtype=torch.bfloat16)
        cu = torch.tensor([0] + lens, device=dev).cumsum(0).to(torch.int32)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        o, _ = chunk_gated_delta_rule(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True, cu_seqlens=cu)
        o.float().sum().backward()
        torch.cuda.synchronize()
        print(f"round {rnd} n_seq={len(lens):3d} T={T:5d}: {1e3*(time.perf_counter()-t0):8.1f} ms")
