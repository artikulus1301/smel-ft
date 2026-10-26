"""Check whether torch._grouped_mm works on this GPU and compare it with a per-expert loop."""
import time, torch, torch.nn.functional as F

dev = "cuda"
E, H, I, K = 256, 2048, 512, 8
W_gu = torch.randn(E, 2 * I, H, device=dev, dtype=torch.bfloat16) * 0.02
for T in [2048, 4096]:
    x = torch.randn(T * K, H, device=dev, dtype=torch.bfloat16)
    counts = torch.bincount(torch.randint(0, E, (T * K,), device=dev), minlength=E)
    offs = counts.cumsum(0).to(torch.int32)
    try:
        y = torch._grouped_mm(x, W_gu.transpose(1, 2), offs=offs)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(10):
            y = torch._grouped_mm(x, W_gu.transpose(1, 2), offs=offs)
        torch.cuda.synchronize()
        gm = (time.perf_counter() - t) / 10 * 1e3
        # loop reference
        cl = [0] + offs.tolist()
        ref = torch.cat([x[cl[e]:cl[e + 1]] @ W_gu[e].T for e in range(E)])
        err = ((y.float() - ref.float()).norm() / ref.float().norm()).item()
        print(f"T={T}: grouped_mm gate_up {gm:.2f} ms, rel err vs loop {err:.1e}")
    except Exception as e:
        print("grouped_mm unavailable:", repr(e)[:300])
        break
    t = time.perf_counter()
    for _ in range(3):
        cl = [0] + offs.tolist()
        ref = torch.cat([x[cl[e]:cl[e + 1]] @ W_gu[e].T for e in range(E)])
    torch.cuda.synchronize()
    print(f"T={T}: loop gate_up {(time.perf_counter()-t)/3*1e3:.2f} ms")
