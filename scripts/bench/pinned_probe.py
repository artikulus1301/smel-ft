"""Find how much pinned host memory can be allocated: one block vs many 1 GB chunks (each mode in a fresh process)."""
import sys, time, torch

mode = sys.argv[1]
if mode == "single":
    gb = float(sys.argv[2])
    try:
        b = torch.empty(int(gb * 2**30), dtype=torch.uint8, pin_memory=True)
        print("OK", gb)
    except Exception as e:
        print("FAIL", gb, type(e).__name__)
else:
    chunks = []
    for i in range(int(sys.argv[2])):
        try:
            chunks.append(torch.empty(2**30, dtype=torch.uint8, pin_memory=True))
        except Exception as e:
            print("FAIL at chunk", i)
            break
    print("chunks_ok_GB", len(chunks))
    x = torch.ones(1024, device="cuda")
    print("cuda still usable", float(x.sum()))
