"""Where the GPU idles in a torch.profiler chrome trace (e.g. scripts/bench/bench_layer.py --trace trace.json).

    python scripts/bench/trace_gaps.py trace.json [--top 30] [--ours smel]

Prints the window, GPU busy time (union of kernels / copies), and the idle time between GPU events split by the
Python frame the main thread was in during each gap: the innermost frame of our package (--ours) and the innermost
frame at all. Then the GPU time per kernel name and the CPU ops of the window. Idle GPU time inside a step is time
the step waits for the host (kernel launches, .tolist() syncs, Python): the place to look before writing a kernel.
Idea and first version: the steptrace tool of the authors' Soup Kernel Challenge solution (gaps.py there).
"""
import argparse
import bisect
import collections
import json


def analyse(events, ours="smel", top=30, min_gap_us=5.0, out=print):
    gpu = sorted((e["ts"], e["ts"] + e["dur"], e["name"]) for e in events
                 if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset"))
    py = [e for e in events if e.get("ph") == "X" and e.get("cat") == "python_function"]
    cpu_ops = [e for e in events if e.get("ph") == "X" and e.get("cat") == "cpu_op"]
    host = py or cpu_ops
    t0 = min(e["ts"] for e in host)
    t1 = max(e["ts"] + e["dur"] for e in host)
    busy, gaps, end = 0.0, [], t0
    for s, e, _ in gpu:
        if s > end:
            gaps.append((end, s))
        if e > end:
            busy += e - max(s, end)
            end = e
    if t1 > end:
        gaps.append((end, t1))
    total = t1 - t0
    out(f"window {total / 1e3:.1f} ms, GPU busy {busy / 1e3:.1f} ms ({busy / total:.0%}), {len(gpu)} GPU events")
    if not py:
        out("no Python frames in the trace (profile with with_stack=True): no attribution")
        py = [{"tid": 0, "ts": 0, "dur": 0, "name": "?"}]
    main_tid = collections.Counter(e["tid"] for e in py).most_common(1)[0][0]
    py = sorted((e for e in py if e["tid"] == main_tid), key=lambda e: e["ts"])

    gaps = [(s, e) for s, e in gaps if e - s >= min_gap_us]
    mids = [(s + e) / 2 for s, e in gaps]
    covering = [[] for _ in gaps]
    for f in py:  # each frame to the gap midpoints it covers
        lo = bisect.bisect_left(mids, f["ts"])
        hi = bisect.bisect_left(mids, f["ts"] + f["dur"])
        for i in range(lo, hi):
            covering[i].append(f)
    mark = (ours + "/", ours + "\\")
    by_ours, by_inner = collections.Counter(), collections.Counter()
    for (s, e), frames in zip(gaps, covering):
        names = [f["name"] for f in sorted(frames, key=lambda f: f["dur"])]  # innermost first
        by_ours[next((n for n in names if any(m in n for m in mark)), "(not ours)")] += e - s
        by_inner[names[0] if names else "?"] += e - s
    idle = sum(e - s for s, e in gaps)
    out(f"GPU idle {idle / 1e3:.1f} ms in {len(gaps)} gaps >= {min_gap_us:g} us; by innermost frame of {ours}/:")
    for k, v in by_ours.most_common(top):
        out(f"  {v / 1e3:8.1f} ms  {k}")
    out("by innermost frame:")
    for k, v in by_inner.most_common(top):
        out(f"  {v / 1e3:8.1f} ms  {k}")
    per = collections.Counter()
    for s, e, n in gpu:
        per[n[:90]] += e - s
    out("GPU time by kernel:")
    for k, v in per.most_common(top):
        out(f"  {v / 1e3:8.1f} ms  {k}")
    ops = collections.Counter(e["name"] for e in cpu_ops)
    out(f"cpu ops in window: {sum(ops.values())}; top: {ops.most_common(12)}")
    return {"window_ms": total / 1e3, "busy_ms": busy / 1e3, "idle_ms": idle / 1e3, "by_ours": dict(by_ours)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--min-gap-us", type=float, default=5.0)
    ap.add_argument("--ours", default="smel", help="package whose frames the idle time is attributed to")
    args = ap.parse_args()
    events = json.load(open(args.trace, encoding="utf-8"))["traceEvents"]
    analyse(events, args.ours, args.top, args.min_gap_us)


if __name__ == "__main__":
    main()
