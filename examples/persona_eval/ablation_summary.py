"""Perplexity table of all evaluated adapters (artifacts/adapter_*/eval_ppl.json), grouped by variant."""
import glob
import json
import math
import os
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NAMES = {"L20_lr2e-4": ("A: router-selected experts, L=20", 0), "L20_domain_s1": ("A: router-selected experts, L=20", 1),
         "L20_random_s0": ("B: random experts, L=20", 0), "L20_random_s1": ("B: random experts, L=20", 1),
         "L20_shared_s0": ("C: shared experts only, L=20", 0), "L20_shared_s1": ("C: shared experts only, L=20", 1),
         "L16_domain_s0": ("D: router-selected experts, L=16", 0)}
rows = {}
base = None
for p in sorted(glob.glob(os.path.join(ROOT, "artifacts", "adapter_*", "eval_ppl.json"))):
    tag = os.path.basename(os.path.dirname(p))[len("adapter_"):]
    if tag not in NAMES:
        continue
    s = json.load(open(p))
    base = s
    name, seed = NAMES[tag]
    rows.setdefault(name, []).append((seed, math.exp(s["domain"]["adapter_loss"]), 100 * s["general"]["ppl_change"]))
print(f"before: domain ppl {math.exp(base['domain']['base_loss']):.2f}, general {math.exp(base['general']['base_loss']):.3f}")
print("| variant | seeds | domain ppl (per seed) | mean | general ppl, Δ% |")
print("|---|---|---|---|---|")
for name, r in rows.items():
    r.sort()
    dom = [x[1] for x in r]
    print(f"| {name} | {len(r)} | {', '.join(f'{x:.2f}' for x in dom)} | {statistics.mean(dom):.2f} | "
          f"{', '.join(f'{x[2]:+.1f}' for x in r)} |")
