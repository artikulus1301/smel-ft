"""Control selections for the ablation: same L and the same number of experts per layer as the router-based
selection, but chosen at random ("random"), or no routed experts at all ("shared": shared-expert LoRA only).
Usage: python scripts/make_selection.py --kind random --seed 0 --out configs/selection_random_s0.yaml
"""
import argparse
import os
import random
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from smel.pipeline import ROOT, load_config  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--base", default=os.path.join(ROOT, "configs", "selection.yaml"))
ap.add_argument("--kind", choices=["random", "shared"], required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
args = ap.parse_args()
base = yaml.safe_load(open(args.base))
cfg = load_config()
stats = torch.load(os.path.join(cfg["paths"]["artifacts"], "routing_stats.pt"))
gm = stats["domain"]["d"]["resp"]["gate_mass"]
gm = gm / gm.sum(-1, keepdim=True)
rng = random.Random(args.seed)
out = dict(base)
out["experts"], cov = {}, {}
for l, ids in base["experts"].items():
    l = int(l)
    new = sorted(rng.sample(range(gm.shape[1]), len(ids))) if args.kind == "random" else []
    out["experts"][l] = new
    cov[l] = round(float(gm[l, new].sum()) if new else 0.0, 4)
out["domain_mass_covered"] = cov
out["variant"] = f"{args.kind}_s{args.seed}" if args.kind == "random" else "shared"
yaml.safe_dump(out, open(args.out, "w", encoding="utf-8"), sort_keys=False, allow_unicode=True)
base_cov = sum(base["domain_mass_covered"].values()) / len(base["domain_mass_covered"])
print(f"{args.out}: {sum(map(len, out['experts'].values()))} experts; mean domain gate mass covered "
      f"{sum(cov.values())/len(cov):.3f} (router-based selection: {base_cov:.3f})")
