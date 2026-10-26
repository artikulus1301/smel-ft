"""Ablation driver: trains and evaluates the control variants one after another (lr 2e-4, 3 epochs).
Every step is skipped when its output already exists, so the driver can simply be restarted.
Selections: configs/selection_domain_L20.yaml (= select_experts.py output), the others from scripts/make_selection.py:
  python scripts/make_selection.py --base configs/selection_domain_L20.yaml --kind random --seed 0 --out configs/selection_random_s0.yaml
  python scripts/make_selection.py --base configs/selection_domain_L20.yaml --kind shared --out configs/selection_shared.yaml
  python scripts/select_experts.py --set select.L_override=16 --out configs/selection_domain_L16.yaml
Usage: python examples/persona_eval/run_ablation.py
"""
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from smel.pipeline import load_config  # noqa: E402

WORK = load_config()["paths"]["work_dir"]
PY = sys.executable
C = os.path.join(ROOT, "configs")

RUNS = [  # tag, selection file, seed
    ("L20_random_s0", "selection_random_s0.yaml", 0),
    ("L20_shared_s0", "selection_shared.yaml", 0),
    ("L20_domain_s1", "selection_domain_L20.yaml", 1),
    ("L20_random_s1", "selection_random_s1.yaml", 1),
    ("L20_shared_s1", "selection_shared.yaml", 1),
    ("L16_domain_s0", "selection_domain_L16.yaml", 0),
]


def run(args, log):
    t0 = time.time()
    with open(os.path.join(ROOT, "logs", log), "w", encoding="utf-8") as f:
        r = subprocess.run([PY] + args, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                           env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    print(f"  {' '.join(args[:2])} -> exit {r.returncode} in {(time.time() - t0) / 60:.1f} min", flush=True)
    if r.returncode:
        raise SystemExit(f"failed: {args} (see logs/{log})")


for tag, sel, seed in RUNS:
    adapter = os.path.join(ROOT, "artifacts", f"adapter_{tag}")
    print(f"== {tag}", flush=True)
    if tag.startswith("L16"):
        if not os.path.exists(os.path.join(WORK, "cache_L16", "meta.json")):
            run(["scripts/cache.py", "--L", "16"], "cache_L16.log")
        if not os.path.exists(os.path.join(WORK, "packed_L16", "meta.json")):
            run(["scripts/pack.py", "--selection", os.path.join(C, sel)], "pack_L16.log")
    if not os.path.exists(os.path.join(adapter, "adapter.safetensors")):
        run(["scripts/train.py", "--tag", tag, "--selection", os.path.join(C, sel), "--set", "train.lr=2e-4",
             f"train.seed={seed}"], f"train_{tag}_run.log")
    if not os.path.exists(os.path.join(adapter, "eval_ppl.json")):
        run(["scripts/eval.py", "--adapter", adapter], f"eval_{tag}.log")
print("ablation done", flush=True)
