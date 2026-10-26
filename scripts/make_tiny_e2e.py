"""Create a tiny random qwen3_5_moe checkpoint with the real vocabulary and a config to dry-run the whole pipeline
(probe -> select -> cache -> pack -> train -> eval -> merge) before touching the real model.
Usage: python scripts/make_tiny_e2e.py --domain <chat jsonl> --general <jsonl used as 'general' text> [--tokenizer <dir|hub id>]
Then run every stage with --config configs/tiny_e2e.yaml.
"""
import argparse
import os
import shutil
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from smel.pipeline import ROOT, load_config  # noqa: E402
from smel.tiny import build_tiny_model, save_tiny_checkpoint  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=os.path.join(ROOT, "work", "tiny_e2e"))
ap.add_argument("--general", required=True)
ap.add_argument("--domain", default=None, help="default: paths.domain_data")
ap.add_argument("--tokenizer", default="Qwen/Qwen3.6-35B-A3B", help="local dir or hub id (only tokenizer files are fetched)")
args = ap.parse_args()

model = build_tiny_model(seed=0, dtype=torch.bfloat16, device="cuda", vocab=248320, n_layers=8, n_experts=32,
                         top_k=4)
mdir = os.path.join(args.root, "model")
save_tiny_checkpoint(model, mdir, n_shards=4)
cfg = load_config()
# the real tokenizer / chat template (only these small files are needed, not the weights)
for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"):
    if os.path.isdir(args.tokenizer):
        src = os.path.join(args.tokenizer, f)
        if not os.path.exists(src):
            continue
    else:
        from huggingface_hub import hf_hub_download
        src = hf_hub_download(args.tokenizer, f)
    shutil.copy(src, mdir)
cfg["paths"].update({"model_dir": mdir, "work_dir": os.path.join(args.root, "work"), "general_data": args.general,
                     "domain_data": args.domain or cfg["paths"]["domain_data"],
                     "artifacts": os.path.join(args.root, "artifacts"), "reports": os.path.join(args.root, "reports"),
                     "logs": os.path.join(args.root, "logs"),
                     "tokenizer": mdir, "progress": os.path.join(args.root, "progress.md")})
cfg["probe"].update({"cache_candidates": [4, 6], "max_domain_tokens": 60000, "max_general_tokens": 40000})
cfg["train"].update({"epochs": 1, "val_every": 3, "monitor_every": 3, "ckpt_every": 4})
cfg["select"]["k_max"] = 4
out = os.path.join(ROOT, "configs", "tiny_e2e.yaml")
yaml.safe_dump(cfg, open(out, "w"), sort_keys=False)
print("checkpoint:", mdir, "\nconfig:", out)
