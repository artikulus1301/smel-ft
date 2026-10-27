"""Create a tiny random MoE checkpoint of the same class as the real model (Qwen3.5/3.6-MoE or Gemma 4) with the
real vocabulary, and a config to dry-run the whole pipeline (probe -> select -> cache -> pack -> train -> eval ->
merge) before touching the real model.
Usage: python scripts/make_tiny_e2e.py --domain <chat jsonl> --general <jsonl used as 'general' text>
           [--arch qwen3_5_moe|gemma4] [--tokenizer <dir|hub id>]
Then run every stage with --config configs/tiny_e2e.yaml (or configs/tiny_e2e_gemma4.yaml).
"""
import argparse
import os
import shutil
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from smel.pipeline import ROOT, load_config  # noqa: E402
from smel.tiny import build_tiny_gemma4, build_tiny_model, save_tiny_checkpoint  # noqa: E402

TOKENIZERS = {"qwen3_5_moe": ("Qwen/Qwen3.6-35B-A3B", 248320), "gemma4": ("google/gemma-4-26B-A4B-it", 262144)}

ap = argparse.ArgumentParser()
ap.add_argument("--arch", choices=list(TOKENIZERS), default="qwen3_5_moe")
ap.add_argument("--root", default=None)
ap.add_argument("--general", required=True)
ap.add_argument("--domain", default=None, help="default: paths.domain_data")
ap.add_argument("--tokenizer", default=None, help="local dir or hub id (only tokenizer files are fetched)")
args = ap.parse_args()
tag = "" if args.arch == "qwen3_5_moe" else "_" + args.arch
root = args.root or os.path.join(ROOT, "work", "tiny_e2e" + tag)
tok_src = args.tokenizer or TOKENIZERS[args.arch][0]
vocab = TOKENIZERS[args.arch][1]

if args.arch == "gemma4":
    model = build_tiny_gemma4(seed=0, dtype=torch.bfloat16, device="cuda", vocab=vocab, n_layers=6, n_experts=16,
                              top_k=4)
    candidates, k_max = [2, 4], 4
else:
    model = build_tiny_model(seed=0, dtype=torch.bfloat16, device="cuda", vocab=vocab, n_layers=8, n_experts=32,
                             top_k=4)
    candidates, k_max = [4, 6], 4
mdir = os.path.join(root, "model")
save_tiny_checkpoint(model, mdir, n_shards=4)
cfg = load_config()
# the real tokenizer / chat template (only these small files are needed, not the weights)
for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"):
    if os.path.isdir(tok_src):
        src = os.path.join(tok_src, f)
        if not os.path.exists(src):
            continue
    else:
        from huggingface_hub import hf_hub_download
        src = hf_hub_download(tok_src, f)
    shutil.copy(src, mdir)
cfg["paths"].update({"model_dir": mdir, "work_dir": os.path.join(root, "work"), "general_data": args.general,
                     "domain_data": args.domain or cfg["paths"]["domain_data"],
                     "artifacts": os.path.join(root, "artifacts"), "reports": os.path.join(root, "reports"),
                     "logs": os.path.join(root, "logs"), "tokenizer": mdir,
                     "progress": os.path.join(root, "progress.md"),
                     "selection": os.path.join(root, "selection.yaml")})
cfg["probe"].update({"cache_candidates": candidates, "max_domain_tokens": 60000, "max_general_tokens": 40000})
cfg["train"].update({"epochs": 1, "val_every": 3, "monitor_every": 3, "ckpt_every": 4})
cfg["select"]["k_max"] = k_max
out = os.path.join(ROOT, "configs", f"tiny_e2e{tag}.yaml")
yaml.safe_dump(cfg, open(out, "w"), sort_keys=False)
print("checkpoint:", mdir, "\nconfig:", out)
