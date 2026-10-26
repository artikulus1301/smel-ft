"""Download the original bf16 safetensors of Qwen/Qwen3.6-35B-A3B (~72 GB) with resume; refuses to start when the
disk would be left with less than --keep-free GB.
Usage: python scripts/download.py [--dir <dir, default: paths.model_dir>] [--keep-free 15]
"""
import argparse
import os
import shutil
import sys
import time

from huggingface_hub import HfApi, snapshot_download

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qwen/Qwen3.6-35B-A3B")
ap.add_argument("--dir", default=None)
ap.add_argument("--keep-free", type=float, default=15.0)
ap.add_argument("--retries", type=int, default=100)
args = ap.parse_args()
if args.dir is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from smel.pipeline import load_config
    args.dir = load_config()["paths"]["model_dir"]

os.makedirs(args.dir, exist_ok=True)
info = HfApi().model_info(args.repo, files_metadata=True)
have = 0
for s in info.siblings:
    p = os.path.join(args.dir, s.rfilename)
    if os.path.exists(p) and os.path.getsize(p) == (s.size or 0):
        have += s.size or 0
total = sum(s.size or 0 for s in info.siblings)
free = shutil.disk_usage(args.dir).free
left = free - (total - have)
print(f"total {total/1e9:.1f} GB, already {have/1e9:.1f} GB, free {free/1e9:.1f} GB -> after download "
      f"{left/1e9:.1f} GB free")
if left < args.keep_free * 1e9:
    raise SystemExit(f"refusing: less than {args.keep_free} GB would stay free")
for attempt in range(1, args.retries + 1):
    try:
        snapshot_download(args.repo, local_dir=args.dir, max_workers=4)
        break
    except Exception as e:  # network hiccups: finished files are kept, retry the rest
        print(f"attempt {attempt} failed: {type(e).__name__}: {str(e)[:200]}", flush=True)
        if attempt == args.retries:
            raise
        time.sleep(min(300, 15 * attempt))
print("done:", args.dir)
