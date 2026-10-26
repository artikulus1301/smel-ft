"""Stage 4: cache the input of layer L for every domain (train/val/eval) and general sequence.

If probe already saved this L, the files are hard-linked (no extra disk). Otherwise layers 0..L-1 are run
layer by layer in bf16. Output: work_dir/cache_L{L}/h_{domain,general}.npy + meta.json.
Usage: python scripts/cache.py [--L 30]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from smel.bottom import run_bottom, sets_signature  # noqa: E402
from smel.pipeline import ROOT, append_progress, build_token_sets, load_config, parse_overrides  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--L", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config, parse_overrides(args.set))
    L = args.L if args.L is not None else yaml.safe_load(open(os.path.join(ROOT, "configs", "selection.yaml")))["L"]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["paths"]["tokenizer"])
    sets = build_token_sets(cfg, tok)
    work = cfg["paths"]["work_dir"]
    out = os.path.join(work, f"cache_L{L}")
    os.makedirs(out, exist_ok=True)
    t0 = time.perf_counter()
    probe_dir = os.path.join(work, "probe", f"L{L}")
    names = ("domain", "general")
    sizes = {n: int(sets[n].offsets[-1]) for n in names}
    want = sets_signature({n: (sets[n], list(range(len(sets[n])))) for n in names})
    sig_path = os.path.join(work, "probe", "sets_sig.json")
    reused = (os.path.exists(sig_path) and json.load(open(sig_path)) == want
              and all(os.path.exists(os.path.join(probe_dir, f"h_{n}.npy")) for n in names))
    if reused:
        for n in names:
            dst = os.path.join(out, f"h_{n}.npy")
            os.link(os.path.join(probe_dir, f"h_{n}.npy"), dst + ".tmp")
            os.replace(dst + ".tmp", dst)
        how = f"hard-linked from {probe_dir}"
    else:
        tmp = os.path.join(work, f"cache_run_L{L}")
        run_bottom(cfg, {n: (sets[n], list(range(len(sets[n])))) for n in names}, L, tmp, save_at=[L],
                   mb_tokens=cfg["probe"]["mb_tokens"])
        for n in names:
            os.replace(os.path.join(tmp, f"L{L}", f"h_{n}.npy"), os.path.join(out, f"h_{n}.npy"))
        how = f"computed through layers 0..{L - 1}"
    dt = time.perf_counter() - t0
    gb = sum(os.path.getsize(os.path.join(out, f"h_{n}.npy")) for n in names) / 2**30
    meta = {"L": L, "tokens": sizes, "GB": gb, "seconds": dt, "how": how,
            "token_files": {n: os.path.join(work, f"tokens_{n}.npz") for n in names}}
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
    msg = (f"### cache L={L}\n- {how}; {sum(sizes.values())} tokens, {gb:.2f} GB, {dt:.0f}s "
           f"({sum(sizes.values()) / max(dt, 1e-9):.0f} tok/s)")
    print(msg)
    append_progress(cfg, msg)


if __name__ == "__main__":
    main()
