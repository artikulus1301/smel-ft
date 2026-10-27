"""Check that our quantization path reproduces the bytes of an existing GGUF for an unchanged layer: original bf16
safetensors -> GGUF layout -> ggml_quantize_chunk(same type, same imatrix) == tensor bytes in the GGUF file.
Usage: python scripts/gguf_repro_check.py --gguf base.gguf --imatrix imatrix.gguf --layer 1
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from smel.gguf_patch import GGMLProc, GGUFTarget, Imatrix, hf_to_gguf_tensors  # noqa: E402
from smel.model_io import Checkpoint  # noqa: E402
from smel.pipeline import load_config  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--layer", type=int, default=1)
ap.add_argument("--model", default=None, help="bf16 safetensors dir (default: paths.model_dir)")
ap.add_argument("--gguf", required=True, help="GGUF of the base model")
ap.add_argument("--imatrix", required=True, help="importance matrix the GGUF was quantized with")
args = ap.parse_args()
cfg = load_config()
get = Checkpoint(args.model or cfg["paths"]["model_dir"]).get
ck = Checkpoint(args.model or cfg["paths"]["model_dir"])
l = args.layer
g, tgt, im = GGMLProc(cfg["llamacpp"]["dir"]), GGUFTarget(args.gguf), Imatrix(args.imatrix)
t = hf_to_gguf_tensors(l, get(ck.full_name(l, "mlp.experts.gate_up_proj")), get(ck.full_name(l, "mlp.experts.down_proj")),
                       [get(n) for n in ck.shared_mlp_names(l)], ck.arch, f"blk.{l}.ffn_gate_up_exps.weight" in tgt.info)
for name, w in t.items():
    qtype, _, nb, tname = tgt.info[name]
    t0 = time.perf_counter()
    q = g.quantize(w, qtype, im.get(name))
    ref = tgt.bytes_of(name)
    same = q == ref
    diff = 0 if same else int((np.frombuffer(q, np.uint8) != np.frombuffer(ref, np.uint8)).sum())
    print(f"{name:32s} {tname:8s} {nb/2**20:7.1f} MiB  {time.perf_counter()-t0:5.1f}s  "
          f"{'IDENTICAL' if same else f'differs in {diff} bytes ({100*diff/nb:.3f}%)'}")
