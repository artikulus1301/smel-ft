"""Compare the quantization error (vs the original bf16 weights) of our re-quantized expert tensor and of the
tensor stored in an existing GGUF, per expert. Run without torch in the process (ggml/libomp)."""
import argparse
import json
import os
import sys

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from smel.gguf_patch import GGML, Imatrix  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--layer", type=int, default=2)
ap.add_argument("--tensor", default="ffn_down_exps")
ap.add_argument("--experts", type=int, default=16)
ap.add_argument("--model", default=None, help="bf16 safetensors dir (default: paths.model_dir)")
ap.add_argument("--gguf", required=True, help="GGUF of the base model")
ap.add_argument("--imatrix", required=True, help="importance matrix the GGUF was quantized with")
args = ap.parse_args()


def read_bf16(path, name):
    """Read a bf16 tensor from safetensors without torch -> float32 numpy."""
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        hdr = json.loads(f.read(n))
        meta = hdr[name]
        a, b = meta["data_offsets"]
        f.seek(8 + n + a)
        raw = np.frombuffer(f.read(b - a), dtype=np.uint16)
    return (raw.astype(np.uint32) << 16).view(np.float32).reshape(meta["shape"])


if args.model is None:  # no smel.pipeline import here: torch + ggml in one process abort (OpenMP clash)
    import yaml
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _md = yaml.safe_load(open(os.path.join(_root, "configs", "default.yaml")))["paths"]["model_dir"]
    args.model = _md if os.path.isabs(_md) else os.path.join(_root, _md)
wm = json.load(open(os.path.join(args.model, "model.safetensors.index.json")))
wm = wm["weight_map"]
p = f"model.language_model.layers.{args.layer}.mlp.experts."
if args.tensor == "ffn_down_exps":
    w = read_bf16(os.path.join(args.model, wm[p + "down_proj"]), p + "down_proj")
else:
    gu = read_bf16(os.path.join(args.model, wm[p + "gate_up_proj"]), p + "gate_up_proj")
    I = gu.shape[1] // 2
    w = gu[:, :I] if args.tensor == "ffn_gate_exps" else gu[:, I:]
name = f"blk.{args.layer}.{args.tensor}.weight"
t = next(t for t in GGUFReader(args.gguf).tensors if t.name == name)
qtype = int(t.tensor_type)
raw = np.asarray(t.data).view(np.uint8).reshape(-1)
per = raw.size // w.shape[0]
imx = Imatrix(args.imatrix).get(name)
g = GGML()
rows = []
for e in range(args.experts):
    ref = dequantize(raw[e * per:(e + 1) * per], t.tensor_type).reshape(w.shape[1:])
    ours = dequantize(np.frombuffer(g.quantize(w[e:e + 1], qtype, imx[e:e + 1]), np.uint8), t.tensor_type)
    ours = ours.reshape(w.shape[1:])
    # imatrix-weighted relative error (what the quantizer optimizes) and plain relative error
    wt = imx[e][None, :]
    werr = lambda q: float(np.sqrt(((q - w[e]) ** 2 * wt).sum() / ((w[e] ** 2) * wt).sum()))  # noqa: E731
    rel = lambda q: float(np.linalg.norm(q - w[e]) / np.linalg.norm(w[e]))  # noqa: E731
    rows.append((e, rel(ref), rel(ours), werr(ref), werr(ours), float(np.abs(ref - ours).max())))
print(f"{name} ({t.tensor_type.name}): expert | rel err unsloth / ours | imatrix-weighted unsloth / ours | max|diff|")
for r in rows:
    print(f"  {r[0]:3d} | {r[1]:.4f} / {r[2]:.4f} | {r[3]:.4f} / {r[4]:.4f} | {r[5]:.2e}")
a = np.array(rows)
print(f"mean rel err unsloth {a[:,1].mean():.5f} ours {a[:,2].mean():.5f}; weighted {a[:,3].mean():.5f} / {a[:,4].mean():.5f}")
