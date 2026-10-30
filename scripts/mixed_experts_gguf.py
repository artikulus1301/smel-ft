"""Mixed-precision experts GGUF (see smel/mixed_experts.py and inference/README.md): the most used experts of every MoE
layer keep the source quantization, the others are re-quantized to a lower-bit type. Needs a llama.cpp build with
inference/llama_cpp_patches/mixed-experts-b11827.patch to run.

  python scripts/mixed_experts_gguf.py --src model.gguf --out mixed.gguf \
      --usage imatrix_dialogs.gguf --hot 128 --hot-layer 0:192 1:192 2:192 \
      --cold-type IQ2_XXS --cold-down IQ3_XXS --hf merged_model_dir --imatrix imatrix_of_the_source_quant.gguf

--usage: llama-imatrix outputs on text like the real workload (their routing counts rank the experts; layers not in
the counts, e.g. an MTP block, stay as they are). --cold-type copy keeps the cold bytes as well: the result must compute
exactly what the source computes (test of the patch). --random-hot SEED picks the hot experts at random.
llama.cpp (ggml-base for quantization): --llama-cpp, env SMEL_LLAMA_CPP_DIR or llama-server on PATH.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from smel.mixed_experts import build_mixed_gguf, choose_sets, parse_hot_layer, read_counts, EXPS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="source GGUF (the hot experts keep its bytes)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--usage", nargs="*", default=[], help="llama-imatrix GGUFs whose routing counts rank the experts")
    ap.add_argument("--random-hot", type=int, default=None)
    ap.add_argument("--hot", type=int, default=128, help="hot experts per layer")
    ap.add_argument("--hot-layer", nargs="*", default=[], help="per-layer overrides, L:N")
    ap.add_argument("--cold-type", default="IQ2_XXS", help="ggml type of the cold gate/up experts, or 'copy'")
    ap.add_argument("--cold-down", default=None, help="type of the cold ffn_down experts (default: --cold-type)")
    ap.add_argument("--hf", help="merged HF model the cold experts are re-quantized from")
    ap.add_argument("--imatrix", help="importance matrix for the cold quantization")
    ap.add_argument("--llama-cpp", default=None, help="llama.cpp build dir with ggml-base")
    ap.add_argument("--workers", type=int, default=12, help="quantization processes")
    ap.add_argument("--zero-experts", type=int, default=0, help="8 for builds of the first version of the patch")
    args = ap.parse_args()

    import gguf
    r = gguf.GGUFReader(args.src)
    arch = r.get_field("general.architecture").contents()
    n_expert = int(r.get_field(f"{arch}.expert_count").contents())
    layers = sorted({int(m.group(1)) for t in r.tensors if (m := EXPS.match(t.name))})
    del r
    counts = read_counts(args.usage) if args.usage else None
    if counts is not None:
        layers = [l for l in layers if l in counts]
    elif args.random_hot is None:
        raise SystemExit("--usage or --random-hot is needed to choose the hot experts")
    hot, cold = choose_sets(n_expert, layers, args.hot, parse_hot_layer(args.hot_layer), counts, args.random_hot)
    cold_types = {"gate": args.cold_type, "up": args.cold_type, "down": args.cold_down or args.cold_type}
    build_mixed_gguf(args.src, args.out, hot, cold, cold_types, hf=args.hf, imatrix=args.imatrix,
                     llama_cpp=args.llama_cpp, workers=args.workers, n_zero=args.zero_experts,
                     log=lambda s: print(s, flush=True))
    print(f"saved {args.out}: {os.path.getsize(args.out) / 2**30:.2f} GiB (source {os.path.getsize(args.src) / 2**30:.2f})")


if __name__ == "__main__":
    main()
