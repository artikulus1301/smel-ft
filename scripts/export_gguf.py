"""Stage 9b: build the fine-tuned GGUF as a patched copy of an existing GGUF of the base model
(e.g. an unsloth UD quant with the MTP head).

Only the rows of the trained experts (ffn_{gate,up,down}_exps of the selected experts) and the shared-expert tensors of
the trained layers are re-quantized - same ggml type as in the source file, same importance matrix, merged
weights rounded to bf16 exactly as in the merged safetensors. All other bytes stay identical, so "before" vs "after"
differ only by the training.
Usage: python scripts/export_gguf.py --adapter artifacts/adapter_L20 --base-gguf base.gguf --imatrix imatrix.gguf
       [--out out.gguf] [--reuse]
"""
import argparse
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from smel.gguf_patch import GGMLProc, GGUFTarget, Imatrix, hf_to_gguf_tensors  # noqa: E402
from smel.lora import BlockLoRA  # noqa: E402
from smel.model_io import Checkpoint  # noqa: E402
from smel.pack import merge_experts, merge_mlp  # noqa: E402
from smel.pipeline import append_progress, load_config, parse_overrides  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--base-gguf", required=True, help="GGUF of the base model to patch (left untouched)")
    ap.add_argument("--imatrix", default=None, help="importance matrix the base GGUF was quantized with")
    ap.add_argument("--out", default=None)
    ap.add_argument("--reuse", action="store_true", help="re-patch an existing --out file instead of copying")
    args = ap.parse_args()
    cfg = load_config(args.config, parse_overrides(args.set))
    ck = Checkpoint(cfg["paths"]["model_dir"])
    d = ck.dims
    lora, meta = BlockLoRA.load(args.adapter, d)
    name = os.path.basename(os.path.normpath(args.adapter)).replace("adapter_", "")
    base_name = os.path.splitext(os.path.basename(args.base_gguf))[0]
    out = args.out or os.path.join(cfg["paths"]["work_dir"], "gguf", f"{base_name}-smel-{name}.gguf")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    t0 = time.perf_counter()
    tmp = out + ".tmp"
    if args.reuse and os.path.exists(out) and os.path.getsize(out) == os.path.getsize(args.base_gguf):
        # re-patch an earlier patched copy in place (no second 17 GB file): first restore every FFN tensor of the
        # layers that any adapter of this block may have touched from the base file, then patch as usual
        os.replace(out, tmp)
        src, dst = GGUFTarget(args.base_gguf), GGUFTarget(tmp)
        with open(args.base_gguf, "rb") as fb, open(tmp, "r+b") as fh:
            for tname, (_, off, nb, _) in src.info.items():
                parts = tname.split(".")
                if (len(parts) > 2 and parts[1].isdigit() and int(parts[1]) < d.n_layers
                        and parts[2].startswith("ffn_") and parts[2] != "ffn_gate_inp"):
                    assert dst.info[tname][1:3] == (off, nb)
                    fb.seek(off)
                    fh.seek(off)
                    fh.write(fb.read(nb))
        del src, dst
    else:
        shutil.copyfile(args.base_gguf, tmp)
    tgt, im, g = GGUFTarget(tmp), (Imatrix(args.imatrix) if args.imatrix else None), GGMLProc(cfg["llamacpp"]["dir"])
    report = []
    with open(tmp, "r+b") as fh:
        for key, LL in lora.layers.items():
            l = int(key)
            gu = dn = sh = None
            if LL.experts is not None:
                gu, dn = merge_experts(ck.get(ck.full_name(l, "mlp.experts.gate_up_proj")).cuda(),
                                       ck.get(ck.full_name(l, "mlp.experts.down_proj")).cuda(), LL.experts)
                gu, dn = gu.cpu(), dn.cpu()
            if LL.shared is not None:
                ws = [ck.get(n).cuda() for n in ck.shared_mlp_names(l)]
                sh = [t.cpu() for t in merge_mlp(*ws, LL.shared)]
            ids = LL.experts.expert_ids.tolist() if LL.experts is not None else []
            fused = f"blk.{l}.ffn_gate_up_exps.weight" in tgt.info
            for tname, w in hf_to_gguf_tensors(l, gu, dn, sh, ck.arch, fused).items():
                qtype, off, nb, tyname = tgt.info[tname]
                imx = im.get(tname) if im else None
                if "_exps" in tname:
                    per = nb // w.shape[0]  # bytes of one expert
                    for e in ids:
                        q = g.quantize(w[e:e + 1], qtype, None if imx is None else imx[e:e + 1])
                        assert len(q) == per
                        fh.seek(off + e * per)
                        fh.write(q)
                    report.append((tname, tyname, len(ids)))
                else:
                    tgt.write(tname, g.quantize(w, qtype, imx), fh)
                    report.append((tname, tyname, "all"))
            print(f"layer {l}: {len(ids)} experts + shared re-quantized", flush=True)
    g.close()
    del tgt  # GGUFReader keeps a memmap of the file open; Windows refuses to rename an open file
    import gc
    gc.collect()
    os.replace(tmp, out)
    json.dump({"base_gguf": args.base_gguf, "adapter": os.path.abspath(args.adapter), "imatrix": args.imatrix,
               "patched": report}, open(out + ".json", "w"), indent=1)
    msg = (f"### export_gguf\n- {out} ({os.path.getsize(out)/2**30:.2f} GB): patched copy of {args.base_gguf}; "
           f"{len(report)} tensors touched in {time.perf_counter()-t0:.0f}s")
    print(msg)
    append_progress(cfg, msg)


if __name__ == "__main__":
    main()
