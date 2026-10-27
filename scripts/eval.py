"""Stage 8: perplexity before/after on the held-out domain set and on the general set (forgetting check).

Computed from the cache (layers < L are frozen). By default the block runs with bf16 experts, i.e. exactly the merged
model; --packed evaluates the int4 training-time block instead. Writes the numbers into reports/eval.md.
Usage: python scripts/eval.py --adapter artifacts/adapter_L20 [--packed]
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from smel.layer_runner import LayerMajorRunner, make_ctx  # noqa: E402
from smel.lora import BlockLoRA  # noqa: E402
from smel.loss import LMHead  # noqa: E402
from smel.model_io import Checkpoint  # noqa: E402
from smel.pipeline import (HiddenStore, TokenSet, append_progress, load_config, mb_rows, microbatches,  # noqa: E402
                           parse_overrides)
from smel.stream import CheckpointSource, PackedSource  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--packed", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config, parse_overrides(args.set))
    ck = Checkpoint(cfg["paths"]["model_dir"])
    d = ck.dims
    lora, meta = BlockLoRA.load(args.adapter, d)
    L = meta["L"]
    layers = list(range(L, d.n_layers))
    experts = {int(k): v for k, v in meta["experts"].items()}
    work = cfg["paths"]["work_dir"]
    sets = {}
    for name, split in (("domain", "eval"), ("general", "eval")):
        ts = TokenSet.load(os.path.join(work, f"tokens_{name}.npz"))
        store = HiddenStore(os.path.join(work, f"cache_L{L}", f"h_{name}.npy"))
        mbs = microbatches(ts, ts.indices(split), cfg["train"]["mb_tokens"])
        sets[name] = (ts, store, mbs)
    norm = ck.get(ck.norm_name()).cuda()
    head = LMHead.from_weight(ck.get(ck.lm_head_name()).cuda(), int8=args.packed and cfg["model"]["lm_head_int8"],
                              softcap=d.logit_softcap)
    res = {}
    for which, use_lora in (("base", False), ("adapter", True)):
        if args.packed:
            src = PackedSource(os.path.join(work, f"packed_L{L}"), d, layers, "cuda")
        else:
            src = CheckpointSource(ck, "cuda", selection=experts)
        run = LayerMajorRunner(d, src, layers, lora if use_lora else None, norm, head)
        hs, ctxs, tg, owner = [], [], [], []
        for name, (ts, store, mbs) in sets.items():
            for mb in mbs:
                rows = mb_rows(ts, mb)
                hs.append(store.read(rows).cuda())
                ctxs.append(make_ctx([ts.lengths[i] for i in mb], d, "cuda", torch.bfloat16))
                tg.append(torch.from_numpy(ts.tgt[rows].astype(np.int64)).cuda())
                owner.append(name)
        with torch.no_grad():
            out, _ = run.forward(hs, ctxs)
        for name in sets:
            idx = [i for i, o in enumerate(owner) if o == name]
            denom = sum(int((tg[i] != -100).sum()) for i in idx)
            loss, _ = run.head_loss([out[i] for i in idx], [tg[i] for i in idx], denom, grad=False)
            res[(which, name)] = (loss, denom)
        src.close()
    lines = [f"## Perplexity (block {'int4' if args.packed else 'bf16 = merged model'}, adapter {args.adapter})\n",
             "| set | tokens | before: loss / ppl | after: loss / ppl | ppl change |", "|---|---|---|---|---|"]
    summary = {}
    for name in sets:
        lb, n = res[("base", name)]
        la, _ = res[("adapter", name)]
        ch = math.exp(la) / math.exp(lb) - 1
        summary[name] = {"base_loss": lb, "adapter_loss": la, "ppl_change": ch, "tokens": n}
        lines.append(f"| {name} eval | {n} | {lb:.4f} / {math.exp(lb):.3f} | {la:.4f} / {math.exp(la):.3f} | "
                     f"{100 * ch:+.1f}% |")
    warn = ""
    if summary["general"]["ppl_change"] > 0.05:
        warn = (f"\n**Warning:** general-text perplexity grew by {100*summary['general']['ppl_change']:.1f}% "
                f"(> 5%): the model partly forgets general knowledge.\n")
    text = "\n".join(lines) + "\n" + warn
    print(text)
    p = os.path.join(cfg["paths"]["reports"], "eval.md")
    old = open(p, encoding="utf-8").read() if os.path.exists(p) else "# Eval\n"
    open(p, "w", encoding="utf-8").write(old.rstrip() + "\n\n" + text)
    json.dump(summary, open(os.path.join(args.adapter, "eval_ppl.json"), "w"), indent=1)
    append_progress(cfg, "### eval\n" + text)


if __name__ == "__main__":
    main()
