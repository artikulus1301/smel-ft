"""Stage 9a: merge the LoRA adapter into the bf16 weights of the selected experts and shared experts.

The original folder is never modified. The output folder gets NTFS hard links for unchanged shards (no extra disk)
and rewritten copies of the shards that hold changed tensors, plus config/tokenizer files.
Usage: python scripts/merge.py --adapter artifacts/adapter_L20 --out work/merged_L20 [--verify]
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from smel.lora import BlockLoRA  # noqa: E402
from smel.model_io import Checkpoint  # noqa: E402
from smel.pack import merge_experts, merge_mlp  # noqa: E402
from smel.pipeline import append_progress, load_config, parse_overrides  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="only report which shards change and the disk needed")
    args = ap.parse_args()
    cfg = load_config(args.config, parse_overrides(args.set))
    src_dir = cfg["paths"]["model_dir"]
    ck = Checkpoint(src_dir)
    d = ck.dims
    lora, meta = BlockLoRA.load(args.adapter, d)
    changes = {}  # checkpoint tensor name -> (kind, layer)
    for key, LL in lora.layers.items():
        l = int(key)
        if LL.experts is not None:
            for short in ("mlp.experts.gate_up_proj", "mlp.experts.down_proj"):
                changes[ck.full_name(l, short)] = ("experts", l)
        if LL.shared is not None:   # Qwen: shared expert; Gemma 4: the dense MLP
            for n in ck.shared_mlp_names(l):
                changes[n] = ("shared", l)
    shards = sorted(set(ck.weight_map.values()))
    touched = sorted({ck.weight_map[n] for n in changes})
    need = sum(os.path.getsize(os.path.join(src_dir, s)) for s in touched)
    free = shutil.disk_usage(os.path.dirname(os.path.abspath(args.out)) or ".").free
    print(f"{len(changes)} tensors change in {len(touched)}/{len(shards)} shards; new data {need/2**30:.1f} GB, "
          f"free {free/2**30:.1f} GB")
    if args.dry_run:
        return
    if need * 1.05 > free:
        raise SystemExit("not enough free disk for the rewritten shards")
    os.makedirs(args.out, exist_ok=True)
    for f in os.listdir(src_dir):
        sp, dp = os.path.join(src_dir, f), os.path.join(args.out, f)
        if os.path.isdir(sp) or os.path.exists(dp):
            continue
        if f.endswith(".safetensors"):
            if f in touched:
                continue
            os.link(sp, dp)  # hard link: unchanged shard, no extra space
        else:
            shutil.copy2(sp, dp)
    cache = {}
    for s in touched:
        dp = os.path.join(args.out, s)
        if os.path.exists(dp):
            continue
        with safe_open(os.path.join(src_dir, s), "pt") as f:
            meta_st = f.metadata()
            tensors = {k: f.get_tensor(k) for k in f.keys()}
        for name in list(tensors):
            if name not in changes:
                continue
            kind, l = changes[name]
            LL = lora.get(l)
            if kind == "experts":
                if ("experts", l) not in cache:
                    gu_n, dn_n = ck.full_name(l, "mlp.experts.gate_up_proj"), ck.full_name(l, "mlp.experts.down_proj")
                    gu, dn = merge_experts(ck.get(gu_n).cuda(), ck.get(dn_n).cuda(), LL.experts)
                    cache[("experts", l)] = {gu_n: gu.cpu(), dn_n: dn.cpu()}
                tensors[name] = cache[("experts", l)].pop(name)
                if not cache[("experts", l)]:
                    del cache[("experts", l)]
            else:
                if ("shared", l) not in cache:
                    names = ck.shared_mlp_names(l)
                    m = merge_mlp(*[ck.get(n).cuda() for n in names], LL.shared)
                    cache[("shared", l)] = {n: t.cpu() for n, t in zip(names, m)}
                tensors[name] = cache[("shared", l)].pop(name)
                if not cache[("shared", l)]:
                    del cache[("shared", l)]
        save_file(tensors, dp + ".tmp", metadata=meta_st or {"format": "pt"})
        os.replace(dp + ".tmp", dp)
        print("rewrote", s, flush=True)
        del tensors
    assert not cache, f"tensors split across shards were not all written: {list(cache)}"
    json.dump({"adapter": os.path.abspath(args.adapter), "base": src_dir, "changed_tensors": sorted(changes),
               "rewritten_shards": touched}, open(os.path.join(args.out, "smel_merge.json"), "w"), indent=1)
    msg = f"### merge\n- {len(changes)} tensors merged, {len(touched)} shards rewritten -> {args.out}"
    if args.verify:
        ck2 = Checkpoint(args.out)
        worst = 0.0
        for name, (kind, l) in list(changes.items())[:6]:
            a, b = ck.get(name).float(), ck2.get(name).float()
            LL = lora.get(l)
            if kind == "experts":
                gu, dn = merge_experts(ck.get(ck.full_name(l, "mlp.experts.gate_up_proj")).cuda(),
                                       ck.get(ck.full_name(l, "mlp.experts.down_proj")).cuda(), LL.experts)
                want = (gu if "gate_up" in name else dn).float().cpu()
            else:
                names = ck.shared_mlp_names(l)
                m = dict(zip(names, merge_mlp(*[ck.get(n).cuda() for n in names], LL.shared)))
                want = m[name].float().cpu()
            worst = max(worst, float((b - want).abs().max()))
            assert not torch.equal(a, b) or float((want - a).abs().max()) == 0.0
        msg += f"\n- verify: merged tensors == base + delta (max abs diff {worst:.2e})"
    print(msg)
    append_progress(cfg, msg)


if __name__ == "__main__":
    main()
