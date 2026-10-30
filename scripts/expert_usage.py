"""How concentrated the routing of a MoE model is on your workload: the share of expert selections that a fixed set of
the N most used experts per layer covers (= what the hot set of a mixed-precision GGUF serves at full precision).

  render: python scripts/expert_usage.py render --dialogs dialogs.jsonl --tokenizer MODEL_DIR --holdout 5 10 \
              [--general general.jsonl] --out-dir work/usage
          dialogs.jsonl: one dialog per line, {"n": id, "messages": [...]}; general.jsonl: {"text": ...} or
          {"messages": [...]}. Writes calib_train.txt, calib_holdout.txt, general.txt (chat template, special tokens as
          text) for llama-imatrix:
            llama-imatrix -m model.gguf -f work/usage/calib_train.txt -o work/usage/imatrix_train.gguf -c 512 --parse-special
  report: python scripts/expert_usage.py report --out-dir work/usage
          (needs imatrix_train.gguf, optional imatrix_holdout.gguf / imatrix_general.gguf) -> expert_usage.md + .json
The hot sets are ranked on train, so holdout shows how well they carry over to new topics.
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from smel.mixed_experts import coverage, read_counts  # noqa: E402

NS = [8, 16, 32, 48, 64, 96, 128, 192]


def render(args):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    os.makedirs(args.out_dir, exist_ok=True)
    chat = lambda m: tok.apply_chat_template(m, tokenize=False, enable_thinking=False)  # noqa: E731
    dialogs = [json.loads(l) for l in open(args.dialogs, encoding="utf-8") if l.strip()]
    parts = {"train": [], "holdout": []}
    for d in dialogs:
        parts["holdout" if d.get("n") in args.holdout else "train"].append(chat(d["messages"]))
    for k, v in parts.items():
        if v:
            open(os.path.join(args.out_dir, f"calib_{k}.txt"), "w", encoding="utf-8").write("\n".join(v))
            print(k, len(v), "dialogs,", sum(len(tok(x)["input_ids"]) for x in v), "tokens")
    if args.general:
        gen = [json.loads(l) for l in open(args.general, encoding="utf-8") if l.strip()]
        gen = [g["text"] if "text" in g else chat(g["messages"]) for g in gen]
        random.Random(0).shuffle(gen)
        out, n = [], 0
        for t in gen:
            out.append(t)
            n += len(tok(t)["input_ids"])
            if n >= args.general_tokens:
                break
        open(os.path.join(args.out_dir, "general.txt"), "w", encoding="utf-8").write("\n\n".join(out))
        print("general", len(out), "texts,", n, "tokens")


def report(args):
    src = {k: read_counts([os.path.join(args.out_dir, f"imatrix_{k}.gguf")]) for k in ("train", "holdout", "general")
           if os.path.exists(os.path.join(args.out_dir, f"imatrix_{k}.gguf"))}
    tr = src["train"]
    n_exp = len(next(iter(tr.values())))
    names = {"train": "train (ranked on itself)", "holdout": "held-out topics", "general": "general text"}
    lines = ["# Expert usage (llama-imatrix routing counts)\n",
             f"MoE layers: {len(tr)}, experts per layer: {n_exp}. Coverage = share of expert selections that go to the "
             "N most used experts of the layer, ranked on the train text.\n",
             "| N per layer | share of experts | " + " | ".join(names[k] for k in src) + " |",
             "|---|---|" + "---|" * len(src)]
    res = {"n_experts": n_exp, "layers": len(tr), "rows": []}
    for n in NS:
        row = {"N": n, **{k: coverage(tr, ev, n)[1] for k, ev in src.items()}}
        res["rows"].append(row)
        lines.append(f"| {n} | {n / n_exp:.0%} | " + " | ".join(f"{row[k]:.1%}" for k in src) + " |")
    ref = src.get("holdout", tr)
    per_layer, _ = coverage(tr, ref, 64)
    res["per_layer_N64"] = per_layer
    lines.append("\nCoverage per layer at N=64 (" + ("held-out" if "holdout" in src else "train") + "): "
                 + ", ".join(f"{l}: {v:.0%}" for l, v in per_layer.items()))
    open(os.path.join(args.out_dir, "expert_usage.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    json.dump(res, open(os.path.join(args.out_dir, "expert_usage.json"), "w"), indent=1)
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("render")
    p.add_argument("--dialogs", required=True)
    p.add_argument("--tokenizer", required=True, help="model dir with the tokenizer and chat template")
    p.add_argument("--holdout", type=int, nargs="*", default=[], help="dialog ids kept out of the ranking")
    p.add_argument("--general", default=None)
    p.add_argument("--general-tokens", type=int, default=20000)
    p.add_argument("--out-dir", required=True)
    p = sub.add_parser("report")
    p.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    {"render": render, "report": report}[args.cmd](args)


if __name__ == "__main__":
    main()
