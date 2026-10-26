"""Build the 'general' Russian corpus for router comparison: Russian Wikipedia paragraphs (plain text) + Russian
assistant chats (GrandMaster-PRO-MAX test split). Output: chat/text jsonl in paths.general_data.
Usage: python scripts/make_general.py [--wiki-tokens 100000] [--chat-tokens 100000]
"""
import argparse
import json
import os
import random
import sys

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from smel.pipeline import load_config  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--wiki-tokens", type=int, default=100000)
ap.add_argument("--chat-tokens", type=int, default=100000)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
cfg = load_config()
out = cfg["paths"]["general_data"]
raw = os.path.join(os.path.dirname(out), "raw")
os.makedirs(raw, exist_ok=True)
from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained(cfg["paths"]["tokenizer"])
rng = random.Random(args.seed)
rows = []

wiki = hf_hub_download("wikimedia/wikipedia", "20231101.ru/train-00007-of-00021.parquet", repo_type="dataset",
                       local_dir=raw)
texts = pq.read_table(wiki, columns=["text"]).column("text").to_pylist()
rng.shuffle(texts)
n = 0
for t in texts:
    paras = [p.strip() for p in t.split("\n") if len(p.strip()) > 300]
    if not paras:
        continue
    chunk = "\n".join(paras[:3])[:3000]
    k = len(tok(chunk, add_special_tokens=False)["input_ids"])
    rows.append({"text": chunk, "source": "ruwiki"})
    n += k
    if n >= args.wiki_tokens:
        break
n_wiki = n

chat = hf_hub_download("Vikhrmodels/GrandMaster-PRO-MAX", "data/test-00000-of-00001.parquet", repo_type="dataset",
                       local_dir=raw)
tab = pq.read_table(chat)
convs = tab.column("conversation").to_pylist() if "conversation" in tab.column_names else \
    tab.column("messages").to_pylist()
rng.shuffle(convs)
n = 0
for c in convs:
    msgs = [{"role": m["role"], "content": m["content"]} for m in c if m.get("role") in ("system", "user", "assistant")]
    if not any(m["role"] == "assistant" for m in msgs) or not any(m["role"] == "user" for m in msgs):
        continue
    text = " ".join(m["content"] for m in msgs)
    cyr = sum("Ѐ" <= ch <= "ӿ" for ch in text)
    if cyr < 0.5 * max(1, sum(ch.isalpha() for ch in text)):
        continue  # keep Russian conversations only
    k = len(tok(tok.apply_chat_template(msgs, tokenize=False), add_special_tokens=False)["input_ids"])
    if k > 2048:
        continue
    rows.append({"messages": msgs, "source": "grandmaster"})
    n += k
    if n >= args.chat_tokens:
        break
rng.shuffle(rows)
with open(out, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"{out}: {len(rows)} rows, wiki ~{n_wiki} tok, chat ~{n} tok; columns of chat set: {tab.column_names}")
