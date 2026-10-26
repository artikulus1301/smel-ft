"""Token statistics of a chat jsonl with the model's tokenizer and chat template."""
import sys, json, statistics
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(sys.argv[2])
rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
lens, asst, turns, langs = [], [], [], {"cyr": 0, "lat": 0, "kk": 0}
for r in rows:
    msgs = r["messages"]
    ids = tok(tok.apply_chat_template(msgs, tokenize=False), add_special_tokens=False)["input_ids"]
    lens.append(len(ids))
    a = sum(len(tok(m["content"], add_special_tokens=False)["input_ids"]) for m in msgs if m["role"] == "assistant")
    asst.append(a)
    turns.append(sum(m["role"] == "assistant" for m in msgs))
    for m in msgs:
        for ch in m["content"]:
            if "Ѐ" <= ch <= "ӿ":
                langs["cyr"] += 1
                if ch in "әғқңөұүһіӘҒҚҢӨҰҮҺІ":
                    langs["kk"] += 1
            elif ch.isascii() and ch.isalpha():
                langs["lat"] += 1
print(f"rows={len(rows)} total_tokens={sum(lens)} assistant_tokens={sum(asst)}")
print(f"len mean={statistics.mean(lens):.0f} median={statistics.median(lens)} max={max(lens)} p95={sorted(lens)[int(.95*len(lens))]}")
print(f"assistant turns per row: {dict((t, turns.count(t)) for t in set(turns))}")
print("chars:", langs)
print("example template:\n" + tok.apply_chat_template(rows[0]["messages"], tokenize=False)[:700])
