"""Blind pairwise judge on IDENTICAL contexts: a candidate model (GGUF in a local llama-server) answers every turn of
reference dialogs - same system prompt, same history - and an LLM judge (any OpenAI-compatible API) compares its answer
with the recorded reference answer, both orders; a verdict counts only if the two orders agree. Two-sided sign test.

Comparing answers from two separately generated dialogs is NOT the same: the dialogs diverge from the first turn, and a
judge shown one dialog's history prefers the answers that belong to it.

  SMEL_JUDGE_API_KEY=... python examples/persona_eval/same_context_judge.py dialogs.jsonl --gguf cand.gguf \
      --names reference candidate --out work/judge/cand.md
dialogs.jsonl: one dialog per line, {"n": id, "system": "...", "messages": [{"role": "user"|"assistant", ...}]}
(or the system prompt as the first message); an assistant message may carry its own "system" (the system prompt that
was used for that answer, if it changed during the dialog).
Judge: env SMEL_JUDGE_API_KEY, SMEL_JUDGE_BASE_URL (default https://api.openai.com/v1), SMEL_JUDGE_MODEL.
llama-server: see gen_compare.py (SMEL_LLAMA_SERVER, SMEL_LLAMA_SERVER_FLAGS / _ARGS).
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_compare import FLAGS, post, server_exe, wait_ready  # noqa: E402

CRITERIA = (
    "Which reply of the character is better at this point of the conversation? In order of importance: 1) continues "
    "the conversation naturally and answers the last message; 2) stays in character (persona of the system prompt); "
    "3) no breaking character, no stage directions; 4) correct, natural language; 5) does not invent facts the "
    "conversation does not contain. Short and lively beats long."
)
PROMPT = ("{criteria}\n\nSystem prompt of the character:\n{system}\n\nConversation so far:\n{history}\n\n"
          "Last message: {last}\n\nReply 1: {a}\n\nReply 2: {b}\n\nAnswer with one character: 1, 2, or 0 (tie).")
# long dialogs: without a cache limit llama-server keeps up to 8 GB of prompt cache in RAM
CACHE_FLAGS = ["--cache-ram", "1024", "--ctx-checkpoints", "8"]


def contexts(path):
    """[(dialog id, system, history up to and incl. the user message, reference answer)] for every assistant turn"""
    out = []
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        d = json.loads(line)
        system = d.get("system") or next((m["content"] for m in d["messages"] if m["role"] == "system"), "")
        msgs = [m for m in d["messages"] if m["role"] != "system"]
        for i, m in enumerate(msgs):
            if m["role"] == "assistant" and i and msgs[i - 1]["role"] == "user":
                hist = [{"role": x["role"], "content": x["content"]} for x in msgs[:i]]
                out.append((d.get("n", len(out)), m.get("system") or system, hist, m["content"]))
    return out


def ask_judge(client, model, criteria, system, hist, a, b, tries=10):
    text = "\n".join(f"{'User' if m['role'] == 'user' else 'Character'}: {m['content']}" for m in hist[-9:-1]) or "(start)"
    prompt = PROMPT.format(criteria=criteria, system=system, history=text, last=hist[-1]["content"], a=a, b=b)
    for attempt in range(tries):
        try:
            r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}],
                                               temperature=0, max_tokens=5)
            m = re.search(r"[012]", r.choices[0].message.content or "")
            return m.group(0) if m else "?"
        except Exception as e:  # rate limits, hung requests (timeout below)
            print(f"  judge error ({type(e).__name__}), retry {attempt + 1}", flush=True)
            time.sleep(min(60, 5 * (attempt + 1)))
    raise SystemExit("judge API keeps failing")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dialogs")
    ap.add_argument("--gguf", required=True, help="candidate model")
    ap.add_argument("--names", nargs=2, default=["reference", "candidate"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--criteria-file", help="judge instructions instead of the generic CRITERIA")
    ap.add_argument("--port", type=int, default=8092)
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--sampling", default='{"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5}',
                    help="JSON of sampling params for the candidate (use the ones the reference dialogs were made with)")
    args = ap.parse_args()
    from openai import OpenAI
    key = os.environ.get("SMEL_JUDGE_API_KEY")
    if not key:
        raise SystemExit("set SMEL_JUDGE_API_KEY (and SMEL_JUDGE_BASE_URL / SMEL_JUDGE_MODEL)")
    # some APIs hang for minutes without an error: give up after 40 s and retry
    client = OpenAI(api_key=key, base_url=os.environ.get("SMEL_JUDGE_BASE_URL", "https://api.openai.com/v1"),
                    timeout=40.0, max_retries=0)
    model = os.environ.get("SMEL_JUDGE_MODEL", "gpt-4o-mini")
    criteria = open(args.criteria_file, encoding="utf-8").read() if args.criteria_file else CRITERIA
    ctxs = contexts(args.dialogs)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    # 1) candidate answers (cached next to the report)
    cache = args.out + ".answers.json"
    if os.path.exists(cache):
        answers = json.load(open(cache, encoding="utf-8"))
    else:
        log = open(args.out + ".server.log", "w")
        proc = subprocess.Popen([server_exe(), "-m", args.gguf, "--port", str(args.port)] + FLAGS + CACHE_FLAGS,
                                stdout=log, stderr=subprocess.STDOUT)
        answers = []
        try:
            print(f"server ready in {wait_ready(args.port, proc):.0f}s", flush=True)
            for k, (_, system, hist, _) in enumerate(ctxs):
                msgs = ([{"role": "system", "content": system}] if system else []) + hist
                r = post(args.port, {"messages": msgs, "max_tokens": args.max_tokens, **json.loads(args.sampling),
                                     "chat_template_kwargs": {"enable_thinking": False}})
                answers.append(r["choices"][0]["message"]["content"].strip())
                if k % 10 == 9:
                    print(f"generated {k + 1}/{len(ctxs)}", flush=True)
        finally:
            proc.terminate()
            try:
                proc.wait(30)
            except subprocess.TimeoutExpired:
                proc.kill()
        json.dump(answers, open(cache, "w", encoding="utf-8"), ensure_ascii=False)

    # 2) blind judge, both orders
    ref_name, cand_name = args.names
    results, per_dialog = [], {}
    for (n, system, hist, ref), cand in zip(ctxs, answers):
        v1 = ask_judge(client, model, criteria, system, hist, ref, cand)   # reference shown first
        v2 = ask_judge(client, model, criteria, system, hist, cand, ref)   # candidate shown first
        s1 = {"1": ref_name, "2": cand_name, "0": "tie"}.get(v1)
        s2 = {"1": cand_name, "2": ref_name, "0": "tie"}.get(v2)
        res = s1 if (s1 is not None and s1 == s2) else "tie"
        results.append(res)
        per_dialog.setdefault(n, []).append(res)
        print(f"[{len(results):3d}] dialog {n}: {res}", flush=True)

    r_w, c_w = results.count(ref_name), results.count(cand_name)
    n_dec = r_w + c_w
    p = min(1.0, 2 * sum(math.comb(n_dec, i) for i in range(min(r_w, c_w) + 1)) / 2 ** n_dec) if n_dec else 1.0
    lines = [f"# Blind judge on identical contexts: {ref_name} vs {cand_name}\n",
             f"Contexts: {len(results)} from `{os.path.basename(args.dialogs)}`; candidate `{os.path.basename(args.gguf)}`; "
             f"judge `{model}`. Decided: {n_dec}; tie or the two orders disagree: {len(results) - n_dec}.\n",
             "| answers | wins | share of decided |", "|---|---|---|",
             f"| {ref_name} | {r_w} | {r_w / max(1, n_dec):.0%} |", f"| {cand_name} | {c_w} | {c_w / max(1, n_dec):.0%} |",
             f"\nTwo-sided sign test on decided pairs: p = {p:.3f}.\n",
             "| dialog | " + " | ".join(str(n) for n in per_dialog) + " |", "|---|" + "---|" * len(per_dialog),
             f"| {ref_name}:{cand_name} | " + " | ".join(
                 f"{v.count(ref_name)}:{v.count(cand_name)}" for v in per_dialog.values()) + " |"]
    open(args.out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
