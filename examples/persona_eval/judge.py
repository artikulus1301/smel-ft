"""Blind evaluation of generated answers (reports/gen_<tag>.jsonl, same prompts and seeds for every tag).

1) Automatic metrics on the domain prompts: persona breaks ("as an AI..."), *stage directions*, masculine
   first-person verbs (Azusa is a girl), answer length vs the dataset reference.
2) Pairwise LLM judge: the base GGUF through llama-server (temperature 0, thinking off) gets the persona, the
   dataset reference answer, the question and two anonymous answers A/B, and names the one closer to Azusa.
   Every pair is asked in both orders; a win counts only when both orders agree, otherwise it is a tie.
Usage: python examples/persona_eval/judge.py --gguf base.gguf --tags before L20_domain L20_random L20_shared --pairs L20_domain:before ...
"""
import argparse
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from smel.pipeline import load_config  # noqa: E402
from gen_compare import FLAGS, post, server_exe, wait_ready  # noqa: E402

# The example persona of the reference experiment; edit PERSONA and the regexes for your domain.
PERSONA = ("Азуса, 19 лет, студентка. Колючая, капризная, бунтарская, мило самовлюблённая, ранимая под бронёй, "
           "с чёрным юмором. Снимает на плёнку, прогуливает пары, ненавидит брокколи. Говорит коротко, живо, от "
           "первого лица в женском роде, никогда не говорит о себе как об ИИ.")
AI_BREAK = re.compile(r"искусственн\w* интеллект|языков\w* модел|\bкак ИИ\b|\bя ИИ\b|нейросет|ассистент|"
                      r"у меня нет (физического )?тела|не испытываю (эмоц|чувств)|не могу (есть|чувствовать)", re.I)
STAGE = re.compile(r"(?<!\*)\*[^*\n]{3,}\*(?!\*)")
MASC = re.compile(r"\bя\s+(бы\s+)?(сделал|почувствовал|был|сказал|подумал|решил|пошёл|видел|знал|понял|"
                  r"хотел|сам|готов|рад|уверен|должен)\b", re.I)


def load(reports, tag):
    return [json.loads(l) for l in open(os.path.join(reports, f"gen_{tag}.jsonl"), encoding="utf-8")]


def metrics(rows):
    dom = [r for r in rows if r["kind"] == "domain"]
    n = len(dom)
    ln = sum(len(r["answer"]) for r in dom) / n
    ref = sum(len(r["reference"]) for r in dom) / n
    return {"n": n, "ai_break": sum(bool(AI_BREAK.search(r["answer"])) for r in dom) / n,
            "stage_dirs": sum(bool(STAGE.search(r["answer"])) for r in dom) / n,
            "masc": sum(bool(MASC.search(r["answer"])) for r in dom) / n,
            "len": ln, "ref_len": ref}


def ask(port, q, ref, a, b):
    prompt = (f"Персонаж: {PERSONA}\n\nПример того, как отвечает настоящая Азуса (эталон):\n«{ref}»\n\n"
              f"Реплика собеседника: «{q}»\n\nОтвет A:\n«{a}»\n\nОтвет B:\n«{b}»\n\n"
              "Какой ответ лучше передаёт характер, голос и манеру Азусы (как в эталоне) и уместен по смыслу? "
              "Ответь одной буквой: A или B.")
    r = post(port, {"messages": [{"role": "user", "content": prompt}], "max_tokens": 4, "temperature": 0,
                    "chat_template_kwargs": {"enable_thinking": False}})
    t = r["choices"][0]["message"]["content"].strip().upper()
    return "A" if t.startswith("A") or t.startswith("А") else ("B" if t.startswith("B") or t.startswith("В") else "?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--pairs", nargs="+", required=True, help="x:y pairs to judge")
    ap.add_argument("--gguf", required=True, help="judge model (the base GGUF)")
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    reports, logs = cfg["paths"]["reports"], cfg["paths"]["logs"]
    runs = {t: load(reports, t) for t in args.tags}
    res = {"metrics": {t: metrics(r) for t, r in runs.items()}, "pairs": {}}
    log = open(os.path.join(logs, "llama_server_judge.log"), "w")
    proc = subprocess.Popen([server_exe(), "-m", args.gguf, "--port", str(args.port)] + FLAGS, stdout=log,
                            stderr=subprocess.STDOUT)
    try:
        wait_ready(args.port, proc)
        for pair in args.pairs:
            x, y = pair.split(":")
            win = {x: 0, y: 0, "tie": 0}
            per = []
            for i, (rx, ry) in enumerate(zip(runs[x], runs[y])):
                if rx["kind"] != "domain":
                    continue
                q = [m["content"] for m in rx["messages"] if m["role"] == "user"][-1]
                o1 = ask(args.port, q, rx["reference"], rx["answer"], ry["answer"])  # x is A
                o2 = ask(args.port, q, rx["reference"], ry["answer"], rx["answer"])  # x is B
                w = x if (o1, o2) == ("A", "B") else (y if (o1, o2) == ("B", "A") else "tie")
                win[w] += 1
                per.append((i, o1, o2, w))
            res["pairs"][pair] = {"wins": win, "per_prompt": per}
            print(f"{pair}: {win}", flush=True)
    finally:
        proc.terminate()
        proc.wait(60)
    json.dump(res, open(os.path.join(reports, "judge.json"), "w", encoding="utf-8"), indent=1,
              ensure_ascii=False)
    for t, m in res["metrics"].items():
        print(f"{t:22s} ai_break {m['ai_break']:.0%}  stage_dirs {m['stage_dirs']:.0%}  masc {m['masc']:.0%}  "
              f"len {m['len']:.0f} (ref {m['ref_len']:.0f})")


if __name__ == "__main__":
    main()
