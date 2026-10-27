"""Generate answers to the held-out domain prompts (+ a few general ones) with a GGUF through llama-server, thinking
off, fixed seed per prompt. Run once for the base GGUF ("before") and once per fine-tuned GGUF; --report writes the
answers side by side into reports/eval.md.
llama-server: env SMEL_LLAMA_SERVER or llamacpp.dir in the config or PATH. FLAGS are tuned for an 8 GB GPU + 24 GB RAM
(MoE experts of 32 layers on the CPU); extra/override flags: env SMEL_LLAMA_SERVER_ARGS.
Usage: python examples/persona_eval/gen_compare.py --gguf base.gguf --tag before --n-domain 50
       python examples/persona_eval/gen_compare.py --report before after
"""
import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from smel.data import load_jsonl, split_rows  # noqa: E402
from smel.pipeline import load_config  # noqa: E402

def server_exe(cfg=None):
    exe = os.environ.get("SMEL_LLAMA_SERVER")
    d = (cfg or {}).get("llamacpp", {}).get("dir")
    if not exe and d:
        exe = next((os.path.join(d, n) for n in ("llama-server.exe", "llama-server") if os.path.exists(os.path.join(d, n))), None)
    exe = exe or shutil.which("llama-server")
    if not exe:
        raise SystemExit("llama-server not found: set SMEL_LLAMA_SERVER, llamacpp.dir or add it to PATH")
    return exe


_DEFAULT_FLAGS = ("-ngl 99 --n-cpu-moe 32 -t 8 -c 8192 -np 1 -fa on -ctk q8_0 -ctv q8_0 --load-mode mmap+mlock "
                  "--spec-type draft-mtp --spec-draft-n-max 2 --no-mmproj --jinja")
# SMEL_LLAMA_SERVER_FLAGS replaces the defaults (e.g. a model without an MTP head in its GGUF); _ARGS appends
FLAGS = (shlex.split(os.environ.get("SMEL_LLAMA_SERVER_FLAGS", _DEFAULT_FLAGS)) + ["--host", "127.0.0.1"]
         + shlex.split(os.environ.get("SMEL_LLAMA_SERVER_ARGS", "")))
# general questions (Russian, like the domain data): answers should not degrade after fine-tuning
GENERAL_PROMPTS = [
    "Почему небо голубое? Ответь коротко.",
    "Напиши функцию на Python, которая проверяет, является ли строка палиндромом.",
    "Кратко перескажи сюжет «Преступления и наказания».",
    "Сколько будет 17 × 23? Покажи вычисление.",
    "Дай три совета, как лучше высыпаться.",
]


def prompts(cfg, n_domain):
    rows = load_jsonl(cfg["paths"]["domain_data"])
    ev = split_rows(rows, cfg["data"]["n_val"], cfg["data"]["n_eval"], cfg["data"]["seed"])["eval"]
    out = []
    for r in ev[:n_domain]:
        msgs = r["messages"]
        last_user = max(i for i, m in enumerate(msgs) if m["role"] == "user")
        ref = next((m["content"] for m in msgs[last_user + 1:] if m["role"] == "assistant"), "")
        out.append({"kind": "domain", "messages": msgs[: last_user + 1], "reference": ref})
    out += [{"kind": "general", "messages": [{"role": "user", "content": p}], "reference": ""}
            for p in GENERAL_PROMPTS]
    return out


def post(port, body, timeout=600):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def wait_ready(port, proc, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise SystemExit(f"llama-server exited with code {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                if json.loads(r.read()).get("status") == "ok":
                    return time.time() - t0
        except Exception:
            pass
        time.sleep(2)
    raise SystemExit("llama-server did not become ready")


def generate(cfg, gguf, tag, port, n_domain, max_tokens):
    os.makedirs(cfg["paths"]["logs"], exist_ok=True)
    log = open(os.path.join(cfg["paths"]["logs"], f"llama_server_{tag}.log"), "w")
    proc = subprocess.Popen([server_exe(cfg), "-m", gguf, "--port", str(port)] + FLAGS, stdout=log, stderr=subprocess.STDOUT)
    try:
        ready = wait_ready(port, proc)
        print(f"server ready in {ready:.0f}s")
        res = []
        for i, p in enumerate(prompts(cfg, n_domain)):
            t0 = time.time()
            r = post(port, {"messages": p["messages"], "max_tokens": max_tokens, "temperature": 0.7, "top_p": 0.8,
                            "top_k": 20, "min_p": 0, "presence_penalty": 1.5, "seed": 1234 + i,
                            "chat_template_kwargs": {"enable_thinking": False}})
            ans = r["choices"][0]["message"]["content"]
            tps = r.get("timings", {}).get("predicted_per_second")
            res.append({**p, "answer": ans, "tok_s": tps, "seconds": time.time() - t0})
            print(f"[{i}] {p['kind']} {tps and round(tps, 1)} tok/s: {ans[:80]!r}", flush=True)
        out = os.path.join(cfg["paths"]["reports"], f"gen_{tag}.jsonl")
        with open(out, "w", encoding="utf-8") as f:
            for x in res:
                f.write(json.dumps({**x, "gguf": gguf}, ensure_ascii=False) + "\n")
        print("saved", out)
    finally:
        proc.terminate()
        try:
            proc.wait(30)
        except subprocess.TimeoutExpired:
            proc.kill()


def report(cfg, tags):
    runs = {t: load_jsonl(os.path.join(cfg["paths"]["reports"], f"gen_{t}.jsonl")) for t in tags}
    n = len(runs[tags[0]])
    md = [f"## Answers: {' vs '.join(tags)}\n",
          "llama-server, thinking off, temp 0.7, top_p 0.8, top_k 20, presence_penalty 1.5, same seed per prompt.\n"]
    for i in range(n):
        p = runs[tags[0]][i]
        sysm = next((m["content"] for m in p["messages"] if m["role"] == "system"), None)
        user = [m["content"] for m in p["messages"] if m["role"] == "user"][-1]
        md.append(f"### {i + 1}. [{p['kind']}] {user}")
        if sysm:
            md.append(f"*system:* {sysm}\n")
        if p["reference"]:
            md.append(f"**dataset reference:** {p['reference']}\n")
        for t in tags:
            md.append(f"**{t}:** {runs[t][i]['answer'].strip()}\n")
    p = os.path.join(cfg["paths"]["reports"], "eval.md")
    old = open(p, encoding="utf-8").read() if os.path.exists(p) else "# Eval\n"
    open(p, "w", encoding="utf-8").write(old.rstrip() + "\n\n" + "\n".join(md) + "\n")
    print("appended to", p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--gguf")
    ap.add_argument("--tag")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--n-domain", type=int, default=15)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--report", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.report:
        report(cfg, args.report)
    else:
        generate(cfg, args.gguf, args.tag, args.port, args.n_domain, args.max_tokens)


if __name__ == "__main__":
    main()
