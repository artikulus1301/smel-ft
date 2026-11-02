"""SMEL FT web UI: a local page that runs the pipeline stages (scripts/*.py) one at a time and shows their output.

Standard library only (+ pyyaml and psutil, already dependencies). A project is a config file in configs/projects/
made from configs/default.yaml (Qwen) or configs/gemma4.yaml (Gemma 4); its outputs go to work/<name>,
artifacts/<name>, reports/<name> and logs/<name>, and every stage is the same command as in the README.
Usage: python ui/server.py [--port 7860] [--host 127.0.0.1]
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

import yaml

import chat_backend as cb

ROOT =os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECTS = os.path.join(ROOT, "configs", "projects")
PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
CHAT_PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat.html")

MODELS = {
    "qwen": {"label": "Qwen3.6-35B-A3B", "config": "configs/default.yaml", "repo": "Qwen/Qwen3.6-35B-A3B",
             "size_gb": 72},
    "gemma4": {"label": "Gemma 4 26B-A4B", "config": "configs/gemma4.yaml", "repo": "google/gemma-4-26B-A4B-it",
               "size_gb": 52},
}
# settings the page can change: key -> type
SETTINGS = {
    "paths.model_dir": str, "paths.domain_data": str, "paths.general_data": str, "paths.work_dir": str,
    "data.max_len": int,
    "select.coverage": float, "select.L_override": int, "select.vram_budget_gb": float, "select.ram_budget_gb": float,
    "train.lr": float, "train.epochs": int, "train.r": int, "train.alpha": int, "train.tokens_per_step": int,
}
NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
PIPELINE = ["probe", "select", "cache", "pack", "train", "eval"]


# ----------------------------------------------------------------------------------------------- projects
def project_path(name):
    if not NAME_RE.match(name or ""):
        raise ValueError("name: letters, digits, _ and -, up to 40")
    return os.path.join(PROJECTS, f"{name}.yaml")


def abspath(p):
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(ROOT, p))


def read_project(name):
    with open(project_path(name), encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_project(name, cfg):
    os.makedirs(PROJECTS, exist_ok=True)
    with open(project_path(name), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)


def create_project(name, model, model_dir=None):
    if model not in MODELS:
        raise ValueError(f"model: one of {list(MODELS)}")
    if os.path.exists(project_path(name)):
        raise ValueError(f"project {name} exists")
    with open(os.path.join(ROOT, MODELS[model]["config"]), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    p = cfg["paths"]
    if model_dir:
        p["model_dir"] = model_dir
    p.update({"work_dir": f"work/{name}", "domain_data": f"data/{name}/domain.jsonl",
              "artifacts": f"artifacts/{name}", "reports": f"reports/{name}", "logs": f"logs/{name}",
              "progress": f"logs/{name}/progress.md", "selection": f"configs/projects/{name}.selection.yaml"})
    cfg["ui"] = {"model": model, "created": time.strftime("%Y-%m-%d %H:%M")}
    write_project(name, cfg)
    return cfg


def list_projects():
    if not os.path.isdir(PROJECTS):
        return []
    out = []
    for f in sorted(os.listdir(PROJECTS)):
        if f.endswith(".yaml") and not f.endswith(".selection.yaml"):
            name = f[:-5]
            try:
                out.append({"name": name, "model": read_project(name).get("ui", {}).get("model")})
            except Exception:
                pass
    return out


def get_key(cfg, key):
    d = cfg
    for k in key.split("."):
        d = (d or {}).get(k)
    return d


def update_settings(name, values):
    cfg = read_project(name)
    for key, v in values.items():
        if key not in SETTINGS:
            raise ValueError(f"unknown setting {key}")
        if v in ("", None):
            v = None
            if key.startswith("paths."):
                raise ValueError(f"{key} is required")
        else:
            v = SETTINGS[key](v)
        sect, k = key.split(".")
        cfg.setdefault(sect, {})[k] = v
    write_project(name, cfg)
    return cfg


# ----------------------------------------------------------------------------------------------- data check
def check_dataset(path, max_rows=200000):
    """Format and quality check of a chat jsonl ({"messages": [...]}) or text jsonl ({"text": ...})."""
    res = {"path": path, "exists": os.path.exists(path)}
    if not res["exists"]:
        return res
    res["size_mb"] = round(os.path.getsize(path) / 2**20, 2)
    n = chats = texts = 0
    errors, seen = [], set()
    dups = short = asst_n = nospace = 0
    asst_chars, turns = [], []
    roles = {}
    sample = None
    with open(path, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            if n >= max_rows:
                res["truncated"] = True
                break
            n += 1
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"line {i}: not JSON ({e.msg})")
                continue
            key = line.strip()
            if key in seen:
                dups += 1
            seen.add(key)
            if isinstance(r, dict) and isinstance(r.get("text"), str):
                texts += 1
                continue
            msgs = r.get("messages") if isinstance(r, dict) else None
            if not isinstance(msgs, list) or not msgs:
                errors.append(f"line {i}: no \"messages\" list or \"text\"")
                continue
            bad = [m for m in msgs if not isinstance(m, dict) or not isinstance(m.get("content"), str)
                   or m.get("role") not in ("system", "user", "assistant")]
            if bad:
                errors.append(f"line {i}: message without role system/user/assistant or text content")
                continue
            a = [m["content"] for m in msgs if m["role"] == "assistant"]
            if not a:
                errors.append(f"line {i}: no assistant message (nothing to learn)")
                continue
            chats += 1
            sample = sample or msgs
            turns.append(len(a))
            for m in msgs:
                roles[m["role"]] = roles.get(m["role"], 0) + 1
            for c in a:
                asst_n += 1
                asst_chars.append(len(c))
                short += len(c.strip()) < 20
                nospace += bool(re.search(r"[.,!?;:][^\s\d.,!?;:)\]»\"'…-]", c))
    asst_chars.sort()
    res.update({"rows": n, "chats": chats, "texts": texts, "errors": errors[:10], "n_errors": len(errors),
                "duplicates": dups, "roles": roles,
                "assistant_turns": asst_n, "turns_per_chat": round(sum(turns) / max(1, len(turns)), 2),
                "assistant_chars_median": asst_chars[len(asst_chars) // 2] if asst_chars else 0,
                "assistant_chars_p95": asst_chars[int(0.95 * (len(asst_chars) - 1))] if asst_chars else 0,
                "short_answers": round(short / max(1, asst_n), 3),
                "no_space_after_punct": round(nospace / max(1, asst_n), 3),
                "sample": [{"role": m["role"], "content": m["content"][:600]} for m in (sample or [])[:4]]})
    warn = []
    if res["n_errors"]:
        warn.append("errors")
    if chats and chats < 200:
        warn.append("few")
    if res["short_answers"] > 0.2:
        warn.append("short")
    if res["no_space_after_punct"] > 0.1:
        warn.append("nospace")
    if dups > 0.05 * max(1, n):
        warn.append("dups")
    res["warnings"] = warn
    return res


# ----------------------------------------------------------------------------------------------- status
def project_status(name):
    cfg = read_project(name)
    p = {k: abspath(v) if isinstance(v, str) else v for k, v in cfg["paths"].items()}
    md = p["model_dir"]
    shards = [f for f in os.listdir(md) if f.endswith(".safetensors")] if os.path.isdir(md) else []
    st = {"model": {"ok": os.path.exists(os.path.join(md, "config.json")) and bool(shards), "shards": len(shards)},
          "domain": {"ok": os.path.exists(p["domain_data"])},
          "general": {"ok": os.path.exists(p["general_data"])},
          "probe": {"ok": os.path.exists(os.path.join(p["artifacts"], "routing_stats.pt"))}}
    sel = None
    if os.path.exists(p["selection"]):
        with open(p["selection"], encoding="utf-8") as f:
            sel = yaml.safe_load(f)
    st["select"] = {"ok": sel is not None}
    if sel:
        L = sel["L"]
        st["select"].update({"L": L, "L_auto": sel.get("L_auto"), "mode": sel.get("mode"),
                             "experts": sum(len(v) for v in sel["experts"].values()),
                             "estimate": sel.get("estimate", {})})
        st["cache"] = {"ok": os.path.exists(os.path.join(p["work_dir"], f"cache_L{L}", "meta.json"))}
        st["pack"] = {"ok": os.path.exists(os.path.join(p["work_dir"], f"packed_L{L}", "head.safetensors"))}
    else:
        st["cache"] = st["pack"] = {"ok": False}
    adapters = []
    if os.path.isdir(p["artifacts"]):
        for d in sorted(os.listdir(p["artifacts"])):
            ad = os.path.join(p["artifacts"], d)
            if d.startswith("adapter_") and os.path.isdir(ad):
                ev = os.path.join(ad, "eval_ppl.json")
                adapters.append({"tag": d[len("adapter_"):],
                                 "eval": json.load(open(ev, encoding="utf-8")) if os.path.exists(ev) else None})
    st["train"] = {"ok": bool(adapters)}
    st["eval"] = {"ok": any(a["eval"] for a in adapters)}
    runs = []
    if os.path.isdir(p["logs"]):
        runs = sorted(f[len("train_"):-len(".jsonl")] for f in os.listdir(p["logs"])
                      if f.startswith("train_") and f.endswith(".jsonl"))
    reports = []
    if os.path.isdir(p["reports"]):
        reports = sorted(f for f in os.listdir(p["reports"]) if f.endswith(".md"))
    try:
        free = shutil.disk_usage(p["work_dir"] if os.path.exists(p["work_dir"]) else ROOT).free / 1e9
    except OSError:
        free = None
    settings = {k: get_key(cfg, k) for k in SETTINGS}
    model = cfg.get("ui", {}).get("model")
    return {"name": name, "model": model, "model_info": MODELS.get(model), "settings": settings, "status": st,
            "adapters": adapters, "train_runs": runs, "reports": reports, "disk_free_gb": free,
            "progress": os.path.exists(p["progress"])}


def train_log(name, tag):
    cfg = read_project(name)
    path = os.path.join(abspath(cfg["paths"]["logs"]), f"train_{tag}.jsonl")
    out, head = [], None
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("step") == 0 and "val" in r:
                    head = r["val"]
                if r.get("loss") is not None:
                    out.append({k: r.get(k) for k in ("step", "epoch", "loss", "val", "lr", "tok_s", "step_s",
                                                       "vram_peak_gb")})
    return {"tag": tag, "val0": head, "steps": out}


# ----------------------------------------------------------------------------------------------- jobs
def stage_commands(name, stage, params):
    """Commands (argv lists) of one stage; python = this interpreter, cwd = repository root."""
    cfg = read_project(name)
    conf = project_path(name)
    py = sys.executable
    s = lambda script: [py, os.path.join(ROOT, "scripts", script), "--config", conf]  # noqa: E731
    if stage == "download":
        m = MODELS[cfg["ui"]["model"]]
        return [[py, os.path.join(ROOT, "scripts", "download.py"), "--repo", m["repo"],
                 "--dir", abspath(cfg["paths"]["model_dir"])]]
    if stage == "general":
        return [s("make_general.py")]
    if stage == "stats":
        return [[py, os.path.join(ROOT, "scripts", "dataset_stats.py"), abspath(cfg["paths"]["domain_data"]),
                 abspath(cfg["paths"].get("tokenizer") or cfg["paths"]["model_dir"])]]
    if stage == "probe":
        return [s("probe.py") + (["--fresh"] if params.get("fresh") else [])]
    if stage == "select":
        return [s("select_experts.py")]
    if stage == "cache":
        return [s("cache.py")]
    if stage == "pack":
        return [s("pack.py")]
    tag = params.get("tag") or ""
    if tag and not NAME_RE.match(tag):
        raise ValueError("tag: letters, digits, _ and -")
    if stage == "train":
        c = s("train.py")
        if tag:
            c += ["--tag", tag]
        if params.get("max_steps"):
            c += ["--max-steps", str(int(params["max_steps"]))]
        if params.get("resume"):
            c += ["--resume"]
        return [c]
    if stage == "eval":
        if not tag:
            raise ValueError("eval: choose an adapter")
        return [s("eval.py") + ["--adapter", os.path.join(abspath(cfg["paths"]["artifacts"]), f"adapter_{tag}")]
                + (["--packed"] if params.get("packed") else [])]
    if stage == "selftest":
        return [[py, "-m", "pytest", "-q", "-p", "no:cacheprovider"]]
    if stage == "pipeline":
        start = params.get("from") or PIPELINE[0]
        if start not in PIPELINE:
            raise ValueError(f"from: one of {PIPELINE}")
        out = []
        for st in PIPELINE[PIPELINE.index(start):]:
            if st == "eval":   # the default tag of train.py is L{L}, known only after select has run
                out.append(("eval_last", tag))
            else:
                out += stage_commands(name, st, {**params, "max_steps": None, "resume": False})
        return out
    raise ValueError(f"unknown stage {stage}")


class Job:
    def __init__(self, project, stage, cmds, log_path):
        self.project, self.stage, self.cmds, self.log_path = project, stage, cmds, log_path
        self.state, self.rc, self.started, self.ended = "running", None, time.time(), None
        self.step = 0
        self.proc = None
        self.stop_requested = False
        self.buf, self.base = bytearray(), 0      # base: bytes dropped from the front of buf
        self.lock = threading.Lock()

    def write(self, data):
        with self.lock:
            self.buf += data
            if len(self.buf) > 4 << 20:   # keep the last 2 MB in memory; the log file has everything
                drop = len(self.buf) - (2 << 20)
                del self.buf[:drop]
                self.base += drop
        self.log.write(data)
        self.log.flush()

    def run(self):
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self.log = open(self.log_path, "ab")
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        try:
            for i, cmd in enumerate(self.cmds):
                self.step = i
                if self.stop_requested:
                    break
                if isinstance(cmd, tuple):            # ("eval_last", tag): the adapter of the train step before
                    tag = cmd[1] or f"L{read_selection_L(self.project)}"
                    cmd = stage_commands(self.project, "eval", {"tag": tag})[0]
                self.write(f"\n$ {' '.join(_short(c) for c in cmd)}\n".encode())
                kw = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else \
                    {"start_new_session": True}
                self.proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                             stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, **kw)
                while True:
                    chunk = self.proc.stdout.read1(65536)
                    if not chunk:
                        break
                    self.write(chunk)
                self.rc = self.proc.wait()
                if self.rc != 0:
                    break
            self.state = "stopped" if self.stop_requested else ("done" if self.rc == 0 else "failed")
        except Exception as e:
            self.write(f"\n[ui] {type(e).__name__}: {e}\n".encode())
            self.state = "failed"
        finally:
            self.ended = time.time()
            self.write(f"\n[ui] {self.stage}: {self.state} (exit {self.rc}) after "
                       f"{(self.ended - self.started) / 60:.1f} min\n".encode())
            self.log.close()

    def stop(self):
        self.stop_requested = True
        p = self.proc
        if p and p.poll() is None:
            if os.name == "nt":   # the whole tree: scripts may start their own children
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(p.pid)], capture_output=True)
            else:
                import signal
                os.killpg(p.pid, signal.SIGTERM)

    def info(self, since=0):
        with self.lock:
            data, base = bytes(self.buf), self.base
        start = min(max(0, since - base), len(data))
        return {"project": self.project, "stage": self.stage, "state": self.state, "rc": self.rc,
                "started": self.started, "ended": self.ended, "step": self.step, "steps": len(self.cmds),
                "log": data[start:].decode("utf-8", errors="replace"), "offset": base + len(data),
                "log_path": self.log_path}


def read_selection_L(name):
    cfg = read_project(name)
    with open(abspath(cfg["paths"]["selection"]), encoding="utf-8") as f:
        return yaml.safe_load(f)["L"]


def _short(a):
    a = a.replace(ROOT + os.sep, "")
    return f'"{a}"' if " " in a else a


JOB = {"cur": None}
JOB_LOCK = threading.Lock()


def start_job(name, stage, params):
    with JOB_LOCK:
        cur = JOB["cur"]
        if cur and cur.state == "running":
            raise RuntimeError(f"busy: {cur.project}/{cur.stage} is running")
        if cb.MODEL.running():      # the chat model and a stage would not fit in VRAM together
            raise RuntimeError("busy: the chat model is loaded; stop it in the Chat window first")
        cmds = stage_commands(name, stage, params or {})
        cfg = read_project(name)
        log_path = os.path.join(abspath(cfg["paths"]["logs"]), f"ui_{stage}_{time.strftime('%Y%m%d_%H%M%S')}.log")
        job = Job(name, stage, cmds, log_path)
        JOB["cur"] = job
        threading.Thread(target=job.run, daemon=True).start()
        return job


# ----------------------------------------------------------------------------------------------- system
_SYS = {"t": 0, "v": None}


def system_info():
    if time.time() - _SYS["t"] < 3 and _SYS["v"]:
        return _SYS["v"]
    v = {}
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,temperature.gpu,"
                              "utilization.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout.strip().splitlines()[0]
        n, mu, mt, t, u = [x.strip() for x in out.split(",")]
        v["gpu"] = {"name": n, "mem_used_gb": int(mu) / 1024, "mem_total_gb": int(mt) / 1024, "temp": int(t),
                    "util": int(u)}
    except Exception:
        v["gpu"] = None
    try:
        import psutil
        m = psutil.virtual_memory()
        v["ram"] = {"used_gb": (m.total - m.available) / 2**30, "total_gb": m.total / 2**30}
    except Exception:
        v["ram"] = None
    _SYS.update(t=time.time(), v=v)
    return v


# ----------------------------------------------------------------------------------------------- chat
def chat_state():
    s = cb.load_settings()
    dirs = [s["gguf_dir"]]
    for p in list_projects():       # where scripts/export_gguf.py writes by default
        try:
            dirs.append(os.path.join(abspath(read_project(p["name"])["paths"]["work_dir"]), "gguf"))
        except Exception:
            pass
    exe = cb.find_server(s)
    try:
        command = " ".join(cb.build_command(s, exe)) if exe and s["gguf"] else ""
    except ValueError:
        command = ""
    cur = JOB["cur"]
    return {"settings": s, "defaults": cb.DEFAULTS, "status": cb.MODEL.status(), "ggufs": cb.list_ggufs(dirs),
            "server_exe": exe, "command": command, "system": system_info(),
            "busy_job": bool(cur and cur.state == "running")}


class ClientGone(Exception):
    pass


# ----------------------------------------------------------------------------------------------- http
class Handler(BaseHTTPRequestHandler):
    server_version = "smel-ui"

    def log_message(self, fmt, *args):
        pass

    def send(self, code, body, ctype="application/json; charset=utf-8"):
        if not isinstance(body, (bytes, bytearray)):
            body = json.dumps(body, ensure_ascii=False, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}") if n else {}

    def route(self, method):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        parts = [unquote(x) for x in u.path.strip("/").split("/") if x]
        try:
            if method == "GET" and not parts:
                with open(PAGE, "rb") as f:
                    return self.send(200, f.read(), "text/html; charset=utf-8")
            if method == "GET" and parts == ["chat"]:
                with open(CHAT_PAGE, "rb") as f:
                    return self.send(200, f.read(), "text/html; charset=utf-8")
            if parts[:1] != ["api"]:
                return self.send(404, {"error": "not found"})
            parts = parts[1:]
            if parts[:1] == ["chat"]:
                return self.chat_route(method, parts[1:], q)
            if method == "GET" and parts == ["state"]:
                j = JOB["cur"]
                return self.send(200, {"models": MODELS, "projects": list_projects(), "system": system_info(),
                                       "job": j.info(1 << 62) if j else None, "python": sys.executable})
            if method == "GET" and parts == ["job"]:
                j = JOB["cur"]
                return self.send(200, j.info(int(q.get("since", 0))) if j else None)
            if method == "POST" and parts == ["job", "stop"]:
                if JOB["cur"]:
                    JOB["cur"].stop()
                return self.send(200, {"ok": True})
            if method == "POST" and parts == ["projects"]:
                b = self.body()
                create_project(b.get("name"), b.get("model"), b.get("model_dir") or None)
                return self.send(200, project_status(b["name"]))
            if len(parts) >= 2 and parts[0] == "projects":
                name, rest = parts[1], parts[2:]
                project_path(name)
                if not os.path.exists(project_path(name)):
                    return self.send(404, {"error": f"no project {name}"})
                return self.project_route(method, name, rest, q)
            return self.send(404, {"error": "not found"})
        except (ValueError, KeyError, FileNotFoundError) as e:
            return self.send(400, {"error": str(e)})
        except RuntimeError as e:
            return self.send(409, {"error": str(e)})

    def chat_route(self, method, rest, q):
        M = cb.MODEL
        if method == "GET" and rest == ["state"]:
            return self.send(200, chat_state())
        if method == "PUT" and rest == ["settings"]:
            cb.save_settings(self.body())
            return self.send(200, chat_state())
        if method == "POST" and rest == ["start"]:
            cur = JOB["cur"]
            if cur and cur.state == "running":
                raise RuntimeError(f"busy: {cur.project}/{cur.stage} is running; the GPU is in use")
            M.start()
            return self.send(200, chat_state())
        if method == "POST" and rest == ["stop"]:
            M.stop()
            return self.send(200, chat_state())
        if method == "POST" and rest == ["warmup"]:
            M.warm_async(str(self.body().get("system") or ""))
            return self.send(200, {"status": "scheduled"})
        if method == "GET" and rest == ["log"]:
            return self.send(200, cb.tail(cb.LOG, int(q.get("n", 60))).encode(), "text/plain; charset=utf-8")
        if method == "POST" and rest == ["completions"]:
            body = M.prepare_chat(self.body())
            state = M.status()["state"]
            if state not in ("ready", "sleeping"):
                raise RuntimeError(f"the model is {state}: start it first")
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()

            def write(b):
                try:
                    self.wfile.write(b)
                    self.wfile.flush()
                except OSError:
                    raise ClientGone()
            try:
                M.stream_chat(body, write)
            except ClientGone:
                pass
            except Exception as e:
                try:
                    write(("data: " + json.dumps({"error": f"{type(e).__name__}: {e}"}) + "\n\n").encode())
                except ClientGone:
                    pass
            return
        return self.send(404, {"error": "not found"})

    def project_route(self, method, name, rest, q):
        cfg = read_project(name)
        if method == "GET" and not rest:
            return self.send(200, project_status(name))
        if method == "PUT" and rest == ["settings"]:
            update_settings(name, self.body())
            return self.send(200, project_status(name))
        if method == "GET" and rest == ["dataset"]:
            kind = q.get("kind", "domain")
            return self.send(200, check_dataset(abspath(cfg["paths"][f"{kind}_data"])))
        if method == "PUT" and rest == ["upload"]:
            kind = q.get("kind", "domain")
            if kind not in ("domain", "general"):
                raise ValueError("kind: domain | general")
            dest = os.path.join(ROOT, "data", name, f"{kind}.jsonl")
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            n = int(self.headers.get("Content-Length") or 0)
            with open(dest + ".part", "wb") as f:
                while n > 0:
                    chunk = self.rfile.read(min(n, 1 << 20))
                    if not chunk:
                        break
                    f.write(chunk)
                    n -= len(chunk)
            os.replace(dest + ".part", dest)
            update_settings(name, {f"paths.{kind}_data": os.path.relpath(dest, ROOT).replace(os.sep, "/")})
            return self.send(200, check_dataset(dest))
        if method == "POST" and rest == ["run"]:
            b = self.body()
            job = start_job(name, b.get("stage"), b.get("params") or {})
            return self.send(200, job.info())
        if method == "GET" and len(rest) == 2 and rest[0] == "train":
            return self.send(200, train_log(name, rest[1]))
        if method == "GET" and rest == ["report"]:
            f = q.get("file", "")
            base = abspath(cfg["paths"]["progress"]) if f == "progress" else \
                os.path.join(abspath(cfg["paths"]["reports"]), os.path.basename(f))
            if not os.path.exists(base):
                return self.send(404, {"error": "no report"})
            ctype = {"png": "image/png", "md": "text/markdown; charset=utf-8"}.get(base.rsplit(".", 1)[-1],
                                                                                 "text/plain; charset=utf-8")
            with open(base, "rb") as fh:
                return self.send(200, fh.read(), ctype)
        return self.send(404, {"error": "not found"})

    def do_GET(self):
        self.route("GET")

    def do_POST(self):
        self.route("POST")

    def do_PUT(self):
        self.route("PUT")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 opens the page (and the stages) to the network")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"SMEL FT UI: http://{a.host}:{a.port}  (python {sys.executable})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        if JOB["cur"]:
            JOB["cur"].stop()
    finally:
        cb.MODEL.stop()


if __name__ == "__main__":
    main()
