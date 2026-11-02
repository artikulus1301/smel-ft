"""Chat backend of the SMEL FT UI: starts llama-server for a GGUF with this project's inference settings, wakes it before
the first message, and proxies streamed chat requests. Standard library only (+ pyyaml, psutil).

The settings come from measurements on the reference app (Qwen3.6-35B-A3B, RTX 4070 Laptop 8 GB, 24 GB RAM):
  - experts in RAM (--n-cpu-moe 99): 4.4 GB VRAM; experts on the GPU did not generate faster (22.6 vs 22.3 tok/s);
  - MTP draft of 1 token: 22.8 tok/s against 21.1 without it and 20.9 with 2 (needs a GGUF with the MTP head);
  - weights are mmapped; generation speed follows how much of them stays in RAM (29 tok/s with all 12 GB resident,
    24 with 9 GB, 15 with 5 GB), so free RAM matters more than any flag. mlock pins them but needs the free RAM;
  - micro-batch 1024: long prompts ~1.7x faster than 512; q8_0 KV cache; one slot with the prompt cache, so each turn
    computes only the new tokens;
  - sleep after idle unloads the model (RAM ~15 -> ~0.5 GB); warm-up (read the GGUF in order into the OS cache, then a
    1-token request with the system prompt) makes the first answer after sleep ~5 s instead of 20-50 s.
GGUFs with mixed-precision experts (inference/README.md) need the patched llama.cpp build.
"""
import atexit
import json
import os
import shlex
import shutil
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.environ.get("SMEL_CHAT_CONFIG") or os.path.join(ROOT, "configs", "chat.local.yaml")   # gitignored (*.local.yaml)
LOG = os.path.join(ROOT, "logs", "chat_llama_server.log")
ALIAS = "smel"

DEFAULTS = {"gguf": "", "gguf_dir": "", "llama_server": "", "extra": "",
            "port": 8091, "ctx": 8192, "n_cpu_moe": 99, "threads": 0,
            "spec_draft": -1,            # -1 = auto: 1 for Qwen, 0 for Gemma (no MTP head)
            "sleep_min": 15, "ubatch": 1024, "cache_ram_mb": 1024, "mlock": False, "prefetch": True}
# key -> (type, min, max)
SCHEMA = {"port": (int, 1024, 65535), "ctx": (int, 512, 262144), "n_cpu_moe": (int, 0, 999), "threads": (int, 0, 256),
          "spec_draft": (int, -1, 8), "sleep_min": (int, 0, 1440), "ubatch": (int, 32, 8192),
          "cache_ram_mb": (int, 0, 65536), "mlock": (bool,), "prefetch": (bool,),
          "gguf": (str,), "gguf_dir": (str,), "llama_server": (str,), "extra": (str,)}


def _clean(key, v):
    t, *rng = SCHEMA[key]
    if t is bool:
        return v if isinstance(v, bool) else str(v).strip().lower() in ("1", "true", "yes", "on")
    if t is int:
        v = int(float(v))
        if not rng[0] <= v <= rng[1]:
            raise ValueError(f"{key}: {rng[0]}..{rng[1]}")
        return v
    return str(v or "").strip()


def load_settings():
    s = dict(DEFAULTS)
    if os.path.exists(CONFIG):
        with open(CONFIG, encoding="utf-8") as f:
            for k, v in (yaml.safe_load(f) or {}).items():
                if k in SCHEMA:
                    try:
                        s[k] = _clean(k, v)
                    except (ValueError, TypeError):
                        pass
    return s


def save_settings(values):
    s = load_settings()
    for k, v in values.items():
        if k not in SCHEMA:
            raise ValueError(f"unknown setting {k}")
        s[k] = _clean(k, v)
    os.makedirs(os.path.dirname(CONFIG), exist_ok=True)
    with open(CONFIG, "w", encoding="utf-8") as f:
        yaml.safe_dump(s, f, allow_unicode=True, sort_keys=False)
    return s


def effective(s):
    """Settings with the automatic values resolved."""
    s = dict(s)
    if s["spec_draft"] < 0:
        s["spec_draft"] = 0 if "gemma" in os.path.basename(s["gguf"]).lower() else 1
    if not s["threads"]:
        try:
            import psutil
            s["threads"] = psutil.cpu_count(logical=False) or 4
        except Exception:
            s["threads"] = max(1, (os.cpu_count() or 8) // 2)
    return s


def find_server(s):
    """llama-server: the setting (file or build folder), env SMEL_LLAMA_SERVER, PATH."""
    names = ("llama-server.exe", "llama-server")
    for cand in (s.get("llama_server"), os.environ.get("SMEL_LLAMA_SERVER")):
        if cand and os.path.isfile(cand):
            return cand
        if cand and os.path.isdir(cand):
            hit = next((os.path.join(cand, n) for n in names if os.path.isfile(os.path.join(cand, n))), None)
            if hit:
                return hit
    return shutil.which("llama-server")


def build_command(s, exe):
    s = effective(s)
    cmd = [exe, "-m", s["gguf"], "--alias", ALIAS, "-ngl", "99", "--n-cpu-moe", str(s["n_cpu_moe"]),
           "-t", str(s["threads"]), "-c", str(s["ctx"]), "-np", "1", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0",
           "-ub", str(s["ubatch"]), "--cache-ram", str(s["cache_ram_mb"]), "--ctx-checkpoints", "8",
           "--load-mode", "mmap+mlock" if s["mlock"] else "mmap"]
    if s["sleep_min"] > 0:
        cmd += ["--sleep-idle-seconds", str(s["sleep_min"] * 60)]
    if s["spec_draft"] > 0:
        cmd += ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(s["spec_draft"])]
    return cmd + ["--no-mmproj", "--jinja", "--host", "127.0.0.1", "--port", str(s["port"])] + shlex.split(s["extra"])


def list_ggufs(dirs):
    """GGUF files directly in the given folders (mmproj files skipped)."""
    out, seen = [], set()
    for d in dirs:
        if not d or not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            p = os.path.join(d, f)
            if f.lower().endswith(".gguf") and "mmproj" not in f.lower() and os.path.isfile(p) and p not in seen:
                seen.add(p)
                out.append({"path": p.replace(os.sep, "/"), "name": f, "size_gb": round(os.path.getsize(p) / 2**30, 1)})
    return out


def tail(path, n=15):
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 16384))
            return "\n".join(f.read().decode("utf-8", errors="replace").splitlines()[-n:])
    except OSError:
        return ""


def prefetch_file(path, chunk=64 << 20):
    """Read the file in order into the OS cache. Seconds instead of the tens of seconds that page faults take."""
    if not path or not os.path.exists(path):
        return 0.0
    t0 = time.time()
    buf = bytearray(chunk)
    with open(path, "rb", buffering=0) as f:
        while f.readinto(buf):
            pass
    return time.time() - t0


def _url(s, path):
    return f"http://127.0.0.1:{s['port']}{path}"


def _health(s):
    """'ok' (a sleeping server answers ok and is not woken), 'loading', or None."""
    try:
        with urllib.request.urlopen(_url(s, "/health"), timeout=1.5) as r:
            return "ok" if json.loads(r.read()).get("status") == "ok" else "loading"
    except urllib.error.HTTPError as e:
        return "loading" if e.code == 503 else None
    except Exception:
        return None


def _sleeping(s):
    try:
        with urllib.request.urlopen(_url(s, "/props"), timeout=2) as r:     # /props does not wake the server
            return bool(json.loads(r.read()).get("is_sleeping"))
    except Exception:
        return False


def _port_busy(port):
    try:
        socket.create_connection(("127.0.0.1", port), 0.3).close()
        return True
    except OSError:
        return False


def kill_tree(p):
    if p.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(p.pid)], capture_output=True)
    else:
        import signal
        os.killpg(p.pid, signal.SIGTERM)
    try:
        p.wait(30)
    except subprocess.TimeoutExpired:
        p.kill()


class Model:
    """The llama-server of the chat window: one process, one slot (the conversation's prompt cache)."""

    def __init__(self):
        self.proc, self.started = None, None
        self.lock = threading.RLock()
        self.warm_lock = threading.Lock()
        self.act_lock = threading.Lock()
        self.warm_done = False
        self.active = 0                 # generations in flight
        self.last_warm = None           # result of the last warm-up, for the UI

    def running(self):
        return self.proc is not None and self.proc.poll() is None

    def status(self):
        s = effective(load_settings())
        out = {"state": "stopped", "external": False, "gguf": s["gguf"], "port": s["port"], "spec_draft": s["spec_draft"],
               "warm": self.warm_done, "last_warm": self.last_warm}
        p = self.proc
        if p is not None:
            rc = p.poll()
            if rc is None:
                out.update(pid=p.pid, uptime=time.time() - self.started)
                if _health(s) == "ok":
                    out["state"] = "sleeping" if _sleeping(s) else "ready"
                else:
                    out["state"] = "loading"
            else:
                out.update(state="error" if rc != 0 else "stopped", rc=rc)
                if rc != 0:
                    out["log_tail"] = tail(LOG)
        elif _health(s) == "ok":          # a server started outside the UI on this port
            out.update(state="sleeping" if _sleeping(s) else "ready", external=True)
        return out

    def start(self):
        with self.lock:
            st = self.status()
            if st["state"] in ("loading", "ready", "sleeping"):
                return st
            s = effective(load_settings())
            exe = find_server(s)
            if not exe:
                raise ValueError("llama-server not found: set its path (the patched build for mixed-precision GGUFs)")
            if not s["gguf"] or not os.path.isfile(s["gguf"]):
                raise ValueError(f"GGUF not found: {s['gguf'] or '(none chosen)'}")
            if _port_busy(s["port"]):
                raise ValueError(f"port {s['port']} is busy: another program listens there")
            os.makedirs(os.path.dirname(LOG), exist_ok=True)
            cmd = build_command(s, exe)
            kw = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
            with open(LOG, "wb") as log:
                log.write(("$ " + " ".join(cmd) + "\n").encode())
                log.flush()
                self.proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, **kw)
            self.started, self.warm_done, self.last_warm = time.time(), False, None
            return self.status()

    def stop(self):
        with self.lock:
            p, self.proc, self.warm_done = self.proc, None, False
            if p is not None:
                kill_tree(p)

    # ------------------------------------------------------------------------------------------------ warm-up
    def warm(self, system=""):
        """Wake a sleeping server and page the weights in: read the GGUF in order, then a 1-token request with the
        chat's system prompt (it stays in the slot's cache, so the first real message does not recompute it)."""
        if self.active or not self.warm_lock.acquire(blocking=False):
            return "busy"
        try:
            s = effective(load_settings())
            st = self.status()["state"]
            if st not in ("ready", "sleeping"):
                return "down" if st != "loading" else "loading"
            sleeping = st == "sleeping"
            if self.warm_done and not sleeping:
                return "skipped"
            t0, read_s = time.time(), 0.0
            if s["prefetch"]:
                read_s = prefetch_file(s["gguf"])
            msgs = ([{"role": "system", "content": system}] if system.strip() else []) + [{"role": "user", "content": "."}]
            body = {"model": ALIAS, "messages": msgs, "max_tokens": 1, "stream": False, "temperature": 0.0, "id_slot": 0,
                    "cache_prompt": True, "chat_template_kwargs": {"enable_thinking": False}}
            req = urllib.request.Request(_url(s, "/v1/chat/completions"), data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=180) as r:
                    ok = r.status == 200
            except Exception:
                ok = False
            self.warm_done = ok
            self.last_warm = {"ok": ok, "was_sleeping": sleeping, "read_s": round(read_s, 1),
                              "total_s": round(time.time() - t0, 1)}
            return "warm" if ok else "down"
        finally:
            self.warm_lock.release()

    def warm_async(self, system=""):
        threading.Thread(target=self.warm, args=(system,), daemon=True).start()

    # ------------------------------------------------------------------------------------------------ chat
    def prepare_chat(self, req):
        """Validated llama-server request body. Raises ValueError for a bad request."""
        msgs = req.get("messages")
        if not isinstance(msgs, list) or not msgs:
            raise ValueError("messages: a non-empty list")
        for m in msgs:
            if not isinstance(m, dict) or m.get("role") not in ("user", "assistant") or not isinstance(m.get("content"), str):
                raise ValueError("messages: {role: user|assistant, content: text}")
        system = str(req.get("system") or "").strip()
        if system:
            msgs = [{"role": "system", "content": system}] + msgs
        try:
            return {"model": ALIAS, "messages": msgs, "stream": True, "id_slot": 0, "cache_prompt": True,
                    "temperature": float(req.get("temperature", 0.7)), "top_p": float(req.get("top_p", 0.8)),
                    "top_k": int(req.get("top_k", 20)), "min_p": 0.0,
                    "presence_penalty": float(req.get("presence_penalty", 1.5)),
                    "max_tokens": int(req.get("max_tokens", 512)),
                    "chat_template_kwargs": {"enable_thinking": bool(req.get("thinking"))}}
        except (TypeError, ValueError):
            raise ValueError("sampling parameters must be numbers")

    def stream_chat(self, body, write):
        """Copy llama-server's SSE stream to write(bytes). A write error (the browser left) closes the upstream
        connection, which makes llama-server stop generating."""
        s = effective(load_settings())
        req = urllib.request.Request(_url(s, "/v1/chat/completions"), data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with self.act_lock:
            self.active += 1
        try:
            try:
                with urllib.request.urlopen(req, timeout=600) as r:
                    for line in r:
                        write(line)
            except urllib.error.HTTPError as e:
                msg = e.read().decode("utf-8", errors="replace")[:500]
                write(("data: " + json.dumps({"error": f"llama-server {e.code}: {msg}"}) + "\n\n").encode())
        finally:
            with self.act_lock:
                self.active -= 1


MODEL = Model()
atexit.register(MODEL.stop)
