"""ui/chat_backend.py and the chat routes of ui/server.py, against a fake llama-server (no GPU, no model)."""
import json
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.request

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "ui"))
import chat_backend as cb  # noqa: E402
import server  # noqa: E402

FAKE = os.path.join(HERE, "fake_llama_server.py")


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def chat(tmp_path, monkeypatch):
    monkeypatch.setattr(cb, "CONFIG", str(tmp_path / "chat.yaml"))
    monkeypatch.setattr(cb, "LOG", str(tmp_path / "chat.log"))
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x" * 1000)
    port = free_port()
    cb.save_settings({"gguf": str(gguf), "port": port})
    monkeypatch.setattr(cb, "find_server", lambda s: "fake")
    monkeypatch.setattr(cb, "build_command", lambda s, exe: [sys.executable, FAKE, "--port", str(s["port"])])
    model = cb.Model()
    monkeypatch.setattr(cb, "MODEL", model)
    yield model, port
    model.stop()


def wait_state(model, states, timeout=30):
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = model.status()
        if st["state"] in states:
            return st
        time.sleep(0.2)
    raise AssertionError(f"state {model.status()['state']}, wanted {states}")


def posts(port):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/_posts", timeout=5) as r:
        return json.loads(r.read())


def test_settings_and_command(tmp_path, monkeypatch):
    monkeypatch.setattr(cb, "CONFIG", str(tmp_path / "chat.yaml"))
    with pytest.raises(ValueError):
        cb.save_settings({"port": 80})
    with pytest.raises(ValueError):
        cb.save_settings({"nope": 1})
    s = cb.save_settings({"gguf": "C:/x/Qwen3.6-azusa.gguf", "n_cpu_moe": "40", "mlock": "false"})
    assert s["n_cpu_moe"] == 40 and s["mlock"] is False and cb.load_settings()["n_cpu_moe"] == 40
    j = " ".join(cb.build_command({**cb.DEFAULTS, "gguf": "C:/x/Qwen3.6-azusa.gguf", "threads": 8, "mlock": True,
                                   "extra": "--no-warmup"}, "llama-server"))
    assert "--n-cpu-moe 99" in j and "-ub 1024" in j and "-ctk q8_0" in j and "--load-mode mmap+mlock" in j
    assert "--spec-type draft-mtp --spec-draft-n-max 1" in j        # auto: Qwen gets the MTP draft
    assert "--sleep-idle-seconds 900" in j and "--no-warmup" in j and "-np 1" in j and "-t 8" in j
    gem = " ".join(cb.build_command({**cb.DEFAULTS, "gguf": "gemma-4-26B.gguf", "threads": 4, "sleep_min": 0,
                                     "mlock": False}, "llama-server"))
    assert "--spec-type" not in gem and "--sleep-idle-seconds" not in gem and "--load-mode mmap --" in gem
    assert cb.DEFAULTS["mlock"] is False
    two = " ".join(cb.build_command({**cb.DEFAULTS, "gguf": "q.gguf", "threads": 4, "spec_draft": 2}, "llama-server"))
    assert "--spec-draft-n-max 2" in two


def test_list_ggufs(tmp_path):
    for n in ("a.gguf", "mmproj-b.gguf", "c.txt"):
        (tmp_path / n).write_bytes(b"x")
    assert [g["name"] for g in cb.list_ggufs([str(tmp_path), "", str(tmp_path / "none")])] == ["a.gguf"]


def test_start_errors(chat, tmp_path):
    model, port = chat
    cb.save_settings({"gguf": str(tmp_path / "missing.gguf")})
    with pytest.raises(ValueError, match="GGUF not found"):
        model.start()
    cb.save_settings({"gguf": str(tmp_path / "m.gguf")})
    busy = socket.socket()
    busy.bind(("127.0.0.1", port))
    busy.listen()
    try:
        with pytest.raises(ValueError, match="busy"):
            model.start()
    finally:
        busy.close()


def test_lifecycle_and_warmup(chat):
    model, port = chat
    assert model.status()["state"] == "stopped"
    model.start()
    wait_state(model, {"sleeping"})                    # the fake starts asleep
    assert model.warm("You are Azusa.") == "warm"
    p = posts(port)
    assert len(p) == 1 and p[0]["max_tokens"] == 1 and p[0]["id_slot"] == 0
    assert p[0]["messages"][0] == {"role": "system", "content": "You are Azusa."}
    st = model.status()
    assert st["state"] == "ready" and st["warm"] and st["last_warm"]["was_sleeping"]
    assert model.warm("You are Azusa.") == "skipped"   # awake and warm: nothing to do
    assert len(posts(port)) == 1
    model.stop()
    assert model.status()["state"] == "stopped" and model.warm("") == "down"


def test_prepare_chat(chat):
    model, _ = chat
    with pytest.raises(ValueError):
        model.prepare_chat({"messages": []})
    with pytest.raises(ValueError):
        model.prepare_chat({"messages": [{"role": "tool", "content": "x"}]})
    with pytest.raises(ValueError):
        model.prepare_chat({"messages": [{"role": "user", "content": "x"}], "top_k": "many"})
    b = model.prepare_chat({"messages": [{"role": "user", "content": "hi"}], "system": " be brief "})
    assert b["messages"][0] == {"role": "system", "content": "be brief"} and b["stream"] and b["cache_prompt"]
    assert b["id_slot"] == 0 and b["top_k"] == 20 and b["chat_template_kwargs"] == {"enable_thinking": False}


def test_http_chat_and_job_conflict(chat):
    model, port = chat
    srv = server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"

    def call(method, path, body=None):
        req = urllib.request.Request(base + path, data=json.dumps(body).encode() if body is not None else None,
                                     method=method, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    msgs = [{"role": "user", "content": "hi"}]
    try:
        code, page = call("GET", "/chat")
        assert code == 200 and b"SMEL FT" in page
        code, r = call("POST", "/api/chat/completions", {"messages": msgs})
        assert code == 409 and b"start it first" in r                     # not started
        code, r = call("POST", "/api/chat/completions", {"messages": []})
        assert code == 400
        code, r = call("GET", "/api/chat/state")
        st = json.loads(r)
        assert code == 200 and st["status"]["state"] == "stopped" and st["settings"]["port"] == port
        assert st["server_exe"] == "fake"
        assert call("POST", "/api/chat/start")[0] == 200
        wait_state(model, {"sleeping", "ready"})
        # a stage must not start while the model holds the GPU
        with pytest.raises(RuntimeError, match="chat model"):
            server.start_job("x", "probe", {})
        code, r = call("POST", "/api/chat/completions", {"messages": msgs, "system": "S", "top_k": 7})
        assert code == 200
        events = [json.loads(l[6:]) for l in r.decode().splitlines() if l.startswith("data: {")]
        text = "".join(e["choices"][0]["delta"].get("content", "") for e in events)
        assert text == "Hello!" and events[-1]["timings"]["cache_n"] == 7
        sent = posts(port)[-1]
        assert sent["messages"][0] == {"role": "system", "content": "S"} and sent["top_k"] == 7 and sent["stream"]
        assert call("POST", "/api/chat/stop")[0] == 200
        assert model.status()["state"] == "stopped"
    finally:
        srv.shutdown()
