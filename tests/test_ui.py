"""ui/server.py: projects, data check, stage commands, jobs and the HTTP API (no GPU, no model)."""
import json
import os
import sys
import threading
import time
import urllib.request

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ui"))
import server  # noqa: E402


@pytest.fixture
def projects(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "PROJECTS", str(tmp_path / "projects"))
    return tmp_path


def test_project_settings_and_commands(projects):
    cfg = server.create_project("run1", "gemma4", model_dir=str(projects / "model"))
    assert cfg["paths"]["work_dir"] == "work/run1" and cfg["ui"]["model"] == "gemma4"
    assert cfg["model"] and cfg["train"]["r"] == 16          # the rest comes from configs/gemma4.yaml
    with pytest.raises(ValueError):
        server.create_project("run1", "qwen")
    with pytest.raises(ValueError):
        server.create_project("../x", "qwen")
    assert [p["name"] for p in server.list_projects()] == ["run1"]

    cfg = server.update_settings("run1", {"train.lr": "2e-4", "select.L_override": "16", "train.epochs": "2"})
    assert cfg["train"]["lr"] == 2e-4 and cfg["select"]["L_override"] == 16 and cfg["train"]["epochs"] == 2
    cfg = server.update_settings("run1", {"select.L_override": ""})
    assert cfg["select"]["L_override"] is None
    with pytest.raises(ValueError):
        server.update_settings("run1", {"paths.model_dir": ""})
    with pytest.raises(ValueError):
        server.update_settings("run1", {"train.dropout": "0.1"})

    conf = server.project_path("run1")
    (c,) = server.stage_commands("run1", "train", {"tag": "a1", "max_steps": "5", "resume": True})
    assert c[1].endswith("train.py") and c[2:4] == ["--config", conf]
    assert c[4:] == ["--tag", "a1", "--max-steps", "5", "--resume"]
    (c,) = server.stage_commands("run1", "download", {})
    assert c[c.index("--repo") + 1] == "google/gemma-4-26B-A4B-it"
    with pytest.raises(ValueError):
        server.stage_commands("run1", "train", {"tag": "a b; rm"})
    with pytest.raises(ValueError):
        server.stage_commands("run1", "eval", {})
    cmds = server.stage_commands("run1", "pipeline", {"from": "cache", "tag": "a1"})
    assert [os.path.basename(c[1]) for c in cmds[:-1]] == ["cache.py", "pack.py", "train.py"]
    assert cmds[-1] == ("eval_last", "a1")

    st = server.project_status("run1")
    assert not st["status"]["model"]["ok"] and not st["status"]["select"]["ok"]
    assert st["settings"]["train.lr"] == 2e-4


def test_check_dataset(tmp_path):
    p = tmp_path / "d.jsonl"
    good = {"messages": [{"role": "user", "content": "Привет"}, {"role": "assistant", "content": "Здравствуйте! Чем помочь?"}]}
    rows = [good] * 3 + [{"messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "Ок.Да"}]},
                         {"messages": [{"role": "user", "content": "only user"}]}, {"text": "plain"}]
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\nnot json\n", encoding="utf-8")
    r = server.check_dataset(str(p))
    assert r["rows"] == 7 and r["chats"] == 4 and r["texts"] == 1 and r["n_errors"] == 2
    assert r["duplicates"] == 2 and r["assistant_turns"] == 4
    assert r["short_answers"] == 0.25 and r["no_space_after_punct"] == 0.25
    assert {"errors", "few", "short", "nospace", "dups"} <= set(r["warnings"])
    assert not server.check_dataset(str(tmp_path / "missing.jsonl"))["exists"]


def test_job_runs_and_stops(tmp_path):
    j = server.Job("p", "x", [[sys.executable, "-c", "print('hello'); import sys; sys.exit(0)"],
                              [sys.executable, "-c", "print('second')"]], str(tmp_path / "a.log"))
    j.run()
    assert j.state == "done" and j.rc == 0
    info = j.info()
    assert "hello" in info["log"] and "second" in info["log"] and info["offset"] == len(j.buf)
    assert j.info(info["offset"])["log"] == ""
    assert "hello" in open(j.log_path, encoding="utf-8").read()

    j = server.Job("p", "x", [[sys.executable, "-c", "import time; print('go', flush=True); time.sleep(60)"],
                              [sys.executable, "-c", "print('never')"]], str(tmp_path / "b.log"))
    th = threading.Thread(target=j.run)
    th.start()
    t0 = time.time()
    while "go" not in j.info()["log"] and time.time() - t0 < 30:
        time.sleep(0.1)
    j.stop()
    th.join(30)
    assert j.state == "stopped" and "never" not in j.info()["log"]


def test_http_api(projects):
    srv = server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"

    def call(method, path, body=None, raw=None):
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        req = urllib.request.Request(base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    try:
        code, page = call("GET", "/")
        assert code == 200 and b"SMEL FT" in page
        code, st = call("GET", "/api/state")
        assert code == 200 and set(json.loads(st)["models"]) == {"qwen", "gemma4"}
        code, _ = call("POST", "/api/projects", {"name": "web1", "model": "qwen"})
        assert code == 200
        code, _ = call("POST", "/api/projects", {"name": "web1", "model": "qwen"})
        assert code == 400
        code, _ = call("GET", "/api/projects/nope")
        assert code == 404
        line = json.dumps({"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]})
        code, r = call("PUT", "/api/projects/web1/upload?kind=domain", raw=(line + "\n").encode())
        dest = os.path.join(server.ROOT, "data", "web1", "domain.jsonl")
        try:
            assert code == 200 and json.loads(r)["chats"] == 1
            cfg = server.read_project("web1")
            assert cfg["paths"]["domain_data"] == "data/web1/domain.jsonl"
        finally:
            os.remove(dest)
            os.rmdir(os.path.dirname(dest))
        code, _ = call("POST", "/api/projects/web1/run", {"stage": "nope"})
        assert code == 400
        code, _ = call("GET", "/api/projects/web1/report?file=../../README.md")
        assert code == 404                                    # only files inside the project's reports dir
    finally:
        srv.shutdown()
