"""A stand-in for llama-server in the chat UI tests: /health, /props (starts asleep, wakes on the first completion),
/v1/chat/completions (streamed or not), and /_posts with the request bodies it received."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, required=True)
args, _ = ap.parse_known_args()
STATE = {"sleeping": True, "posts": []}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def reply(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/health":
            return self.reply(200, {"status": "ok"})
        if self.path == "/props":
            return self.reply(200, {"is_sleeping": STATE["sleeping"]})
        if self.path == "/_posts":
            return self.reply(200, STATE["posts"])
        self.reply(404, {})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        STATE["posts"].append(body)
        STATE["sleeping"] = False
        if not body.get("stream"):
            return self.reply(200, {"choices": [{"message": {"content": "."}}]})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for piece in ("Hel", "lo", "!"):
            self.wfile.write(("data: " + json.dumps({"choices": [{"delta": {"content": piece}}]}) + "\n\n").encode())
            self.wfile.flush()
        last = {"choices": [{"delta": {}, "finish_reason": "stop"}],
                "timings": {"prompt_n": 5, "cache_n": 7, "predicted_n": 3, "predicted_per_second": 21.5}}
        self.wfile.write(("data: " + json.dumps(last) + "\n\ndata: [DONE]\n\n").encode())
        self.wfile.flush()


ThreadingHTTPServer(("127.0.0.1", args.port), H).serve_forever()
