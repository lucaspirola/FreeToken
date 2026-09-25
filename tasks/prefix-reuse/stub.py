#!/usr/bin/env python3
"""Capture stub for the /clear probe (capture.sh): logs path, headers and body of every request to
captures.jsonl (CAPTURE env) and answers "ok" in the protocol the path expects:
/v1/messages (Anthropic, SSE or not), /v1/chat/completions (OpenAI, SSE or not),
/v1/responses (OpenAI Responses, SSE or not), models and count_tokens probes."""
import json, os, sys, time, uuid, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOG = os.environ.get("CAPTURE", "captures.jsonl")
PORT = int(os.environ.get("PORT", "18080"))
lock = threading.Lock()


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _log(self, body):
        with lock, open(LOG, "a") as f:
            f.write(json.dumps({"t": time.time(), "method": self.command, "path": self.path,
                                "headers": dict(self.headers.items()), "body": body}) + "\n")

    def _json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def _sse(self, events):
        self.send_response(200); self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache"); self.send_header("Connection", "close"); self.end_headers()
        for ev, data in events:
            s = (f"event: {ev}\n" if ev else "") + f"data: {data if isinstance(data, str) else json.dumps(data)}\n\n"
            self.wfile.write(s.encode()); self.wfile.flush()
        self.close_connection = True

    def do_GET(self):
        self._log(None)
        if "models" in self.path:
            return self._json({"object": "list", "data": [{"id": m, "object": "model", "created": 0, "owned_by": "stub",
                                                            "type": "model", "display_name": m}
                                                           for m in ("nemotron-3.5-lightning", "claude-sonnet-4-5", "stub")],
                               "has_more": False})
        return self._json({"ok": True})

    def do_HEAD(self):
        self.send_response(200); self.send_header("Content-Length", "0"); self.end_headers()

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        try:
            body = json.loads(raw) if raw else None
        except ValueError:
            body = raw.decode(errors="replace")
        self._log(body)
        b = body if isinstance(body, dict) else {}
        model = b.get("model", "stub")
        p = self.path.split("?")[0]
        if p.endswith("count_tokens"):
            return self._json({"input_tokens": 10})
        if p.endswith("/messages"):
            mid = "msg_" + uuid.uuid4().hex[:20]
            usage = {"input_tokens": 10, "output_tokens": 1, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
            if not b.get("stream"):
                return self._json({"id": mid, "type": "message", "role": "assistant", "model": model,
                                   "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn",
                                   "stop_sequence": None, "usage": usage})
            return self._sse([
                ("message_start", {"type": "message_start", "message": {"id": mid, "type": "message", "role": "assistant",
                                   "model": model, "content": [], "stop_reason": None, "stop_sequence": None, "usage": usage}}),
                ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
                ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "ok"}}),
                ("content_block_stop", {"type": "content_block_stop", "index": 0}),
                ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                                   "usage": {"output_tokens": 1}}),
                ("message_stop", {"type": "message_stop"})])
        if p.endswith("/chat/completions"):
            cid = "chatcmpl-" + uuid.uuid4().hex[:20]
            usage = {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11}
            if not b.get("stream"):
                return self._json({"id": cid, "object": "chat.completion", "created": int(time.time()), "model": model,
                                   "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                                   "usage": usage})
            base = {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()), "model": model}
            return self._sse([
                (None, {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]}),
                (None, {**base, "choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": None}]}),
                (None, {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
                (None, {**base, "choices": [], "usage": usage}),
                (None, "[DONE]")])
        if p.endswith("/responses"):
            rid = "resp_" + uuid.uuid4().hex[:20]; iid = "msg_" + uuid.uuid4().hex[:20]
            item = {"type": "message", "id": iid, "status": "completed", "role": "assistant",
                    "content": [{"type": "output_text", "text": "ok", "annotations": []}]}
            usage = {"input_tokens": 10, "output_tokens": 1, "total_tokens": 11,
                     "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}}
            resp = {"id": rid, "object": "response", "created_at": int(time.time()), "status": "completed", "model": model,
                    "output": [item], "usage": usage}
            if not b.get("stream"):
                return self._json(resp)
            inprog = {**resp, "status": "in_progress", "output": []}
            return self._sse([
                ("response.created", {"type": "response.created", "response": inprog}),
                ("response.output_item.added", {"type": "response.output_item.added", "output_index": 0,
                                                "item": {**item, "status": "in_progress", "content": []}}),
                ("response.content_part.added", {"type": "response.content_part.added", "item_id": iid, "output_index": 0,
                                                 "content_index": 0, "part": {"type": "output_text", "text": "", "annotations": []}}),
                ("response.output_text.delta", {"type": "response.output_text.delta", "item_id": iid, "output_index": 0,
                                                "content_index": 0, "delta": "ok"}),
                ("response.output_text.done", {"type": "response.output_text.done", "item_id": iid, "output_index": 0,
                                               "content_index": 0, "text": "ok"}),
                ("response.content_part.done", {"type": "response.content_part.done", "item_id": iid, "output_index": 0,
                                                "content_index": 0, "part": {"type": "output_text", "text": "ok", "annotations": []}}),
                ("response.output_item.done", {"type": "response.output_item.done", "output_index": 0, "item": item}),
                ("response.completed", {"type": "response.completed", "response": resp})])
        return self._json({"ok": True})


ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
