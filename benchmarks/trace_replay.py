#!/usr/bin/env python3
"""Replay a captured FreeToken request trace against a running server.

    ft serve ... --trace-dir /var/tmp/ft-trace           # capture (see docs/switchyard.md)
    python benchmarks/trace_replay.py --trace /var/tmp/ft-trace \
        --base-url http://127.0.0.1:1919 --model nemotron-3.5-lightning --out replay.json

The Switchyard soak drives *synthetic* traffic: a fixed scenario mix, fixed prompt
lengths, a closed 16-client loop. That is the right shape for a regression gate and the
wrong shape for a question like "does this scheduler change help the traffic we actually
serve". This replays the real thing -- real inter-arrival times, real session
interleaving, real prompt-length and cached-fraction distributions -- and prints its own
p50/p95/p99, tok/s and error counts **side by side with the trace's originals**, so the
comparison is against measured behaviour rather than against a synthetic baseline.

How prompts come back without the text
--------------------------------------
A trace stores no prompt text by default (``--trace-include-text`` opts in; then this
replays it verbatim and everything below is skipped). What it stores is the *prefix
chain* -- ``chain[i] = sha256(chain[i-1] || message_i)`` -- plus each message's character
count and role. Two requests share exactly the first ``m`` messages iff their chains agree
for ``m`` entries, which is the boundary the radix prefix cache keys on.

So message ``i`` is regenerated as deterministic filler seeded by ``chain[i]``, of a length
that is **a pure function of that message's own recorded ``msg_chars``**:

    words_i = max(1, round(msg_chars[i] * scale))

Purity is the load-bearing property. Turn ``k+1`` of a conversation contains turn ``k``'s
messages, so if a message's regenerated length depended on the request it sits in (an
apportionment of the request's total, say), the same message would come out at two
different lengths in two turns and the shared prefix would break at message 0 -- the
replay would run at ~0 % prefix reuse against a trace that measured 74 %, and the whole
exercise would measure the harness. With the pure rule, a shared prefix is byte-identical
and the cache hits at the recorded boundary.

The price is that a request's *total* length is then not individually adjustable. ``scale``
is one global constant, fitted once so the median replayed prompt matches the median
traced one; the residual per-request error is the traffic's own chars-per-token variance
and is reported as ``prompt_tokens_err_p50/p95``. Read it before trusting a run: above
~10 % the trace mixes prompt kinds (code, CJK, base64) too different for one constant, and
you want ``--trace-include-text`` or a per-kind split.

Session affinity is preserved: each traced session becomes one replay session bound with
``x-switchyard-session-id``, and its turns are issued strictly in order, one at a time.
A turn whose predecessor is still running is held (counted as ``session_queue_ms``) rather
than overlapped -- real conversations are sequential, and overlapping them would destroy
the very prefix structure this reconstructs.

Stdlib only (urllib + threads): it runs on a machine with no torch, no CUDA and no venv.

``--dry-run`` builds every prompt and reports reconstruction fidelity without touching a
server -- what the CPU tests exercise.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import queue
import random
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parent.parent


def _load_request_trace():
    """Import ``freetoken.server.request_trace`` by path.

    Not ``import freetoken.server.request_trace``: that executes
    ``freetoken/server/__init__.py`` -> ``launch`` -> torch, and this script must run
    (and be tested) on a box with no torch installed. The module itself is stdlib-only.
    """
    path = _REPO / "python" / "freetoken" / "server" / "request_trace.py"
    spec = importlib.util.spec_from_file_location("_ft_request_trace", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RT = _load_request_trace()

# 256 short common words. Filler only needs to be deterministic, cheap to tokenize and
# free of anything that could look like a real prompt in a shared trace.
VOCAB = (
    "the of and to in a is that it for as with on was by at be this from or an are not "
    "but had has have they you all we can her his she him one out so up if no do about "
    "which when what who will more time than into other only some could them then these "
    "two may first also new like over such our any most after well way even back where "
    "much go good come take see know just year work think day give many made under while "
    "here own very own state group form part place case point number great small large "
    "long high next early young little own public same able best free full hard head "
    "home hour idea last late left life line list live local main major mean mind month "
    "name near need night north open order paper party past plan play power press price "
    "quiet read real reason record right river road room rule run school sea season second "
    "sense series service set side sign since site size social society sort sound source "
    "south space speak special sport staff stage stand start stay step still stop story "
    "street strong student study subject success such sure system table talk teacher team "
    "term test text thing third though thought three through today together top total "
    "town trade train travel tree true try turn type union unit until use usually value "
    "various view village voice vote wait walk wall want war watch water week west white "
    "whole wide wife wind window wine winter wish woman wonder wood word world write year"
).split()
VOCAB = [w for w in VOCAB if w.isascii()]
#: Mean characters a filler word contributes, including its separating space. Used only as
#: the starting point for the ``scale`` fit.
_CHARS_PER_WORD = statistics.mean(len(w) for w in VOCAB) + 1.0

CHAT_ROUTE = "/v1/chat/completions"
COMPLETIONS_ROUTE = "/v1/completions"


# --------------------------------------------------------------------------- prompts


class FillerBook:
    """Deterministic filler text, memoised by ``(chain_hash, words)``."""

    def __init__(self, cap: int = 20_000):
        self._cache: dict[tuple[str, int], str] = {}
        self._cap = cap

    def text(self, seed_hex: str, words: int) -> str:
        key = (seed_hex, words)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        # Seeded from the string itself (CPython hashes it with sha512, deterministic
        # across runs and platforms), so calibration seeds that are not hex work too.
        rng = random.Random(seed_hex or "")
        out = " ".join(rng.choice(VOCAB) for _ in range(max(1, words)))
        if len(self._cache) < self._cap:
            self._cache[key] = out
        return out


class PromptBuilder:
    """Turn a trace record into the messages to send.

    ``scale`` converts a message's recorded character count into filler words and is the
    single global knob; see the module docstring for why it must not vary per request.
    """

    def __init__(self, scale: float, book: FillerBook | None = None):
        self.scale = scale
        self.book = book or FillerBook()

    def words_for(self, chars: int) -> int:
        return max(1, int(round(chars * self.scale)))

    def total_words(self, rec: dict[str, Any]) -> int:
        return sum(self.words_for(c) for c in rec.get("msg_chars") or [0])

    def build(self, rec: dict[str, Any]) -> list[dict[str, str]] | str:
        """Messages for a chat record, or the prompt string for a completions record."""
        stored = rec.get("messages")
        if stored:
            return stored
        chain = rec.get("msg_chain") or []
        chars = rec.get("msg_chars") or []
        roles = rec.get("msg_roles") or []
        if rec.get("route") == COMPLETIONS_ROUTE and len(chain) == 1:
            return self.book.text(chain[0], self.words_for(chars[0]))
        out = []
        for i, h in enumerate(chain):
            role = roles[i] if i < len(roles) else ("system" if i == 0 else "user")
            if role == "prompt":
                role = "user"
            n = self.words_for(chars[i] if i < len(chars) else 0)
            out.append({"role": role, "content": self.book.text(h, n)})
        return out or [{"role": "user", "content": "hello"}]


def predict_tokens(rec: dict[str, Any], builder: "PromptBuilder", cal: "Calibration") -> float:
    """What the server will report as ``prompt_tokens`` for this record's replayed prompt.

    ``per_message`` is not a rounding detail: a chat template wraps *every* turn in role
    and delimiter tokens, so a 40-message conversation pays it forty times. Fitting only a
    slope and one intercept mis-sized this by 11 % on a four-turn trace, all of it in the
    long conversations -- exactly the requests whose length matters most.
    """
    return (cal.tokens_per_word * builder.total_words(rec)
            + cal.overhead
            + cal.per_message * len(rec.get("msg_chars") or []))


class Calibration:
    """tokens = ``tokens_per_word``*words + ``overhead`` + ``per_message``*messages."""

    __slots__ = ("tokens_per_word", "overhead", "per_message")

    def __init__(self, tokens_per_word: float, overhead: float, per_message: float):
        self.tokens_per_word = tokens_per_word
        self.overhead = overhead
        self.per_message = per_message

    def as_dict(self) -> dict[str, float]:
        return {"tokens_per_word": self.tokens_per_word, "overhead": self.overhead,
                "per_message": self.per_message}


def fit_scale(records: list[dict[str, Any]], cal: "Calibration",
              lo: float = 1e-3, hi: float = 4.0, iters: int = 48) -> float:
    """Solve for the one ``scale`` whose median replayed prompt length matches the trace.

    ``predict_tokens`` is non-decreasing in ``scale`` for every record, so the median ratio
    is too and a bisection is exact to within ``(hi-lo)/2**iters``. Records with no
    recorded ``prompt_tokens`` are ignored.
    """
    usable = [r for r in records if (r.get("prompt_tokens") or 0) > 0 and r.get("msg_chars")]
    if not usable:
        return 1.0 / _CHARS_PER_WORD

    def median_ratio(scale: float) -> float:
        pb = PromptBuilder(scale)
        return statistics.median(
            predict_tokens(r, pb, cal) / r["prompt_tokens"] for r in usable)

    for _ in range(iters):
        mid = (lo + hi) / 2
        if median_ratio(mid) < 1.0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# --------------------------------------------------------------------------- client


class Client:
    """Minimal OpenAI-compatible client. urllib, so no dependency beyond the stdlib."""

    def __init__(self, base_url: str, timeout: float, session_header: str,
                 api_key: str | None = None):
        self.base = base_url.rstrip("/")
        self.timeout = timeout
        self.session_header = session_header
        self.api_key = api_key

    def _post(self, route: str, body: dict[str, Any], session: str | None):
        data = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if session:
            headers[self.session_header] = session
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.base + route, data=data, headers=headers)
        return urllib.request.urlopen(req, timeout=self.timeout)

    def send(self, route: str, body: dict[str, Any], session: str | None) -> dict[str, Any]:
        """Issue one request; return the observed metrics.

        Streaming is preferred because it is the only way to see TTFT. ``stream_options
        .include_usage`` brings the usage block back on the terminal chunk, which is where
        ``prompt_tokens_details.cached_tokens`` lives (needs ``--enable-cache-report``).
        """
        t0 = time.time()
        out: dict[str, Any] = {"status": "ok", "ttft_ms": None, "output_tokens": 0,
                               "prompt_tokens": None, "cached_tokens": None,
                               "finish_reason": None, "error_code": None}
        try:
            resp = self._post(route, body, session)
        except urllib.error.HTTPError as exc:
            out["status"] = "error"
            out["http_status"] = exc.code
            try:
                err = json.loads(exc.read().decode("utf-8", "replace"))
                out["error_code"] = (err.get("error") or {}).get("code")
            except Exception:  # noqa: BLE001
                pass
            out["duration_ms"] = (time.time() - t0) * 1e3
            return out
        except Exception as exc:  # noqa: BLE001 -- timeouts, resets: they are the result
            out["status"] = "error"
            out["error_code"] = type(exc).__name__
            out["duration_ms"] = (time.time() - t0) * 1e3
            return out
        try:
            if body.get("stream"):
                self._read_stream(resp, out, t0)
            else:
                payload = json.loads(resp.read().decode("utf-8", "replace"))
                out["ttft_ms"] = (time.time() - t0) * 1e3
                self._absorb_usage(payload.get("usage"), out)
                ch = (payload.get("choices") or [{}])[0]
                out["finish_reason"] = ch.get("finish_reason")
        except Exception as exc:  # noqa: BLE001
            out["status"] = "error"
            out["error_code"] = type(exc).__name__
        finally:
            resp.close()
        out["duration_ms"] = (time.time() - t0) * 1e3
        return out

    def _read_stream(self, resp, out: dict[str, Any], t0: float) -> None:
        deltas = 0
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except ValueError:
                continue
            if "error" in chunk:
                out["status"] = "error"
                out["error_code"] = (chunk.get("error") or {}).get("code")
                continue
            self._absorb_usage(chunk.get("usage"), out)
            for ch in chunk.get("choices") or []:
                d = ch.get("delta") or {}
                if d.get("content") or d.get("reasoning_content"):
                    if out["ttft_ms"] is None:
                        out["ttft_ms"] = (time.time() - t0) * 1e3
                    deltas += 1
                if ch.get("finish_reason"):
                    out["finish_reason"] = ch["finish_reason"]
        # usage is authoritative when present; the delta count is the fallback.
        if not out["output_tokens"]:
            out["output_tokens"] = deltas

    @staticmethod
    def _absorb_usage(usage: dict[str, Any] | None, out: dict[str, Any]) -> None:
        if not usage:
            return
        if usage.get("completion_tokens") is not None:
            out["output_tokens"] = usage["completion_tokens"]
        if usage.get("prompt_tokens") is not None:
            out["prompt_tokens"] = usage["prompt_tokens"]
        details = usage.get("prompt_tokens_details") or {}
        if details.get("cached_tokens") is not None:
            out["cached_tokens"] = details["cached_tokens"]


# --------------------------------------------------------------------------- replay


def build_body(rec: dict[str, Any], prompt: Any, model: str | None,
               stream: bool, max_tokens_cap: int | None) -> tuple[str, dict[str, Any]]:
    route = rec.get("route") or CHAT_ROUTE
    body: dict[str, Any] = {"model": model or rec.get("model") or "default"}
    if route == COMPLETIONS_ROUTE and isinstance(prompt, str):
        body["prompt"] = prompt
    else:
        route = CHAT_ROUTE
        body["messages"] = prompt if isinstance(prompt, list) else [
            {"role": "user", "content": str(prompt)}]
    mt = rec.get("max_tokens") or rec.get("output_tokens") or 128
    if max_tokens_cap:
        mt = min(mt, max_tokens_cap)
    body["max_tokens"] = int(mt)
    for k, v in (rec.get("sampling") or {}).items():
        # Only forward what the wire actually takes; the trace may carry engine-side names.
        if k in ("temperature", "top_p", "top_k", "presence_penalty", "frequency_penalty",
                 "repetition_penalty", "stop", "reasoning_effort") and v is not None:
            body[k] = v
    if stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    return route, body


class Replayer:
    """Issue the trace against the server on a bounded worker pool.

    Two rules, and they interact:

      * a request goes out when the trace says it did (``--speed`` scales the gaps);
      * turns of ONE session never overlap. A conversation is sequential -- turn k+1's
        prompt is turn k's prompt plus its answer -- so overlapping them would ask the
        server for a prefix that does not exist yet and would not reproduce the recorded
        cache hits. A turn whose predecessor is still running waits, and the wait is
        reported as ``session_queue_ms`` rather than hidden.

    A thread per in-flight request rather than per session: a 20-minute trace has thousands
    of sessions and only tens of concurrent requests, and one OS thread per conversation
    would spend the run's memory on idle stacks. The pool is ``--max-inflight`` wide; when a
    session's turn finishes, its next queued turn becomes runnable.
    """

    def __init__(self, records: list[dict[str, Any]], builder: PromptBuilder,
                 client: Client | None, args: argparse.Namespace):
        self.records = records
        self.builder = builder
        self.client = client
        self.args = args
        self.results: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._ready: "queue.Queue" = queue.Queue()
        self._pending: dict[str, deque] = {}   # session -> turns held behind a running one
        self._busy: set[str] = set()
        self._left = len(records)
        self._done = threading.Condition(self._lock)

    # -- scheduling ---------------------------------------------------------

    def _submit(self, rec: dict[str, Any], due: float) -> None:
        sid = rec.get("session")
        if not sid:
            self._ready.put((None, rec, due))
            return
        with self._lock:
            if sid in self._busy:
                self._pending.setdefault(sid, deque()).append((rec, due))
                return
            self._busy.add(sid)
        self._ready.put((sid, rec, due))

    def _worker(self) -> None:
        while True:
            item = self._ready.get()
            if item is None:
                return
            sid, rec, due = item
            try:
                self._issue(sid, rec, due)
            except Exception as exc:  # noqa: BLE001 -- a harness bug must not wedge the run
                with self._lock:
                    self.results.append({"status": "error", "error_code": f"replay:{exc}",
                                         "ttft_ms": None, "duration_ms": 0.0,
                                         "output_tokens": 0, "session_queue_ms": 0.0,
                                         "trace": {"t": rec.get("t"),
                                                   "prompt_tokens": rec.get("prompt_tokens")}})
            finally:
                with self._lock:
                    if sid:
                        held = self._pending.get(sid)
                        if held:
                            nxt = held.popleft()
                            self._ready.put((sid, nxt[0], nxt[1]))
                        else:
                            self._pending.pop(sid, None)
                            self._busy.discard(sid)
                    self._left -= 1
                    if self._left <= 0:
                        self._done.notify_all()

    # -- one request --------------------------------------------------------

    def _issue(self, sid: str | None, rec: dict[str, Any], due: float) -> None:
        held = max(0.0, time.time() - due)
        prompt = self.builder.build(rec)
        route, body = build_body(rec, prompt, self.args.model,
                                 stream=self.args.stream, max_tokens_cap=self.args.max_tokens)
        res = self.client.send(route, body, sid)
        res["session_queue_ms"] = held * 1e3
        res["trace"] = {k: rec.get(k) for k in
                        ("t", "prompt_tokens", "cached_tokens", "output_tokens",
                         "ttft_ms", "duration_ms", "status", "session")}
        res["sent_words"] = self.builder.total_words(rec)
        with self._lock:
            self.results.append(res)

    # -- the run ------------------------------------------------------------

    def run(self) -> None:
        workers = max(1, min(self.args.max_inflight, len(self.records)))
        threads = [threading.Thread(target=self._worker, daemon=True, name=f"replay-{i}")
                   for i in range(workers)]
        for th in threads:
            th.start()
        t0 = self.records[0]["t"]
        wall0 = time.time()
        speed = max(1e-6, self.args.speed)
        for rec in self.records:
            due = wall0 + (rec["t"] - t0) / speed
            delay = due - time.time()
            if delay > 0:
                time.sleep(delay)
            self._submit(rec, due)
        with self._lock:
            while self._left > 0:
                if not self._done.wait(timeout=self.args.timeout + 30):
                    break
        for _ in threads:
            self._ready.put(None)
        for th in threads:
            th.join(timeout=5.0)


# --------------------------------------------------------------------------- metrics


def pct(vals: list[float], p: float) -> float | None:
    if not vals:
        return None
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summarize(label: str, ttft: list[float], dur: list[float], out_tokens: list[int],
              prompt_tokens: list[int], cached: list[int], errors: int, n: int,
              span_s: float) -> dict[str, Any]:
    tot_out = sum(out_tokens)
    tot_prompt = sum(prompt_tokens)
    return {
        "label": label,
        "requests": n,
        "errors": errors,
        "error_rate": (errors / n) if n else 0.0,
        "ttft_ms_p50": pct(ttft, 0.50), "ttft_ms_p95": pct(ttft, 0.95),
        "ttft_ms_p99": pct(ttft, 0.99),
        "latency_ms_p50": pct(dur, 0.50), "latency_ms_p95": pct(dur, 0.95),
        "latency_ms_p99": pct(dur, 0.99),
        "output_tokens": tot_out,
        "prompt_tokens": tot_prompt,
        "cached_tokens": sum(cached),
        "cached_frac": (sum(cached) / tot_prompt) if tot_prompt else None,
        "output_tok_s": (tot_out / span_s) if span_s > 0 else None,
        "span_s": round(span_s, 3),
    }


def trace_side(records: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in records if r.get("status") == "ok"]
    span = (max(r["t"] + (r.get("duration_ms") or 0) / 1e3 for r in records)
            - min(r["t"] for r in records)) if records else 0.0
    return summarize(
        "trace",
        [r["ttft_ms"] for r in ok if r.get("ttft_ms") is not None],
        [r["duration_ms"] for r in ok if r.get("duration_ms") is not None],
        [r.get("output_tokens") or 0 for r in ok],
        [r.get("prompt_tokens") or 0 for r in ok],
        [r.get("cached_tokens") or 0 for r in ok],
        sum(1 for r in records if r.get("status") != "ok"),
        len(records), span,
    )


def replay_side(results: list[dict[str, Any]], span: float) -> dict[str, Any]:
    ok = [r for r in results if r["status"] == "ok"]
    s = summarize(
        "replay",
        [r["ttft_ms"] for r in ok if r.get("ttft_ms") is not None],
        [r["duration_ms"] for r in ok if r.get("duration_ms") is not None],
        [r.get("output_tokens") or 0 for r in ok],
        [r.get("prompt_tokens") or 0 for r in ok],
        [r.get("cached_tokens") or 0 for r in ok],
        sum(1 for r in results if r["status"] != "ok"),
        len(results), span,
    )
    held = [r.get("session_queue_ms") or 0.0 for r in results]
    s["session_queue_ms_p95"] = pct(held, 0.95)
    err = [abs(r["prompt_tokens"] - (r["trace"].get("prompt_tokens") or 0))
           / max(1, r["trace"].get("prompt_tokens") or 0)
           for r in ok if r.get("prompt_tokens")]
    s["prompt_tokens_err_p50"] = pct(err, 0.50)
    s["prompt_tokens_err_p95"] = pct(err, 0.95)
    return s


def fidelity(records: list[dict[str, Any]], builder: PromptBuilder,
             cal: Calibration) -> dict[str, Any]:
    """Offline reconstruction check: predicted prompt length, and how much of the traced
    prefix sharing the regenerated prompts actually reproduce."""
    err = []
    for r in records:
        want = r.get("prompt_tokens") or 0
        if want <= 0:
            continue
        err.append(abs(predict_tokens(r, builder, cal) - want) / want)
    # Prefix sharing the chain implies, as a fraction of the messages sent.
    seen: dict[str, int] = {}
    shared = total = 0
    for r in records:
        chain = r.get("msg_chain") or []
        chars = r.get("msg_chars") or []
        for i, h in enumerate(chain):
            n = builder.words_for(chars[i] if i < len(chars) else 0)
            total += n
            if h in seen:
                shared += n
                # A hash seen at a different length would mean the pure-length rule broke.
                assert seen[h] == n, f"chain {h} regenerated at {seen[h]} and {n} words"
            else:
                seen[h] = n
    return {
        "predicted_prompt_tokens_err_p50": pct(err, 0.50),
        "predicted_prompt_tokens_err_p95": pct(err, 0.95),
        "reconstructed_shared_word_frac": (shared / total) if total else 0.0,
        "traced_cached_frac": _traced_cached_frac(records),
        "scale": builder.scale,
    }


def _traced_cached_frac(records: list[dict[str, Any]]) -> float | None:
    p = sum(r.get("prompt_tokens") or 0 for r in records)
    c = sum(r.get("cached_tokens") or 0 for r in records)
    return (c / p) if p else None


# --------------------------------------------------------------------------- main


def calibrate(client: Client, model: str | None, small: int = 32,
              large: int = 512) -> Calibration:
    """Three probes solve the whole linear model against the server's real tokenizer.

    Two vary the word count at one message (slope + a lumped intercept); the third splits
    the same word count over three messages, which is the only way to separate the chat
    template's *per-turn* cost from its fixed preamble. Assuming one token per word instead
    would be wrong for every non-English or code-heavy trace, and lumping the per-turn cost
    into the intercept mis-sizes long conversations specifically.
    """
    book = FillerBook()

    def probe(chunks: list[int]) -> int:
        msgs = [{"role": "user" if i % 2 == 0 else "assistant",
                 "content": book.text(f"cal{i}-{n:012x}", n)}
                for i, n in enumerate(chunks)]
        res = client.send(CHAT_ROUTE, {"model": model or "default", "max_tokens": 1,
                                       "messages": msgs}, None)
        if res.get("prompt_tokens") is None:
            raise SystemExit(
                "calibration failed: the server returned no usage.prompt_tokens "
                f"({res.get('error_code') or res.get('status')}). Pass --tokens-per-word, "
                "--template-overhead and --per-message-overhead to skip calibration.")
        return int(res["prompt_tokens"])

    t1 = probe([small])
    t2 = probe([large])
    third = max(1, small // 3)
    t3 = probe([third, third, small - 2 * third])
    a = (t2 - t1) / max(1, (large - small))
    per_message = max(0.0, (t3 - t1) / 2.0)
    return Calibration(a, t1 - a * small - per_message, per_message)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trace", required=True,
                    help="trace .jsonl file, or a --trace-dir directory")
    ap.add_argument("--base-url", default="http://127.0.0.1:1919")
    ap.add_argument("--model", default=None,
                    help="override the model id (default: whatever the trace recorded)")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="time scale; 2.0 replays twice as fast (inter-arrival / 2)")
    ap.add_argument("--max-requests", type=int, default=0, help="0 = the whole trace")
    ap.add_argument("--max-tokens", type=int, default=0,
                    help="cap every request's max_tokens (0 = use the trace's)")
    ap.add_argument("--route", default="", choices=["", CHAT_ROUTE, COMPLETIONS_ROUTE],
                    help="replay only this route")
    ap.add_argument("--session-header", default="x-switchyard-session-id")
    ap.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--max-inflight", type=int, default=256,
                    help="worker pool width; caps concurrent requests (and threads) so a "
                         "long trace cannot fork-bomb the replaying host")
    ap.add_argument("--no-stream", dest="stream", action="store_false", default=True,
                    help="non-streaming (loses TTFT)")
    ap.add_argument("--tokens-per-word", type=float, default=None,
                    help="skip server calibration; assume this slope")
    ap.add_argument("--template-overhead", type=float, default=None,
                    help="skip server calibration; assume this fixed intercept")
    ap.add_argument("--per-message-overhead", type=float, default=None,
                    help="skip server calibration; assume this per-turn template cost")
    ap.add_argument("--dry-run", action="store_true",
                    help="build every prompt and report reconstruction fidelity; no server")
    ap.add_argument("--out", default="", help="write the JSON result here")
    a = ap.parse_args(argv)

    records = [r for r in RT.read_trace(a.trace)]
    if a.route:
        records = [r for r in records if (r.get("route") or CHAT_ROUTE) == a.route]
    if a.max_requests:
        records = records[: a.max_requests]
    if not records:
        print(f"no v{RT.TRACE_VERSION} records in {a.trace}", file=sys.stderr)
        return 2

    if a.tokens_per_word is not None or a.dry_run:
        # Defaults for an offline run: one token per filler word, and the token cost a
        # typical chat template charges per turn. They only set the SCALE the fit starts
        # from, so a dry run reports relative fidelity honestly even when they are wrong.
        cal = Calibration(
            a.tokens_per_word if a.tokens_per_word is not None else 1.0,
            a.template_overhead if a.template_overhead is not None else 3.0,
            a.per_message_overhead if a.per_message_overhead is not None else 7.0,
        )
        client = None
    else:
        client = Client(a.base_url, a.timeout, a.session_header, a.api_key)
        cal = calibrate(client, a.model)
    scale = fit_scale(records, cal)
    builder = PromptBuilder(scale)

    out: dict[str, Any] = {
        "trace": os.path.abspath(a.trace),
        "records": len(records),
        "sessions": len({r.get("session") for r in records if r.get("session")}),
        "speed": a.speed,
        "calibration": cal.as_dict(),
        "fidelity": fidelity(records, builder, cal),
        "original": trace_side(records),
    }
    if not a.dry_run:
        if client is None:
            client = Client(a.base_url, a.timeout, a.session_header, a.api_key)
        rep = Replayer(records, builder, client, a)
        t0 = time.time()
        rep.run()
        out["replay"] = replay_side(rep.results, time.time() - t0)
        out["per_request"] = [
            {k: v for k, v in r.items() if k != "trace"} | {"trace_t": r["trace"]["t"]}
            for r in rep.results
        ]

    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=2))
    _print(out)
    return 0


def _print(out: dict[str, Any]) -> None:
    f = out["fidelity"]
    print(f"trace {out['trace']}: {out['records']} requests, {out['sessions']} sessions")
    cal = out["calibration"]
    print(f"  scale={f['scale']:.4f}  tokens/word={cal['tokens_per_word']:.3f} "
          f"+{cal['overhead']:.1f} +{cal['per_message']:.1f}/message")
    print(f"  prompt-length error p50={_p(f['predicted_prompt_tokens_err_p50'])} "
          f"p95={_p(f['predicted_prompt_tokens_err_p95'])}")
    print(f"  reconstructed shared prefix={_p(f['reconstructed_shared_word_frac'])} "
          f"(traced cached_tokens fraction={_p(f['traced_cached_frac'])})")
    rows = [out["original"]] + ([out["replay"]] if "replay" in out else [])
    keys = ("requests", "errors", "ttft_ms_p50", "ttft_ms_p95", "ttft_ms_p99",
            "latency_ms_p50", "latency_ms_p95", "latency_ms_p99", "output_tok_s",
            "cached_frac")
    print(f"  {'metric':<18}" + "".join(f"{r['label']:>14}" for r in rows))
    for k in keys:
        print(f"  {k:<18}" + "".join(_cell(r.get(k)) for r in rows))


def _cell(v: Any) -> str:
    if v is None:
        return f"{'-':>14}"
    return f"{v:>14.3f}" if isinstance(v, float) else f"{v:>14}"


def _p(v: float | None) -> str:
    return "-" if v is None else f"{100 * v:.1f}%"


if __name__ == "__main__":
    raise SystemExit(main())
