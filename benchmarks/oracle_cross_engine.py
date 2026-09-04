"""Standing cross-engine oracle: the same prompts through FreeToken and llama.cpp.

Why this exists
---------------
The 2026-09-04 262K cross-engine run
(``benchmarks/results/nemotron35_lightning_5080_262k_crossengine_2026-09-04.md``)
found llama.cpp recalling the needle at every depth and length where FreeToken missed
it, which reopened a "model/quant limit" verdict that a single-engine bisect had
already closed. That comparison was hand-assembled from two scripts and a shell loop.
This is the same comparison as a standing tool: one suite, two engines, one report,
a verdict per question.

    agree                -- both engines answered correctly
    both-miss            -- neither did; a model/prompt limit, not an engine bug
    freetoken-only-miss  -- llama.cpp answered, FreeToken did not: reopen the engine bug
    llamacpp-only-miss   -- FreeToken answered, llama.cpp did not

One engine at a time
--------------------
At 262K and above neither engine leaves room for the other on a 16 GiB card, and the
host rule (``tasks/lessons.md``, ``scripts/gpu_lock.sh``) is one model-loading job at a
time. So recording and comparing are separate commands:

    record --engine freetoken   drives a server the caller already started
    record --engine llamacpp    starts and stops ``llama-server`` itself
    compare                     CPU only; merges two recordings into the report

Each recording carries the haystack's SHA-256 and its token count; ``compare`` refuses
to merge two recordings that were not asked the same thing.

What is compared
----------------
1. **Answers.** Every question is graded per engine, with the needle classification
   from ``bench_multi_needle`` (retention / selection / interference / incoherent), so
   a disagreement says *how* the losing engine failed and not just that it did.
2. **Top-k logprobs** at the first N generated positions, when both engines expose
   them. FreeToken's chat endpoint currently rejects ``top_logprobs > 0``
   (``python/freetoken/server/openai_api.py``: "logprobs are not supported") and the
   engine computes no logprobs at all -- ``SamplingParams`` in ``python/freetoken/
   core.py`` has no such field and nothing below the HTTP layer gathers them. That is
   a sampler feature, not a serialization gap, so this tool **probes** for the
   capability at runtime and records the gap in the report rather than pretending to
   measure it. The comparison code is live and will start producing numbers the day
   the engine grows the feature.
3. **Hidden states**, optionally: with ``--hidden-states-dir`` the FreeToken recording
   asks the Switchyard prefill probe to export turn 1's per-layer residuals and notes
   the artifact path, which ``benchmarks/probe_hidden_states_parity.py`` can then be
   pointed at. llama.cpp exports nothing comparable, so this is a FreeToken-vs-
   reference hook hanging off the same run, not a cross-engine tensor diff.

The confound to keep stating
----------------------------
FreeToken serves NVFP4 safetensors and llama.cpp serves a Q4_0 GGUF; there is no GGUF
loader for ``nemotron_h`` in FreeToken and no NVFP4 path in llama.cpp, so on this host
engine and quantization necessarily move together. A ``freetoken-only-miss`` is
therefore "engine or NVFP4", and the report says so every time.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bench_multi_needle as mn  # noqa: E402
from bench_long_context import load_tokenizer  # noqa: E402


DEFAULT_LLAMA_BIN = os.path.expanduser("~/ai/llama.cpp/build/bin/llama-server")
DEFAULT_GGUF = os.path.expanduser(
    "~/ai/models/nemotron35-gguf/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf"
)
DEFAULT_MODEL_DIR = os.path.expanduser(
    "~/ai/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
)

#: Short, deterministic, haystack-free prompts. They cost almost nothing and they
#: separate "this engine cannot retrieve at depth" from "this engine is producing
#: different text than the other one for any prompt at all".
#: ``expect_all`` fragments must all appear; ``expect_any`` needs one of them. Keeping
#: the two apart matters: "list the primes" wants every number, "return a + b" wants
#: either spacing, and one list doing both jobs silently fails the second kind.
GENERIC_PROMPTS: list[dict] = [
    {"id": "generic:arith", "expect_all": ["391"],
     "text": "What is 17 times 23? Reply with the number only."},
    {"id": "generic:capital", "expect_all": ["canberra"],
     "text": "What is the capital city of Australia? Reply with the city name only."},
    {"id": "generic:primes", "expect_all": ["2", "3", "5", "7", "11"],
     "text": "List the first five prime numbers in increasing order, separated by "
             "commas. Numbers only, no other words."},
    {"id": "generic:python", "expect_any": ["a + b", "a+b"],
     "text": "In Python, the body of `def add(a, b):` should return the sum of its two "
             "arguments. Reply with only the expression that follows `return`."},
    {"id": "generic:letters", "expect_all": ["9"],
     "text": "How many letters are in the word 'benchmark'? Reply with the number "
             "only."},
]

ENGINE_FREETOKEN = "freetoken"
ENGINE_LLAMACPP = "llama.cpp"


# ------------------------------------------------------------------ llama-server


class LlamaServer:
    """Start ``llama-server``, wait for ``/health``, and always tear it down.

    Flags default to the ones the 262K cross-engine run used, because that run is the
    reference the oracle is meant to reproduce: ``--n-cpu-moe`` keeps enough routed
    experts on host RAM for a Q4_0 nemotron_h to fit a 16 GiB card, ``-ctk/-ctv q8_0``
    matches FreeToken's quantized KV, and ``--no-context-shift`` makes an
    over-long prompt an error instead of a silently truncated one.
    """

    def __init__(self, *, binary: str, gguf: str, port: int, ctx: int,
                 n_cpu_moe: int, threads: int, log_path: str,
                 extra_args: list[str] | None = None,
                 startup_timeout: float = 1800.0) -> None:
        self.binary = binary
        self.gguf = gguf
        self.port = port
        self.ctx = ctx
        self.n_cpu_moe = n_cpu_moe
        self.threads = threads
        self.log_path = log_path
        self.extra_args = list(extra_args or [])
        self.startup_timeout = startup_timeout
        self.process: subprocess.Popen | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def command(self) -> list[str]:
        return [
            self.binary, "-m", self.gguf,
            "--host", "127.0.0.1", "--port", str(self.port),
            "-c", str(self.ctx), "-np", "1",
            "--no-context-shift", "--cache-ram", "0",
            "-ngl", "999", "--n-cpu-moe", str(self.n_cpu_moe),
            "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0",
            "-b", "4096", "-ub", "512", "-t", str(self.threads),
            "--jinja", "--no-warmup",
        ] + self.extra_args

    def __enter__(self) -> LlamaServer:
        if not os.path.exists(self.binary):
            raise SystemExit(f"llama-server not found at {self.binary}")
        if not os.path.exists(self.gguf):
            raise SystemExit(f"GGUF not found at {self.gguf}")
        command = self.command()
        print(f"[oracle] starting llama-server: {' '.join(command)}", flush=True)
        self.log = open(self.log_path, "w")
        self.process = subprocess.Popen(
            command, stdout=self.log, stderr=subprocess.STDOUT,
            start_new_session=False,
        )
        deadline = time.time() + self.startup_timeout
        while time.time() < deadline:
            if self.process.poll() is not None:
                raise SystemExit(
                    f"llama-server exited with {self.process.returncode} during "
                    f"startup; see {self.log_path}"
                )
            try:
                with urllib.request.urlopen(f"{self.base_url}/health", timeout=5) as r:
                    if r.status == 200:
                        print(f"[oracle] llama-server ready on port {self.port}",
                              flush=True)
                        return self
            except (urllib.error.URLError, urllib.error.HTTPError, OSError):
                pass
            time.sleep(2.0)
        self.__exit__(None, None, None)
        raise SystemExit(
            f"llama-server did not become healthy in {self.startup_timeout}s; "
            f"see {self.log_path}"
        )

    def __exit__(self, *_exc) -> None:
        process, self.process = self.process, None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=60)
        log = getattr(self, "log", None)
        if log is not None:
            log.close()

    def build_info(self) -> dict:
        try:
            out = subprocess.run([self.binary, "--version"], capture_output=True,
                                 text=True, timeout=60)
            version = (out.stdout + out.stderr).strip().splitlines()
        except (OSError, subprocess.SubprocessError):
            version = []
        return {"binary": self.binary, "gguf": self.gguf,
                "gguf_bytes": os.path.getsize(self.gguf)
                if os.path.exists(self.gguf) else None,
                "version": version[:2], "command": self.command()}


# ------------------------------------------------------------------ capabilities


def served_model_id(origin: str, timeout: float = 60.0) -> str:
    with urllib.request.urlopen(f"{origin}/v1/models", timeout=timeout) as response:
        return json.load(response)["data"][0]["id"]


def probe_logprobs(origin: str, model_id: str, timeout: float = 120.0) -> dict:
    """Does this server return per-token top-k logprobs on /v1/chat/completions?

    A one-token non-streamed request is the cheapest possible question. A 4xx means
    the endpoint rejects the parameter (FreeToken does, deliberately); a 200 whose
    response carries no ``logprobs.content`` means it accepted and ignored it, which
    is just as unusable and is reported separately.
    """
    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
        "max_completion_tokens": 1, "temperature": 0.0, "stream": False,
        "logprobs": True, "top_logprobs": 2,
    }
    request = urllib.request.Request(
        f"{origin}/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read()[:400].decode(errors="replace")
        return {"supported": False,
                "reason": f"HTTP {error.code} rejecting top_logprobs: {detail}"}
    except (urllib.error.URLError, OSError) as error:
        return {"supported": False, "reason": f"probe request failed: {error}"}
    entries = ((payload.get("choices") or [{}])[0].get("logprobs") or {}).get("content")
    if not entries:
        return {"supported": False,
                "reason": "accepted top_logprobs but returned no logprobs.content"}
    return {"supported": True, "reason": "logprobs.content present",
            "sample": entries[:1]}


# -------------------------------------------------------------------- recording


def grade_generic(item: dict, answer: str) -> dict:
    lowered = answer.lower()
    required = item.get("expect_all") or []
    alternatives = item.get("expect_any") or []
    missing = [token for token in required if token.lower() not in lowered]
    chosen = [token for token in alternatives if token.lower() in lowered]
    passed = not missing and (not alternatives or bool(chosen))
    return {
        "pass": passed,
        "matched": [t for t in required if t not in missing] + chosen,
        "detail": (f"{len(required) - len(missing)}/{len(required)} required"
                   + (f", {len(chosen)}/{len(alternatives)} alternatives" if alternatives
                      else "")),
    }


def generic_expectation(item: dict) -> str:
    parts = list(item.get("expect_all") or [])
    if item.get("expect_any"):
        parts.append("(" + " | ".join(item["expect_any"]) + ")")
    return " ".join(parts)


def run_generic(origin: str, model_id: str, *, decode: int, timeout: float,
                top_logprobs: int, session_prefix: str) -> list[dict]:
    rows = []
    for item in GENERIC_PROMPTS:
        # One shared session for all of them: each extra session id costs a linear-state
        # slot, and at 1M an idle 1.04M lease gets spilled to make room for it.
        row = mn.chat(origin, model_id, f"{session_prefix}-generic",
                      [{"role": "user", "content": item["text"]}], decode, timeout,
                      top_logprobs=top_logprobs)
        if "error" in row:
            rows.append({"question_id": item["id"], "shape": "generic",
                         "error": row["error"]})
            continue
        verdict = grade_generic(item, row["text"])
        row.update({"question_id": item["id"], "shape": "generic", "kind": "generic",
                    "owner": None, "key": item["id"], "expect": generic_expectation(item),
                    "leak_free": True,
                    **{f"verdict_{k}": v for k, v in verdict.items()}})
        rows.append(row)
        print(f"[oracle] {item['id']:18s} {'PASS' if verdict['pass'] else 'FAIL'} "
              f"{row['text'].strip()[:120]!r}", flush=True)
    return rows


def record(args: argparse.Namespace) -> int:
    tok = load_tokenizer(args.model_dir)
    t0 = time.perf_counter()
    haystack, placed = mn.build_haystack(tok, args.target_prompt_tokens,
                                         args.filler_cursor)
    haystack_tokens = len(tok.encode(haystack, add_special_tokens=False))
    digest = mn.haystack_digest(haystack)
    print(f"[oracle] haystack {haystack_tokens} tokens sha256={digest[:16]} "
          f"built in {time.perf_counter() - t0:.1f}s", flush=True)
    for entry in placed:
        print(f"[oracle]   {entry['key']:8s} {entry['kind']:8s} {entry['code']} "
              f"depth {entry['actual_depth']:.4f} ({entry['role']})", flush=True)
    if args.build_only:
        return 0

    items = mn.questions()
    server = None
    build: dict = {}
    if args.engine == ENGINE_LLAMACPP:
        ctx = args.llama_ctx or (args.target_prompt_tokens + args.llama_ctx_headroom)
        server = LlamaServer(
            binary=args.llama_bin, gguf=args.gguf, port=args.llama_port, ctx=ctx,
            n_cpu_moe=args.n_cpu_moe, threads=args.llama_threads,
            log_path=args.llama_log or f"llama-server-{args.target_prompt_tokens}.log",
            extra_args=args.llama_arg, startup_timeout=args.llama_startup_timeout,
        )
        build = server.build_info()

    context = server if server is not None else _NullContext()
    with context:
        origin = (server.base_url if server is not None
                  else args.base_url.rstrip("/"))
        model_id = served_model_id(origin)
        capability = probe_logprobs(origin, model_id)
        print(f"[oracle] engine={args.engine} model={model_id} "
              f"logprobs={'yes' if capability['supported'] else 'no'} "
              f"({capability['reason'][:120]})", flush=True)
        top_logprobs = args.top_logprobs if capability["supported"] else 0

        first_turn_extra = None
        if args.engine == ENGINE_FREETOKEN and args.hidden_states_dir:
            first_turn_extra = {"kv_transfer_params": {
                "hidden_states_path": args.hidden_states_dir,
                "include_output_tokens": False,
            }}

        def report(item: dict, row: dict) -> None:
            if "error" in row:
                print(f"[oracle] turn {row['turn']} {item['id']}: ERROR {row['error']}",
                      flush=True)
                return
            print(f"[oracle] turn {row['turn']:2d} {item['id']:26s} "
                  f"{'PASS' if row['verdict_pass'] else 'FAIL'} "
                  f"{'leak-free' if row['leak_free'] else 'LEAKED  '} "
                  f"cached={row['cached_tokens']}/{row['prompt_tokens']} "
                  f"[{row['verdict_detail']}]", flush=True)
            print(f"[oracle]      {row['text'].strip()[:240]!r}", flush=True)

        rows, _ = mn.run_suite(
            origin, model_id, haystack, items, session_id=args.session_id,
            decode=args.decode, timeout=args.timeout, top_logprobs=top_logprobs,
            first_turn_extra=first_turn_extra, on_row=report,
        )
        generic = []
        if not args.no_generic and not any("error" in row for row in rows):
            generic = run_generic(origin, model_id, decode=args.generic_decode,
                                  timeout=args.timeout, top_logprobs=top_logprobs,
                                  session_prefix=args.session_id)

    recording = {
        "engine": args.engine,
        "label": args.label,
        "generated": _dt.datetime.now().isoformat(timespec="seconds"),
        "model_id": model_id,
        "model_dir": args.model_dir,
        "build": build,
        "suite": {
            "target_prompt_tokens": args.target_prompt_tokens,
            "haystack_tokens": haystack_tokens,
            "haystack_sha256": digest,
            "filler_cursor": args.filler_cursor,
            "decode": args.decode,
            "needles": placed,
            "question_ids": [item["id"] for item in items],
        },
        "logprobs": capability,
        "hidden_states": ({"dir": args.hidden_states_dir}
                          if args.hidden_states_dir else None),
        "rows": rows,
        "generic": generic,
        "classification": mn.classify_all(rows),
    }
    with open(args.out, "w") as handle:
        json.dump(recording, handle, indent=1)
    print(f"[oracle] wrote {args.out}", flush=True)
    for entry in recording["classification"]:
        print(f"[oracle]   {entry['key']:8s} {entry['class']:18s} "
              f"in_state={str(entry['in_state']):5s} {entry['evidence']}", flush=True)
    failed = [r for r in rows + generic if "error" in r]
    return 1 if failed else 0


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


# ------------------------------------------------------------------- comparison


VERDICTS = {
    (True, True): "agree",
    (False, False): "both-miss",
    (False, True): "freetoken-only-miss",
    (True, False): "llamacpp-only-miss",
}

VERDICT_MEANING = {
    "agree": "both engines answered correctly",
    "both-miss": "neither engine answered: a model/prompt limit, not an engine bug",
    "freetoken-only-miss": "llama.cpp answered and FreeToken did not -- reopen the "
                           "engine bug (confounded with NVFP4 vs Q4_0)",
    "llamacpp-only-miss": "FreeToken answered and llama.cpp did not",
    "missing": "one recording has no row for this question",
}


def index_rows(recording: dict) -> dict[str, dict]:
    rows = list(recording.get("rows") or []) + list(recording.get("generic") or [])
    return {row["question_id"]: row for row in rows if "question_id" in row}


def compare_logprobs(left: list[dict], right: list[dict], positions: int) -> dict:
    """Top-1 agreement and logprob drift over the first ``positions`` generated tokens.

    Compared by decoded token *string*: the two engines carry the same vocabulary but
    reach it through different tokenizer implementations, so token ids are not a safe
    join key while the surface strings are.
    """
    pairs = list(zip(left[:positions], right[:positions]))
    if not pairs:
        return {"positions": 0, "top1_agree": 0, "top1_agree_rate": None,
                "mean_abs_logprob_delta": None, "first_divergence": None}
    agree = 0
    deltas: list[float] = []
    first_divergence = None
    for index, (a, b) in enumerate(pairs):
        if a.get("token") == b.get("token"):
            agree += 1
            if a.get("logprob") is not None and b.get("logprob") is not None:
                deltas.append(abs(float(a["logprob"]) - float(b["logprob"])))
        elif first_divergence is None:
            first_divergence = {"position": index, "freetoken": a.get("token"),
                                "llamacpp": b.get("token")}
    return {
        "positions": len(pairs),
        "top1_agree": agree,
        "top1_agree_rate": agree / len(pairs),
        "mean_abs_logprob_delta": statistics.fmean(deltas) if deltas else None,
        "first_divergence": first_divergence,
    }


def compare(freetoken: dict, llamacpp: dict, *, logprob_positions: int) -> dict:
    ft_suite = freetoken.get("suite") or {}
    lc_suite = llamacpp.get("suite") or {}
    mismatch = []
    for field in ("haystack_sha256", "target_prompt_tokens", "filler_cursor"):
        if ft_suite.get(field) != lc_suite.get(field):
            mismatch.append(
                f"{field}: freetoken={ft_suite.get(field)!r} "
                f"llama.cpp={lc_suite.get(field)!r}"
            )

    ft_rows = index_rows(freetoken)
    lc_rows = index_rows(llamacpp)
    order = [qid for qid in (ft_suite.get("question_ids") or []) if qid in ft_rows]
    order += [qid for qid in ft_rows if qid not in order]
    order += [qid for qid in lc_rows if qid not in order]

    both_logprobs = bool((freetoken.get("logprobs") or {}).get("supported")
                         and (llamacpp.get("logprobs") or {}).get("supported"))

    comparisons = []
    matrix = {key: 0 for key in VERDICTS.values()}
    matrix["missing"] = 0
    for qid in order:
        a, b = ft_rows.get(qid), lc_rows.get(qid)
        if a is None or b is None or "error" in a or "error" in b:
            verdict = "missing"
            entry = {"question_id": qid, "verdict": verdict,
                     "freetoken_pass": None if a is None else a.get("verdict_pass"),
                     "llamacpp_pass": None if b is None else b.get("verdict_pass"),
                     "note": "no row" if a is None or b is None else "engine error"}
        else:
            ft_pass = bool(a.get("verdict_pass"))
            lc_pass = bool(b.get("verdict_pass"))
            verdict = VERDICTS[(ft_pass, lc_pass)]
            entry = {
                "question_id": qid,
                "shape": a.get("shape") or b.get("shape"),
                "owner": a.get("owner"),
                "depth": a.get("depth"),
                "expect": a.get("expect"),
                "leak_free": bool(a.get("leak_free", True))
                and bool(b.get("leak_free", True)),
                "freetoken_pass": ft_pass,
                "llamacpp_pass": lc_pass,
                "freetoken_detail": a.get("verdict_detail"),
                "llamacpp_detail": b.get("verdict_detail"),
                "freetoken_answer": (a.get("text") or "").strip(),
                "llamacpp_answer": (b.get("text") or "").strip(),
                "answers_identical": (a.get("text") or "").strip()
                == (b.get("text") or "").strip(),
                "verdict": verdict,
            }
            if both_logprobs:
                entry["logprobs"] = compare_logprobs(
                    a.get("logprobs") or [], b.get("logprobs") or [],
                    logprob_positions,
                )
        matrix[verdict] += 1
        comparisons.append(entry)

    ft_class = {c["key"]: c for c in freetoken.get("classification") or []}
    lc_class = {c["key"]: c for c in llamacpp.get("classification") or []}
    needle_rows = []
    for key in list(ft_class) + [k for k in lc_class if k not in ft_class]:
        a, b = ft_class.get(key, {}), lc_class.get(key, {})
        needle_rows.append({
            "key": key,
            "depth": next((n["actual_depth"] for n in ft_suite.get("needles") or []
                           if n["key"] == key and n["role"] == "needle"), None),
            "freetoken_class": a.get("class"), "freetoken_in_state": a.get("in_state"),
            "freetoken_evidence": a.get("evidence"),
            "llamacpp_class": b.get("class"), "llamacpp_in_state": b.get("in_state"),
            "llamacpp_evidence": b.get("evidence"),
            "agree": a.get("class") == b.get("class"),
        })

    return {
        "generated": _dt.datetime.now().isoformat(timespec="seconds"),
        "prompt_mismatch": mismatch,
        "suite": {"freetoken": ft_suite, "llamacpp": lc_suite},
        "engines": {
            "freetoken": {k: freetoken.get(k) for k in
                          ("engine", "label", "model_id", "model_dir", "generated",
                           "logprobs", "hidden_states")},
            "llamacpp": {k: llamacpp.get(k) for k in
                         ("engine", "label", "model_id", "build", "generated",
                          "logprobs")},
        },
        "logprobs_compared": both_logprobs,
        "matrix": matrix,
        "comparisons": comparisons,
        "needles": needle_rows,
    }


# ---------------------------------------------------------------------- report


def _cell(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    return _escape(str(value))


def _yes_no(value) -> str:
    return "-" if value is None else ("yes" if value else "no")


def _escape(text: str, limit: int = 160) -> str:
    flat = " ".join((text or "").split())
    if len(flat) > limit:
        flat = flat[: limit - 1] + "…"
    return flat.replace("|", "\\|")


def render_markdown(result: dict) -> str:
    ft = result["engines"]["freetoken"]
    lc = result["engines"]["llamacpp"]
    suite = result["suite"]["freetoken"]
    out: list[str] = []
    add = out.append

    add("# Cross-engine oracle -- FreeToken vs llama.cpp")
    add("")
    add(f"Generated {result['generated']}. "
        f"Prompt: {suite.get('haystack_tokens')} tokens "
        f"(target {suite.get('target_prompt_tokens')}), "
        f"sha256 `{str(suite.get('haystack_sha256'))[:16]}`, "
        f"filler cursor {suite.get('filler_cursor')}.")
    add("")
    if result["prompt_mismatch"]:
        add("> **The two recordings were not asked the same thing.** "
            "Every verdict below is void until this is fixed:")
        add(">")
        for line in result["prompt_mismatch"]:
            add(f"> - {line}")
        add("")

    add("| | FreeToken | llama.cpp |")
    add("|---|---|---|")
    add(f"| model | `{ft.get('model_id')}` | `{lc.get('model_id')}` |")
    add(f"| weights | `{ft.get('model_dir')}` | "
        f"`{(lc.get('build') or {}).get('gguf')}` |")
    add(f"| recorded | {ft.get('generated')} | {lc.get('generated')} |")
    add(f"| label | {ft.get('label') or '-'} | {lc.get('label') or '-'} |")
    add(f"| top_logprobs | {_logprob_cell(ft)} | {_logprob_cell(lc)} |")
    add("")
    add("FreeToken serves NVFP4 safetensors and llama.cpp serves a Q4_0 GGUF; there is "
        "no GGUF loader for `nemotron_h` in FreeToken and no NVFP4 path in llama.cpp, "
        "so **engine and quantization move together** on this host. Read every "
        "`freetoken-only-miss` as \"engine *or* NVFP4\".")
    add("")

    add("## Verdicts")
    add("")
    add("| verdict | count | meaning |")
    add("|---|---:|---|")
    for verdict, count in result["matrix"].items():
        if count or verdict != "missing":
            add(f"| `{verdict}` | {count} | {VERDICT_MEANING[verdict]} |")
    add("")
    add("Agreement matrix (rows FreeToken, columns llama.cpp):")
    add("")
    add("| | llama.cpp PASS | llama.cpp FAIL |")
    add("|---|---:|---:|")
    add(f"| **FreeToken PASS** | {result['matrix']['agree']} | "
        f"{result['matrix']['llamacpp-only-miss']} |")
    add(f"| **FreeToken FAIL** | {result['matrix']['freetoken-only-miss']} | "
        f"{result['matrix']['both-miss']} |")
    add("")

    add("## Per question")
    add("")
    add("| question | shape | depth | expected | FreeToken | llama.cpp | verdict | "
        "leak-free | same text |")
    add("|---|---|---|---|---|---|---|---|---|")
    for entry in result["comparisons"]:
        depth = entry.get("depth")
        add(
            f"| `{entry['question_id']}` | {entry.get('shape') or '-'} "
            f"| {f'{depth:.3f}' if isinstance(depth, float) else '-'} "
            f"| {_escape(str(entry.get('expect') or '-'))} "
            f"| {_cell(entry.get('freetoken_pass'))} "
            f"| {_cell(entry.get('llamacpp_pass'))} "
            f"| `{entry['verdict']}` "
            f"| {'yes' if entry.get('leak_free') else 'no'} "
            f"| {'yes' if entry.get('answers_identical') else 'no'} |"
        )
    add("")

    add("## Needle classification")
    add("")
    add("A miss is only useful if it says *how* it missed. `retention` = nothing "
        "recovered the code; `selection` = a leak-free combined/reverse probe did but "
        "the direct question did not; `interference-near` = the same key's "
        f"`{mn.DISTRACTOR_KIND}` twin came back instead; `interference-cross` = another "
        "key's answer did; `incoherent` = the direct answer carried neither a code nor "
        "a denial.")
    add("")
    add("| needle | depth | FreeToken | in state | llama.cpp | in state | agree |")
    add("|---|---|---|---|---|---|---|")
    for entry in result["needles"]:
        depth = entry.get("depth")
        add(
            f"| {entry['key']} "
            f"| {f'{depth:.4f}' if isinstance(depth, float) else '-'} "
            f"| `{entry.get('freetoken_class')}` "
            f"| {_yes_no(entry.get('freetoken_in_state'))} "
            f"| `{entry.get('llamacpp_class')}` "
            f"| {_yes_no(entry.get('llamacpp_in_state'))} "
            f"| {'yes' if entry.get('agree') else 'no'} |"
        )
    add("")
    for entry in result["needles"]:
        if not entry.get("agree"):
            add(f"- **{entry['key']}** -- FreeToken: {entry.get('freetoken_evidence')}; "
                f"llama.cpp: {entry.get('llamacpp_evidence')}")
    add("")

    add("## Logprobs")
    add("")
    if result["logprobs_compared"]:
        add("| question | positions | top-1 agree | mean abs delta | first divergence |")
        add("|---|---:|---:|---:|---|")
        for entry in result["comparisons"]:
            lp = entry.get("logprobs")
            if not lp or not lp["positions"]:
                continue
            delta = lp["mean_abs_logprob_delta"]
            delta_cell = "-" if delta is None else f"{delta:.4f}"
            divergence = lp["first_divergence"]
            divergence_cell = (
                "-" if divergence is None
                else f"pos {divergence['position']}: "
                     f"{_escape(str(divergence['freetoken']), 24)} vs "
                     f"{_escape(str(divergence['llamacpp']), 24)}"
            )
            add(
                f"| `{entry['question_id']}` | {lp['positions']} "
                f"| {lp['top1_agree']} ({lp['top1_agree_rate']:.2f}) "
                f"| {delta_cell} | {divergence_cell} |"
            )
    else:
        add("Not compared. " + " ".join(
            f"**{name}**: {(engine.get('logprobs') or {}).get('reason', 'unknown')}."
            for name, engine in (("FreeToken", ft), ("llama.cpp", lc))
            if not (engine.get("logprobs") or {}).get("supported")
        ))
        add("")
        add("FreeToken's chat endpoint rejects `top_logprobs > 0` "
            "(`python/freetoken/server/openai_api.py`) and nothing below the HTTP "
            "layer computes logprobs: `SamplingParams` in `python/freetoken/core.py` "
            "has no `logprobs`/`top_logprobs`/`return_logprob` field and the sampler "
            "never gathers per-token or top-k probabilities. Exposing them is a "
            "sampler feature, not a serialization fix, so this oracle compares at the "
            "answer level and records the gap here. The comparison code path is live "
            "and starts producing this table the day the engine grows the feature.")
    add("")

    hidden = ft.get("hidden_states")
    if hidden:
        add("## Hidden states")
        add("")
        add(f"FreeToken's turn-1 prefill probe exported to `{hidden.get('dir')}`. "
            "Point `benchmarks/probe_hidden_states_parity.py --artifact <path>` at it "
            "for the per-layer comparison against transformers. llama.cpp exports "
            "nothing comparable, so this is a FreeToken-vs-reference hook on the same "
            "run, not a cross-engine tensor diff.")
        add("")

    add("## Answers")
    add("")
    for entry in result["comparisons"]:
        if entry["verdict"] == "agree" and entry.get("answers_identical"):
            continue
        add(f"- `{entry['question_id']}` ({entry['verdict']})")
        add(f"  - FreeToken: {_escape(entry.get('freetoken_answer') or '')}")
        add(f"  - llama.cpp: {_escape(entry.get('llamacpp_answer') or '')}")
    add("")
    return "\n".join(out)


def _logprob_cell(engine: dict) -> str:
    capability = engine.get("logprobs") or {}
    if capability.get("supported"):
        return "supported"
    return f"unsupported -- {_escape(capability.get('reason', 'not probed'), 80)}"


def compare_main(args: argparse.Namespace) -> int:
    with open(args.freetoken) as handle:
        freetoken = json.load(handle)
    with open(args.llamacpp) as handle:
        llamacpp = json.load(handle)
    result = compare(freetoken, llamacpp, logprob_positions=args.logprob_positions)
    if args.json_out:
        with open(args.json_out, "w") as handle:
            json.dump(result, handle, indent=1)
        print(f"[oracle] wrote {args.json_out}", flush=True)
    markdown = render_markdown(result)
    if args.markdown:
        with open(args.markdown, "w") as handle:
            handle.write(markdown)
        print(f"[oracle] wrote {args.markdown}", flush=True)
    else:
        print(markdown)
    if result["prompt_mismatch"]:
        print("[oracle] REFUSING the verdicts: the recordings differ:", flush=True)
        for line in result["prompt_mismatch"]:
            print(f"[oracle]   {line}", flush=True)
        return 3
    print(f"[oracle] {json.dumps(result['matrix'])}", flush=True)
    return 2 if result["matrix"]["freetoken-only-miss"] else 0


# ------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    rec = sub.add_parser("record", help="drive one engine and write a recording")
    rec.add_argument("--engine", choices=(ENGINE_FREETOKEN, ENGINE_LLAMACPP),
                     required=True)
    rec.add_argument("--out", required=True, help="recording JSON to write")
    rec.add_argument("--label", default="")
    rec.add_argument("--base-url", default="http://127.0.0.1:8123",
                     help="FreeToken server the caller already started")
    rec.add_argument("--model-dir", default=DEFAULT_MODEL_DIR,
                     help="HF checkpoint dir; the tokenizer that sizes the haystack")
    rec.add_argument("--target-prompt-tokens", type=int, default=262_144)
    rec.add_argument("--filler-cursor", type=int, default=0)
    rec.add_argument("--decode", type=int, default=128)
    rec.add_argument("--generic-decode", type=int, default=48)
    rec.add_argument("--session-id", default="oracle")
    rec.add_argument("--timeout", type=float, default=7200.0)
    rec.add_argument("--top-logprobs", type=int, default=5,
                     help="requested when the engine supports it; 0 disables")
    rec.add_argument("--hidden-states-dir",
                     help="FreeToken only: the server's --hidden-states-dir, so turn 1 "
                          "also exports the Switchyard prefill probe")
    rec.add_argument("--no-generic", action="store_true",
                     help="skip the short haystack-free prompts")
    rec.add_argument("--build-only", action="store_true",
                     help="build and hash the prompt, contact no server")
    rec.add_argument("--llama-bin", default=DEFAULT_LLAMA_BIN)
    rec.add_argument("--gguf", default=DEFAULT_GGUF)
    rec.add_argument("--llama-port", type=int, default=8080)
    rec.add_argument("--llama-ctx", type=int,
                     help="default: --target-prompt-tokens + --llama-ctx-headroom")
    rec.add_argument("--llama-ctx-headroom", type=int, default=8192)
    rec.add_argument("--n-cpu-moe", type=int, default=14,
                     help="routed-expert blocks kept on host RAM (16 GiB card: 14)")
    rec.add_argument("--llama-threads", type=int, default=16)
    rec.add_argument("--llama-log")
    rec.add_argument("--llama-startup-timeout", type=float, default=1800.0)
    rec.add_argument("--llama-arg", action="append", default=[],
                     help="extra llama-server argument, repeatable")

    cmp_ = sub.add_parser("compare", help="merge two recordings into the report")
    cmp_.add_argument("--freetoken", required=True)
    cmp_.add_argument("--llamacpp", required=True)
    cmp_.add_argument("--markdown")
    cmp_.add_argument("--json", dest="json_out")
    cmp_.add_argument("--logprob-positions", type=int, default=16)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "record":
        return record(args)
    return compare_main(args)


if __name__ == "__main__":
    sys.exit(main())
