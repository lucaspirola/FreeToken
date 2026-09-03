"""End-to-end serving gates for Nemotron-H / Nemotron-3.5 Lightning.

Unlike the other benchmark harnesses in this directory, this script never starts a
server: it drives an already-running ``ft serve`` over the OpenAI API so a single
launch profile can be gated by several subcommands in a row.

    ft serve --model ~/ai/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 ... &
    uv run benchmarks/gate_nemotron_h_serving.py --port 8000 \
        --result-md benchmarks/results/nemotron35_lightning_5080_2026-09-04.md all

Shared options precede the subcommand; per-gate options follow it.

Gates:
  batch-invariance  8 fixed prompts greedy alone, then 16 concurrent copies of them;
                    every concurrent answer must reproduce its solo answer exactly.
  prefix-cache      a request that shares a 4K prefix with an earlier one must return
                    the same text as its cold run and report cached prompt tokens.
  elastic-ramp      concurrency 1 -> 6 -> 16 -> 1 with no errors and no truncation.
  tool-call         one OpenAI tools request must finish with ``tool_calls`` and
                    parseable JSON arguments.

Streamed answers are reassembled from every SSE event before anything inspects them:
a single token can be split across ``data:`` events, so grepping the raw stream
produces false failures (tasks/lessons.md, 2026-08-26).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Iterable, Iterator
from typing import Any

# Eight deterministic, short-answer prompts. Greedy decoding makes each one a fixed
# string, so batch invariance is a plain equality check on the reassembled answer.
BATCH_PROMPTS: tuple[str, ...] = (
    "What is 17 multiplied by 23? Answer with the number only.",
    "Name the capital city of Australia. Answer with the city name only.",
    "List the first five prime numbers, separated by commas.",
    "How many bytes are in one kibibyte? Answer with the number only.",
    "Spell the word 'parallel' backwards. Answer with the letters only.",
    "What is the chemical symbol for tungsten? Answer with the symbol only.",
    "Which planet is closest to the Sun? Answer with the planet name only.",
    "What is 2 to the power of 10? Answer with the number only.",
)

WEATHER_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. Boston"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


# --------------------------------------------------------------------------- args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--base-url",
        help="full origin; overrides --host/--port (e.g. http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--model",
        help="served model id; taken from /v1/models when omitted",
    )
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="leave the chat template's reasoning block enabled (slow, still greedy)",
    )
    parser.add_argument(
        "--no-stream",
        dest="stream",
        action="store_false",
        help="use non-streaming responses instead of SSE",
    )
    parser.add_argument("--json", dest="json_out", help="append the report as a JSON line")
    parser.add_argument(
        "--result-md",
        nargs="?",
        const="",
        help="append a markdown section; bare flag writes benchmarks/results/"
        "nemotron35_lightning_5080_<date>.md",
    )
    parser.set_defaults(stream=True)

    subs = parser.add_subparsers(dest="gate", required=True)

    invariance = subs.add_parser(
        "batch-invariance", help="solo greedy answers must survive 16-way concurrency"
    )
    invariance.add_argument("--concurrency", type=int, default=16)

    cache = subs.add_parser("prefix-cache", help="shared-prefix reuse must not change output")
    cache.add_argument("--prefix-tokens", type=int, default=4096)
    cache.add_argument(
        "--enable-cache-report",
        action="store_true",
        help="the server was started with --enable-cache-report; require cached_tokens",
    )
    cache.add_argument("--min-cached-fraction", type=float, default=0.5)

    ramp = subs.add_parser("elastic-ramp", help="1 -> 6 -> 16 -> 1 concurrency without errors")
    ramp.add_argument(
        "--stages",
        default="1,6,16,1",
        help="comma-separated concurrency stages (default 1,6,16,1)",
    )

    tool = subs.add_parser("tool-call", help="one tools request must finish as tool_calls")
    tool.add_argument("--tool-name", default="get_current_weather")

    every = subs.add_parser("all", help="run every gate in order")
    every.add_argument("--concurrency", type=int, default=16)
    every.add_argument("--prefix-tokens", type=int, default=4096)
    every.add_argument("--enable-cache-report", action="store_true")
    every.add_argument("--min-cached-fraction", type=float, default=0.5)
    every.add_argument("--stages", default="1,6,16,1")
    every.add_argument("--tool-name", default="get_current_weather")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def origin_of(args: argparse.Namespace) -> str:
    if getattr(args, "base_url", None):
        return str(args.base_url).rstrip("/")
    return f"http://{args.host}:{args.port}"


def parse_stages(spec: str) -> list[int]:
    stages = [int(piece) for piece in str(spec).split(",") if piece.strip()]
    if not stages or any(stage < 1 for stage in stages):
        raise ValueError(f"invalid concurrency stages: {spec!r}")
    return stages


# ------------------------------------------------------------------ SSE assembly


def iter_sse_payloads(lines: Iterable[bytes | str]) -> Iterator[dict]:
    """Yield decoded JSON objects from an SSE byte/line stream, stopping at [DONE]."""
    for raw in lines:
        line = raw.decode() if isinstance(raw, bytes) else raw
        line = line.strip()
        if not line or not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if payload == "[DONE]":
            return
        yield json.loads(payload)


def _merge_tool_delta(slots: dict[int, dict[str, str]], call: dict) -> None:
    index = int(call.get("index", 0) or 0)
    slot = slots.setdefault(index, {"id": "", "name": "", "arguments": ""})
    if call.get("id"):
        slot["id"] = str(call["id"])
    function = call.get("function") or {}
    if function.get("name"):
        slot["name"] = str(function["name"])
    if function.get("arguments"):
        slot["arguments"] += str(function["arguments"])


def collect_stream(chunks: Iterable[dict]) -> dict[str, Any]:
    """Concatenate every SSE delta into whole fields before anything inspects them.

    Tokens are split across ``data:`` events; content, reasoning and tool-call
    argument fragments must all be joined first.
    """
    content: list[str] = []
    reasoning: list[str] = []
    slots: dict[int, dict[str, str]] = {}
    finish_reason: str | None = None
    usage: dict | None = None
    events = 0
    for chunk in chunks:
        events += 1
        if chunk.get("usage"):
            usage = chunk["usage"]
        for choice in chunk.get("choices", []):
            delta = choice.get("delta") or {}
            if delta.get("content"):
                content.append(str(delta["content"]))
            for key in ("reasoning_content", "reasoning"):
                if delta.get(key):
                    reasoning.append(str(delta[key]))
            for call in delta.get("tool_calls") or []:
                _merge_tool_delta(slots, call)
            if choice.get("finish_reason"):
                finish_reason = str(choice["finish_reason"])
    return {
        "content": "".join(content),
        "reasoning": "".join(reasoning),
        "tool_calls": [slots[index] for index in sorted(slots)],
        "finish_reason": finish_reason,
        "usage": usage,
        "events": events,
    }


def collect_response(body: dict) -> dict[str, Any]:
    """Reduce a non-streaming chat completion to the shape ``collect_stream`` returns."""
    choices = body.get("choices") or [{}]
    message = choices[0].get("message") or {}
    slots: dict[int, dict[str, str]] = {}
    for index, call in enumerate(message.get("tool_calls") or []):
        function = call.get("function") or {}
        slots[index] = {
            "id": str(call.get("id") or ""),
            "name": str(function.get("name") or ""),
            "arguments": str(function.get("arguments") or ""),
        }
    return {
        "content": str(message.get("content") or ""),
        "reasoning": str(message.get("reasoning_content") or message.get("reasoning") or ""),
        "tool_calls": [slots[index] for index in sorted(slots)],
        "finish_reason": str(choices[0].get("finish_reason") or "") or None,
        "usage": body.get("usage"),
        "events": 1,
    }


def cached_tokens(usage: dict | None) -> int:
    details = (usage or {}).get("prompt_tokens_details") or {}
    return int(details.get("cached_tokens") or 0)


def prompt_tokens(usage: dict | None) -> int:
    return int((usage or {}).get("prompt_tokens") or 0)


def answer_digest(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def first_divergence(left: str, right: str) -> int | None:
    """Index of the first differing character, or None when the strings are equal."""
    if left == right:
        return None
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return min(len(left), len(right))


def decode_tool_arguments(call: dict[str, str]) -> dict | None:
    try:
        parsed = json.loads(call.get("arguments") or "")
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


# ----------------------------------------------------------------- gate metrics


def batch_invariance_failures(
    solo: dict[str, str], concurrent_answers: Iterable[tuple[str, str]]
) -> list[str]:
    """Every concurrent answer must equal the solo answer for the same prompt key."""
    failures: list[str] = []
    seen: set[str] = set()
    for key, text in concurrent_answers:
        seen.add(key)
        if key not in solo:
            failures.append(f"{key}: no solo baseline recorded")
            continue
        reference = solo[key]
        if text == reference:
            continue
        where = first_divergence(reference, text)
        failures.append(
            f"{key}: concurrent answer diverged at char {where} "
            f"({answer_digest(reference)} solo vs {answer_digest(text)} concurrent)"
        )
    for key in solo:
        if key not in seen:
            failures.append(f"{key}: no concurrent answer recorded")
    return failures


def prefix_cache_failures(
    cold: dict[str, Any],
    warm: dict[str, Any],
    *,
    expect_cache_report: bool,
    min_cached_fraction: float = 0.5,
) -> list[str]:
    """Warm shared-prefix reuse must be output-identical and actually hit the cache."""
    failures: list[str] = []
    if cold.get("content") != warm.get("content"):
        where = first_divergence(str(cold.get("content", "")), str(warm.get("content", "")))
        failures.append(
            f"warm answer diverged from the cold answer at char {where} "
            f"({answer_digest(str(cold.get('content', '')))} cold vs "
            f"{answer_digest(str(warm.get('content', '')))} warm)"
        )
    if expect_cache_report:
        warm_cached = cached_tokens(warm.get("usage"))
        cold_cached = cached_tokens(cold.get("usage"))
        warm_prompt = prompt_tokens(warm.get("usage"))
        floor = int(warm_prompt * min_cached_fraction)
        if warm_cached <= cold_cached:
            failures.append(
                f"warm cached_tokens {warm_cached} did not exceed cold {cold_cached}"
            )
        if warm_cached < floor:
            failures.append(
                f"warm cached_tokens {warm_cached} is below {min_cached_fraction:.0%} "
                f"of the {warm_prompt}-token prompt ({floor})"
            )
    return failures


def elastic_ramp_failures(stages: Iterable[dict[str, Any]]) -> list[str]:
    """Every ramp stage must complete with no transport errors and non-empty answers."""
    failures: list[str] = []
    for stage in stages:
        concurrency = stage.get("concurrency")
        for error in stage.get("errors") or []:
            failures.append(f"concurrency {concurrency}: {error}")
        completed = int(stage.get("completed") or 0)
        if completed != int(concurrency or 0):
            failures.append(
                f"concurrency {concurrency}: {completed} of {concurrency} requests completed"
            )
        empty = int(stage.get("empty") or 0)
        if empty:
            failures.append(f"concurrency {concurrency}: {empty} empty answers")
    return failures


def tool_call_failures(
    result: dict[str, Any], *, expected_name: str, required_keys: Iterable[str] = ("city",)
) -> list[str]:
    failures: list[str] = []
    if result.get("finish_reason") != "tool_calls":
        failures.append(f"finish_reason is {result.get('finish_reason')!r}, expected 'tool_calls'")
    calls = result.get("tool_calls") or []
    if not calls:
        failures.append("no tool call was returned")
        return failures
    call = calls[0]
    if call.get("name") != expected_name:
        failures.append(f"tool name is {call.get('name')!r}, expected {expected_name!r}")
    arguments = decode_tool_arguments(call)
    if arguments is None:
        failures.append(f"tool arguments are not a JSON object: {call.get('arguments')!r}")
        return failures
    for key in required_keys:
        if key not in arguments:
            failures.append(f"tool arguments are missing {key!r}: {arguments!r}")
    return failures


# --------------------------------------------------------------------- reporting


def render_markdown(report: dict[str, Any]) -> str:
    """Render one gate run as a benchmarks/results/ markdown section."""
    lines = [
        f"## Serving gates {report.get('date', '')}".rstrip(),
        "",
        f"Endpoint: `{report.get('base_url', '')}`  ",
        f"Model: `{report.get('model', '')}`  ",
        f"Streaming: {'on' if report.get('stream', True) else 'off'}, "
        f"thinking: {'on' if report.get('thinking') else 'off'}",
        "",
        "| Gate | Result | Detail |",
        "|---|---|---|",
    ]
    for gate in report.get("gates", []):
        verdict = "PASS" if gate.get("passed") else "**FAIL**"
        detail = str(gate.get("summary", "")).replace("|", "\\|")
        lines.append(f"| {gate.get('name', '')} | {verdict} | {detail} |")
    failures = [
        (gate.get("name", ""), failure)
        for gate in report.get("gates", [])
        for failure in gate.get("failures", [])
    ]
    if failures:
        lines += ["", "Failures:", ""]
        lines += [f"- `{name}`: {failure}" for name, failure in failures]
    lines.append("")
    return "\n".join(lines)


def default_result_path() -> str:
    date = datetime.date.today().isoformat()
    return os.path.join(RESULTS_DIR, f"nemotron35_lightning_5080_{date}.md")


def write_result_markdown(path: str, text: str) -> str:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "a") as handle:
        handle.write(text if text.endswith("\n") else text + "\n")
    return path


# -------------------------------------------------------------------- transport


def get_json(url: str, timeout: float = 30.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read())


def resolve_model(origin: str, requested: str | None, timeout: float = 30.0) -> str:
    if requested:
        return requested
    return get_json(f"{origin}/v1/models", timeout)["data"][0]["id"]


def chat(
    origin: str,
    body: dict[str, Any],
    *,
    timeout: float,
    stream: bool,
) -> dict[str, Any]:
    """One chat completion; streamed answers are reassembled before returning."""
    payload = dict(body)
    payload["stream"] = stream
    if stream:
        payload["stream_options"] = {"include_usage": True}
    request = urllib.request.Request(
        f"{origin}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as error:
        detail = error.read()[:500]
        raise RuntimeError(f"HTTP {error.code}: {detail!r}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"transport error: {error.reason}") from error
    with response:
        if stream:
            result = collect_stream(iter_sse_payloads(response))
        else:
            result = collect_response(json.loads(response.read()))
    result["seconds"] = time.perf_counter() - started
    return result


def chat_body(
    args: argparse.Namespace,
    model: str,
    messages: list[dict[str, Any]],
    **extra: Any,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": args.max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
    }
    if not args.thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    body.update(extra)
    return body


def run_concurrently(
    tasks: list[tuple[str, dict[str, Any]]],
    origin: str,
    *,
    timeout: float,
    stream: bool,
) -> list[tuple[str, dict[str, Any] | None, str | None]]:
    """Issue every request at once; return (key, result, error) in submission order."""
    outcomes: list[tuple[str, dict[str, Any] | None, str | None]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(tasks))) as pool:
        futures = [
            (key, pool.submit(chat, origin, body, timeout=timeout, stream=stream))
            for key, body in tasks
        ]
        for key, future in futures:
            try:
                outcomes.append((key, future.result(), None))
            except Exception as error:  # noqa: BLE001 - reported as a gate failure
                outcomes.append((key, None, f"{type(error).__name__}: {error}"))
    return outcomes


# ------------------------------------------------------------------------ gates


def gate_batch_invariance(args: argparse.Namespace, origin: str, model: str) -> dict[str, Any]:
    solo: dict[str, str] = {}
    for index, prompt in enumerate(BATCH_PROMPTS):
        body = chat_body(args, model, [{"role": "user", "content": prompt}])
        result = chat(origin, body, timeout=args.timeout, stream=args.stream)
        solo[f"p{index}"] = result["content"]

    concurrency = int(getattr(args, "concurrency", 16))
    tasks = []
    for copy in range(concurrency):
        index = copy % len(BATCH_PROMPTS)
        body = chat_body(args, model, [{"role": "user", "content": BATCH_PROMPTS[index]}])
        tasks.append((f"p{index}", body))
    outcomes = run_concurrently(tasks, origin, timeout=args.timeout, stream=args.stream)

    failures = [f"{key}: {error}" for key, _, error in outcomes if error]
    answers = [(key, result["content"]) for key, result, error in outcomes if not error]
    failures += batch_invariance_failures(solo, answers)
    return {
        "name": "batch-invariance",
        "passed": not failures,
        "failures": failures,
        "summary": f"{len(BATCH_PROMPTS)} solo answers reproduced by {concurrency} concurrent copies",
        "details": {
            "concurrency": concurrency,
            "solo_digests": {key: answer_digest(text) for key, text in solo.items()},
        },
    }


def build_shared_prefix(approx_tokens: int, salt: str) -> str:
    """A unique-per-run filler block of roughly ``approx_tokens`` tokens."""
    header = f"Reference dossier {salt}. Read the notes, then answer the final question.\n"
    line = "Note {index}: crate {index} holds {count} calibrated widgets in bay {bay}.\n"
    # Each note line is ~20 tokens for a byte-level BPE; build a few more than needed
    # and let the server tokenize whatever length results.
    notes = [
        line.format(index=i, count=100 + (i * 7) % 89, bay=i % 13)
        for i in range(max(1, approx_tokens // 16))
    ]
    return header + "".join(notes)


def gate_prefix_cache(args: argparse.Namespace, origin: str, model: str) -> dict[str, Any]:
    salt = uuid.uuid4().hex
    prefix = build_shared_prefix(int(getattr(args, "prefix_tokens", 4096)), salt)
    question_a = "How many widgets are in crate 3? Answer with the number only."
    question_b = "How many widgets are in crate 7? Answer with the number only."

    def ask(question: str) -> dict[str, Any]:
        body = chat_body(
            args, model, [{"role": "user", "content": prefix + "\n" + question}]
        )
        return chat(origin, body, timeout=args.timeout, stream=args.stream)

    # B first, so its answer is produced before the prefix has ever been reused; A then
    # shares the prefix; B again must be identical and land on the cached prefix.
    cold = ask(question_b)
    warm_a = ask(question_a)
    warm = ask(question_b)
    failures = prefix_cache_failures(
        cold,
        warm,
        expect_cache_report=bool(getattr(args, "enable_cache_report", False)),
        min_cached_fraction=float(getattr(args, "min_cached_fraction", 0.5)),
    )
    return {
        "name": "prefix-cache",
        "passed": not failures,
        "failures": failures,
        "summary": (
            f"prompt {prompt_tokens(warm.get('usage'))} tokens, "
            f"cached {cached_tokens(cold.get('usage'))} cold -> "
            f"{cached_tokens(warm.get('usage'))} warm"
        ),
        "details": {
            "cold_seconds": cold.get("seconds"),
            "warm_seconds": warm.get("seconds"),
            "shared_request_seconds": warm_a.get("seconds"),
            "cold_cached_tokens": cached_tokens(cold.get("usage")),
            "warm_cached_tokens": cached_tokens(warm.get("usage")),
            "prompt_tokens": prompt_tokens(warm.get("usage")),
        },
    }


def gate_elastic_ramp(args: argparse.Namespace, origin: str, model: str) -> dict[str, Any]:
    stages: list[dict[str, Any]] = []
    for stage_index, concurrency in enumerate(parse_stages(getattr(args, "stages", "1,6,16,1"))):
        tasks = []
        for slot in range(concurrency):
            prompt = (
                f"Stage {stage_index} request {slot}: "
                f"{BATCH_PROMPTS[slot % len(BATCH_PROMPTS)]}"
            )
            tasks.append((f"s{stage_index}r{slot}", chat_body(args, model, [
                {"role": "user", "content": prompt}
            ])))
        started = time.perf_counter()
        outcomes = run_concurrently(tasks, origin, timeout=args.timeout, stream=args.stream)
        elapsed = time.perf_counter() - started
        stages.append(
            {
                "concurrency": concurrency,
                "errors": [f"{key}: {error}" for key, _, error in outcomes if error],
                "completed": sum(1 for _, result, error in outcomes if not error),
                "empty": sum(
                    1 for _, result, error in outcomes if not error and not result["content"]
                ),
                "seconds": elapsed,
            }
        )
    failures = elastic_ramp_failures(stages)
    return {
        "name": "elastic-ramp",
        "passed": not failures,
        "failures": failures,
        "summary": " -> ".join(
            f"{stage['concurrency']}x{stage['seconds']:.1f}s" for stage in stages
        ),
        "details": {"stages": stages},
    }


def gate_tool_call(args: argparse.Namespace, origin: str, model: str) -> dict[str, Any]:
    body = chat_body(
        args,
        model,
        [
            {
                "role": "user",
                "content": "What is the weather in Boston right now? Use the tool.",
            }
        ],
        tools=[WEATHER_TOOL],
        tool_choice="auto",
        max_completion_tokens=max(args.max_tokens, 128),
    )
    try:
        result = chat(origin, body, timeout=args.timeout, stream=args.stream)
    except RuntimeError as error:
        return {
            "name": "tool-call",
            "passed": False,
            "failures": [str(error)],
            "summary": "request failed",
            "details": {},
        }
    failures = tool_call_failures(
        result, expected_name=str(getattr(args, "tool_name", "get_current_weather"))
    )
    calls = result.get("tool_calls") or []
    return {
        "name": "tool-call",
        "passed": not failures,
        "failures": failures,
        "summary": (
            f"finish_reason={result.get('finish_reason')} "
            f"calls={[call.get('name') for call in calls]}"
        ),
        "details": {"tool_calls": calls, "content": result.get("content")},
    }


GATES = {
    "batch-invariance": gate_batch_invariance,
    "prefix-cache": gate_prefix_cache,
    "elastic-ramp": gate_elastic_ramp,
    "tool-call": gate_tool_call,
}


def selected_gates(name: str) -> list[str]:
    return list(GATES) if name == "all" else [name]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    origin = origin_of(args)
    model = resolve_model(origin, args.model, args.timeout)
    report: dict[str, Any] = {
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "base_url": origin,
        "model": model,
        "stream": args.stream,
        "thinking": args.thinking,
        "gates": [],
    }
    for name in selected_gates(args.gate):
        print(f"[gate] {name} ...", flush=True)
        outcome = GATES[name](args, origin, model)
        report["gates"].append(outcome)
        print(
            f"[gate] {name}: {'PASS' if outcome['passed'] else 'FAIL'} — {outcome['summary']}",
            flush=True,
        )
        for failure in outcome["failures"]:
            print(f"    - {failure}", flush=True)

    report["accepted"] = all(gate["passed"] for gate in report["gates"])
    if args.json_out:
        with open(args.json_out, "a") as handle:
            handle.write(json.dumps(report) + "\n")
    if args.result_md is not None:
        path = args.result_md or default_result_path()
        write_result_markdown(path, render_markdown(report))
        print(f"[gate] wrote {path}", flush=True)
    print(f"[gate] overall: {'PASS' if report['accepted'] else 'FAIL'}", flush=True)
    return 0 if report["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
