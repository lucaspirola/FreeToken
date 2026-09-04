"""CPU-only coverage for bench_decode_moe's server command and log scraping.

No GPU and no server: everything here is argument construction and text parsing."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
import bench_decode_moe as bench  # noqa: E402


def _args(*argv: str):
    return bench.parse_args(["--model", "/m", *argv])


def _flag(cmd: list[str], name: str) -> str | None:
    return cmd[cmd.index(name) + 1] if name in cmd else None


def test_defaults_unchanged_when_new_flags_absent():
    cmd = bench.serve_cmd(_args(), "offload", 8000)
    assert _flag(cmd, "--max-running-requests") == "1"
    assert _flag(cmd, "--cuda-graph-max-bs") == "1"
    assert "--nvfp4-backend" not in cmd
    assert "--moe-collect-stats" not in cmd
    # The exact default command, byte for byte.
    assert cmd == [
        sys.executable, "-m", "freetoken.cli", "serve",
        "--model", "/m",
        "--host", "127.0.0.1", "--port", "8000",
        "--moe-backend", "offload",
        "--max-running-requests", "1",
        "--max-seq-len-override", "8448",
        "--memory-ratio", "0.9",
        "--cuda-graph-max-bs", "1",
        "--moe-hybrid-max-fetch", "-1",
        "--moe-cache-policy", "lru",
        "--moe-cache-auto",
    ]


def test_concurrency_sets_running_requests_and_graph_bs():
    cmd = bench.serve_cmd(_args("--concurrency", "8"), "offload", 8000)
    assert _flag(cmd, "--max-running-requests") == "8"
    assert _flag(cmd, "--cuda-graph-max-bs") == "8"


def test_no_graph_still_disables_the_graph_at_high_concurrency():
    cmd = bench.serve_cmd(_args("--concurrency", "4", "--no-graph"), "offload", 8000)
    assert _flag(cmd, "--max-running-requests") == "4"
    assert _flag(cmd, "--cuda-graph-max-bs") == "0"


def test_nvfp4_backend_passthrough():
    cmd = bench.serve_cmd(_args("--nvfp4-backend", "flashinfer"), "offload", 8000)
    assert _flag(cmd, "--nvfp4-backend") == "flashinfer"


def test_nvfp4_backend_choices_match_the_server():
    args_py = (
        Path(__file__).resolve().parents[2]
        / "python/freetoken/server/args.py"
    ).read_text()
    block = args_py.split('"--nvfp4-backend"', 1)[1].split("choices=", 1)[1]
    server_choices = tuple(json.loads(block.split("]", 1)[0].strip() + "]"))
    assert server_choices == bench.NVFP4_BACKENDS


def test_moe_collect_stats_passthrough():
    cmd = bench.serve_cmd(_args("--moe-collect-stats"), "offload", 8000)
    assert "--moe-collect-stats" in cmd


def test_server_arg_is_appended_verbatim_and_whitespace_split():
    cmd = bench.serve_cmd(
        _args(
            "--server-arg", "--host-ram-reserve-gb 3",
            "--server-arg", "--elastic-initial-requests 4",
        ),
        "offload",
        8000,
    )
    assert cmd[-4:] == [
        "--host-ram-reserve-gb", "3", "--elastic-initial-requests", "4",
    ]


def test_serve_cmd_tolerates_a_namespace_without_the_new_attributes():
    """bench_long_context calls serve_cmd with its own namespace."""
    legacy = SimpleNamespace(
        model="/m",
        max_context=None,
        decode=128,
        mem_ratio=0.97,
        no_graph=False,
        hybrid_fetch=-1,
        cache_policy="lfu",
        kv_cache_dtype="q8_0",
        prefill_chunk=8192,
        prefill_hit_d2d=False,
        cache=0,
        cache_rate=None,
    )
    cmd = bench.serve_cmd(legacy, "offload", 8000)
    assert _flag(cmd, "--max-running-requests") == "1"
    assert _flag(cmd, "--cuda-graph-max-bs") == "1"
    assert "--nvfp4-backend" not in cmd
    assert "--moe-collect-stats" not in cmd


def test_long_context_bench_exposes_the_passthrough_flags():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
    import bench_long_context  # noqa: F401

    p = __import__("argparse").ArgumentParser()
    bench.add_server_passthrough_args(p)
    ns = p.parse_args(["--nvfp4-backend", "triton", "--server-arg", "--x 1"])
    assert ns.nvfp4_backend == "triton"
    assert ns.server_arg == ["--x 1"]
    assert ns.moe_collect_stats is False


LOG = """\
[2026-09-04 10:00:00] Scheduler is idle, waiting for new reqs...
[2026-09-04 10:00:01] MoE decode miss stats: {'layer_calls': 128, 'miss_rate': 0.25}
[2026-09-04 10:00:01] MoE decode miss stats per layer: [{"layer": 0, "steps": 64, \
"miss_rate": 0.5}, {"layer": 1, "steps": 64, "miss_rate": 0.0}]
[2026-09-04 10:00:02] Scheduler is idle, waiting for new reqs...
"""


def test_scrape_moe_stats_parses_both_lines(tmp_path):
    log = tmp_path / "server.log"
    log.write_text(LOG)
    stats, per_layer, count = bench.scrape_moe_stats(str(log))
    assert stats == {"layer_calls": 128, "miss_rate": 0.25}
    assert per_layer == [
        {"layer": 0, "steps": 64, "miss_rate": 0.5},
        {"layer": 1, "steps": 64, "miss_rate": 0.0},
    ]
    assert count == 1


def test_scrape_moe_stats_keeps_the_last_dump(tmp_path):
    log = tmp_path / "server.log"
    log.write_text(
        LOG
        + "MoE decode miss stats: {'layer_calls': 256, 'miss_rate': 0.1}\n"
        + "MoE decode miss stats per layer: "
        + json.dumps([{"layer": 0, "steps": 128}])
        + "\n"
    )
    stats, per_layer, count = bench.scrape_moe_stats(str(log))
    assert stats["layer_calls"] == 256
    assert per_layer == [{"layer": 0, "steps": 128}]
    assert count == 2


def test_scrape_moe_stats_on_a_log_without_stats(tmp_path):
    log = tmp_path / "server.log"
    log.write_text("nothing to see\nScheduler is idle, waiting for new reqs...\n")
    assert bench.scrape_moe_stats(str(log)) == (None, None, 0)


def test_scrape_moe_stats_ignores_a_truncated_line(tmp_path):
    log = tmp_path / "server.log"
    log.write_text(LOG + "MoE decode miss stats per layer: [{'layer': 0,\n")
    stats, per_layer, count = bench.scrape_moe_stats(str(log))
    assert count == 1 and per_layer[0]["layer"] == 0 and stats["layer_calls"] == 128


def test_wait_for_moe_stats_requires_a_newer_dump(tmp_path):
    log = tmp_path / "server.log"
    log.write_text(LOG)
    # One dump already present (the warm-up run): waiting for a second one times out.
    assert bench.wait_for_moe_stats(str(log), seen_before=1, timeout=0.0) == (None, None, 1)
    stats, per_layer, seen = bench.wait_for_moe_stats(str(log), seen_before=0, timeout=0.0)
    assert stats == {"layer_calls": 128, "miss_rate": 0.25}
    assert len(per_layer) == 2 and seen == 1


def _dump(stats: dict, per_layer: list) -> str:
    return (
        f"[ts] MoE decode miss stats: {stats!r}\n"
        f"[ts] MoE decode miss stats per layer: {json.dumps(per_layer)}\n"
    )


def _layer_row(layer, steps, active, missing, fetched=0):
    """A logged per-layer row, i.e. raw counters expressed as per-step ratios."""
    return {
        "layer": layer,
        "steps": steps,
        "active_per_step": active / steps,
        "missing_per_step": missing / steps,
        "miss_rate": missing / active if active else 0.0,
        "fetched_per_step": fetched / steps,
        "pageable_stage_calls": 0,
        "pageable_rows": 0,
        "pageable_plan_wait_seconds": 0.0,
        "pageable_gather_seconds": 0.0,
    }


def _agg(calls, active, missing, **extra):
    return {
        "layer_calls": calls,
        "active_per_layer": active / calls,
        "missing_per_layer": missing / calls,
        "miss_rate": missing / active,
        "fetched_per_layer": 0.0,
        "cpu_per_layer": missing / calls,
        "fetch_rate": 0.0,
        "prefill_hit_rows": extra.get("prefill_hit_rows", 0),
        "prefill_rows": extra.get("prefill_rows", 0),
        "pageable_stage_calls": 0,
        "pageable_rows": 0,
        "pageable_gib": 0.0,
        "pageable_plan_wait_seconds": 0.0,
        "pageable_gather_seconds": 0.0,
    }


# Cold warm-up: layer 0 misses every active expert (100 steps, 8 active, 8 missing).
# Measured window: the same 100 steps but only 10% miss. Layer 1 is a steady 50%.
WARM = ([_layer_row(0, 100, 800, 800), _layer_row(1, 100, 800, 400)])
MEASURED = ([_layer_row(0, 200, 1600, 880), _layer_row(1, 200, 1600, 800)])


def test_moe_stats_delta_isolates_the_measured_window(tmp_path):
    log = tmp_path / "server.log"
    log.write_text(
        _dump(_agg(200, 1600, 1200, prefill_rows=10), WARM)
        + _dump(_agg(400, 3200, 1680, prefill_rows=25), MEASURED)
    )
    stats_a, layers_a, seen = bench.wait_for_moe_stats(str(log), 0, timeout=0.0)
    stats_b, layers_b, seen = bench.wait_for_moe_stats(str(log), seen, timeout=0.0)
    assert seen == 2
    stats, per_layer = bench.moe_stats_delta((stats_a, layers_a), (stats_b, layers_b))

    # Cumulative reads 1680/3200 = 52.5%; the measured window alone is 480/1600 = 30%.
    assert stats_b["miss_rate"] == pytest.approx(0.525)
    assert stats["miss_rate"] == pytest.approx(0.30)
    assert stats["layer_calls"] == 200
    assert stats["active_per_layer"] == pytest.approx(8.0)
    assert stats["missing_per_layer"] == pytest.approx(2.4)
    # A layer that missed 100% cold and 10% warm reports 10%, not the ~55% average.
    assert per_layer[0]["miss_rate"] == pytest.approx(0.10)
    assert per_layer[0]["steps"] == 100
    assert per_layer[0]["missing_per_step"] == pytest.approx(0.8)
    # A layer whose behaviour did not change is unaffected by the differencing.
    assert per_layer[1]["miss_rate"] == pytest.approx(0.50)
    # Cumulative scalars are differenced, ratios recomputed.
    assert stats["prefill_rows"] == 15


def test_moe_stats_delta_zeroes_a_layer_with_no_new_steps():
    a = ([_layer_row(0, 100, 800, 800)])
    stats, per_layer = bench.moe_stats_delta(
        (_agg(100, 800, 800), a), (_agg(100, 800, 800), a)
    )
    assert per_layer[0] == {
        "layer": 0,
        "steps": 0,
        "active_per_step": 0.0,
        "missing_per_step": 0.0,
        "miss_rate": 0.0,
        "fetched_per_step": 0.0,
        "pageable_stage_calls": 0,
        "pageable_rows": 0,
        "pageable_plan_wait_seconds": 0.0,
        "pageable_gather_seconds": 0.0,
    }
    assert stats["layer_calls"] == 0 and stats["miss_rate"] == 0.0


def test_load_problems_wraps(tmp_path):
    rows = tmp_path / "aime.jsonl"
    rows.write_text(
        "\n".join(
            json.dumps({"problem": f"p{i} boxed", "answer": str(i)}) for i in range(3)
        )
    )
    got = bench.load_problems(str(rows), 2, 3)
    assert [a for _, a in got] == ["2", "0", "1"]
    assert bench.load_problem(str(rows), 1) == ("p1 boxed", "1")


def test_load_problems_appends_the_boxed_instruction(tmp_path):
    rows = tmp_path / "aime.jsonl"
    rows.write_text(json.dumps({"problem": "what is 1+1?", "answer": "2"}))
    text, _ = bench.load_problems(str(rows), 0, 1)[0]
    assert text.endswith(bench.BOXED_INSTRUCTION)


def _stream(t0: float, first: float, n: int, dt: float, text: str = "x") -> dict:
    """A stub streamed result: n token events, dt apart, starting at ``first``."""
    return {
        "t0": t0,
        "stamps": [first + i * dt for i in range(n)],
        "text": text,
        "usage": {"completion_tokens": n, "prompt_tokens": 10},
    }


def test_decode_metrics_single_stream_keeps_the_bs1_definitions():
    m = bench.decode_metrics([_stream(0.0, 1.0, 5, 0.1)], 5)
    assert m["decode_steps"] == 4
    assert m["decode_tok_s"] == pytest.approx(4 / 0.4)
    assert m["ms_per_token"] == pytest.approx(100.0)
    assert m["ttft_ms"] == pytest.approx(1000.0)
    assert m["events"] == 5
    # No concurrency-only keys leak into a bs=1 row.
    assert "decode_tok_s_streams" not in m
    assert "ttft_ms_max" not in m


def test_decode_metrics_aggregate_spans_all_streams():
    # Stream A: 5 events 0.1 s apart from t=1.0 (ends 1.4). Stream B: 5 events 0.2 s
    # apart from t=1.2 (ends 2.0). Aggregate window = 2.0 - 1.0 = 1.0 s, 8 steps.
    m = bench.decode_metrics([_stream(0.0, 1.0, 5, 0.1), _stream(0.0, 1.2, 5, 0.2)], 5)
    assert m["decode_steps_total"] == 8
    assert m["decode_window_s"] == pytest.approx(1.0)
    assert m["decode_tok_s_aggregate"] == pytest.approx(8.0)
    assert m["decode_tok_s"] == pytest.approx(8.0)  # headline is the aggregate for N>1
    assert m["decode_tok_s_streams"] == pytest.approx([10.0, 5.0])
    assert m["decode_tok_s_stream_median"] == pytest.approx(7.5)
    assert m["decode_tok_s_stream_min"] == pytest.approx(5.0)
    assert m["ttft_ms_p50"] == pytest.approx(1100.0)
    assert m["ttft_ms_max"] == pytest.approx(1200.0)
    assert m["ms_per_token"] == pytest.approx(125.0)


def test_decode_metrics_hashes_every_stream_separately():
    m = bench.decode_metrics(
        [_stream(0.0, 1.0, 3, 0.1, "a"), _stream(0.0, 1.0, 3, 0.1, "b")], 3
    )
    assert len(set(m["output_sha1_streams"])) == 2
    assert m["output_sha1"] == m["output_sha1_streams"][0]


def test_decode_metrics_rejects_a_stream_with_one_event():
    with pytest.raises(SystemExit):
        bench.decode_metrics([_stream(0.0, 1.0, 1, 0.1)], 1)


def test_concurrency_must_be_positive():
    with pytest.raises(SystemExit):
        bench.main(["--model", "/m", "--concurrency", "0"])
