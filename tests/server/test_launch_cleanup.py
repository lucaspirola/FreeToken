from __future__ import annotations

from freetoken import launch


def _context() -> launch.LaunchContext:
    return launch.LaunchContext(
        server=launch.resolve_server_url("http://127.0.0.1:1919"),
        model=launch.ServedModel("ornith", ["ornith"], 262_144),
        extra_args=[],
        dry_run=True,
        launch_id="launch-test",
    )


def test_claude_launch_adds_correlation_header_and_cleanup():
    spec = launch.prepare_claude(_context())
    assert "X-FreeToken-Launch-Id: launch-test" in spec.env["ANTHROPIC_CUSTOM_HEADERS"]
    assert spec.cleanup_url.endswith("/v1/client-launches/launch-test/sessions")


def test_codex_profile_adds_correlation_header_and_cleanup():
    ctx = _context()
    text = launch._codex_profile_text(ctx, launch.Path("/tmp/catalog.json"))
    assert '"X-FreeToken-Launch-Id" = "launch-test"' in text
    assert launch.prepare_codex(ctx).cleanup_url.endswith(
        "/v1/client-launches/launch-test/sessions"
    )


def test_run_command_always_calls_launch_cleanup(monkeypatch):
    cleaned = []
    monkeypatch.setattr(
        launch.subprocess,
        "run",
        lambda *_args, **_kwargs: type("Result", (), {"returncode": 7})(),
    )
    monkeypatch.setattr(launch, "_cleanup_client_launch", cleaned.append)
    spec = launch.CommandSpec(["agent"], {}, cleanup_url="http://cleanup")

    assert launch.run_command(spec) == 7
    assert cleaned == ["http://cleanup"]
