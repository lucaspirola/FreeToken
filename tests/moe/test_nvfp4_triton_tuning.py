"""Config-table plumbing for the tuned Triton NVFP4 fused-MoE kernels.

``benchmarks/tune_nvfp4_moe.py`` writes two things: a Python dict of decode tiles keyed by
``(N, K, top_k, sm_count)`` and per-GEMM M-bucketed prefill JSON under
``moe/configs/triton_<ver>/``. Both must degrade to the pre-existing heuristics on every
host that was never swept -- that fallback is what these tests pin (they need no GPU).

The numerics of the fused ReLU^2 epilogue those configs drive live next door in
``test_nvfp4_backends.py`` (``test_triton_relu2_*``), where the bank fixtures are.
"""

from __future__ import annotations

import json

E, H, I = 128, 2688, 1856  # noqa: E741 -- Nemotron 3.5 Lightning MoE shape notation




def test_nvfp4_moe_config_falls_back_without_a_tuned_file():
    """An unswept device name must yield the heuristic, not raise."""
    from freetoken.moe.fused_nvfp4 import _prefill_config_default, nvfp4_moe_config

    for m in (1, 16, 65, 8192):
        assert nvfp4_moe_config(m, I, H, "No_Such_GPU_9999") == _prefill_config_default(m)


def test_nvfp4_moe_config_reads_and_buckets_a_tuned_file(tmp_path, monkeypatch):
    """FREETOKEN_MOE_CONFIG_DIR override, the filename contract, and nearest-bucket
    selection (the loader is what ``benchmarks/tune_nvfp4_moe.py --write`` targets)."""
    import triton

    from freetoken.moe import fused_nvfp4 as fn

    device_name = "Test_GPU"
    version_dir = tmp_path / f"triton_{triton.__version__.replace('.', '_')}"
    version_dir.mkdir()
    small = dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=32, BLOCK_SIZE_KB=32,
                 GROUP_SIZE_M=1, num_warps=4, num_stages=3)
    big = dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=128, BLOCK_SIZE_KB=64,
               GROUP_SIZE_M=8, num_warps=8, num_stages=4)
    path = version_dir / fn.nvfp4_config_filename(E, I, H, device_name)
    assert path.name == f"nvfp4,E={E},N={I},K={H},device_name={device_name}.json"
    path.write_text(json.dumps({"16": small, "8192": big}))

    monkeypatch.setenv("FREETOKEN_MOE_CONFIG_DIR", str(tmp_path))
    fn._load_nvfp4_moe_configs.cache_clear()
    try:
        assert fn.nvfp4_moe_config(1, I, H, device_name, E) == small
        assert fn.nvfp4_moe_config(1000, I, H, device_name, E) == small  # nearest bucket
        assert fn.nvfp4_moe_config(6000, I, H, device_name, E) == big
        # a shape with no table still falls back, even though the device has one
        assert fn.nvfp4_moe_config(16, 999, 999, device_name, E) == fn._prefill_config_default(16)
    finally:
        fn._load_nvfp4_moe_configs.cache_clear()


def test_shipped_prefill_configs_are_complete():
    """Every tuned JSON in-tree must key the documented M buckets with the exact set of
    keys ``_prefill_gemm`` forwards to the kernel (a typo here is a launch error)."""
    from pathlib import Path

    from freetoken.moe import fused_nvfp4 as fn

    PREFILL_CONFIG_KEYS = fn.PREFILL_CONFIG_KEYS
    PREFILL_M_BUCKETS = fn.PREFILL_M_BUCKETS
    configs = Path(fn.__file__).with_name("configs")
    files = sorted(configs.glob("triton_*/nvfp4,*.json"))
    assert files, "no tuned NVFP4 prefill configs shipped"
    for path in files:
        table = json.loads(path.read_text())
        assert {int(k) for k in table} == set(PREFILL_M_BUCKETS), path
        for cfg in table.values():
            assert set(cfg) == set(PREFILL_CONFIG_KEYS), path


def test_decode_marlin_config_falls_back_and_finds_tuned_entries():
    """Unknown ``(N, K, top_k, sm)`` -> the generic constants; a swept key -> its entry."""
    from freetoken.moe import fused_nvfp4 as fn

    generic = {
        "BLOCK_SIZE_N": fn._DECODE_MARLIN_BLOCK_N,
        "BLOCK_SIZE_KW": fn._DECODE_MARLIN_BLOCK_KW,
        "num_warps": fn._DECODE_MARLIN_WARPS,
    }
    assert fn.decode_marlin_config(768, 4096, 8, 132) == generic
    for (n, k, top_k, sm), cfg in fn._DECODE_MARLIN_CONFIGS.items():
        assert fn.decode_marlin_config(n, k, top_k, sm) == cfg
        assert set(cfg) == set(generic)
        # a different SM count is a different GPU -> must not borrow the tuned tile
        assert fn.decode_marlin_config(n, k, top_k, sm + 1) == generic


def test_prefill_launch_env_override_forces_every_key(monkeypatch):
    """``FREETOKEN_NVFP4_PREFILL_*`` are the twins of ``FREETOKEN_EXTEND_*`` /
    ``FREETOKEN_DECODE_*``: they make a tile A/B two invocations of one binary. Every key
    the kernel takes must be reachable, and an unset variable must not disturb the table."""
    from freetoken.moe import fused_nvfp4 as fn

    base = fn.nvfp4_moe_config(8192, I, H, "No_Such_GPU_9999")
    forced = dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=256, BLOCK_SIZE_KB=16,
                  GROUP_SIZE_M=4, num_warps=8, num_stages=2)
    assert set(forced) == set(fn.PREFILL_CONFIG_KEYS)
    for key, var in fn._PREFILL_ENV_KEYS.items():
        monkeypatch.setenv(var, str(forced[key]))
    assert fn.nvfp4_moe_config(8192, I, H, "No_Such_GPU_9999") == forced
    for var in fn._PREFILL_ENV_KEYS.values():
        monkeypatch.delenv(var)
    assert fn.nvfp4_moe_config(8192, I, H, "No_Such_GPU_9999") == base


def test_nearest_bucket_ties_go_to_the_smaller_bucket(tmp_path, monkeypatch):
    """Nearest is on |bucket - M| and a tie takes the SMALLER bucket.

    Pinned because it is the rule that decides which M a new bucket steals: it must not
    depend on the JSON's key order (``min`` over a dict would otherwise settle a tie by
    insertion order). 384 is exactly between 256 and 512 and stays with 256.
    """
    import triton

    from freetoken.moe import fused_nvfp4 as fn

    device_name = "Tie_GPU"
    version_dir = tmp_path / f"triton_{triton.__version__.replace('.', '_')}"
    version_dir.mkdir()
    tiles = {b: dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=64, BLOCK_SIZE_KB=64,
                     GROUP_SIZE_M=1, num_warps=4, num_stages=b) for b in (256, 512)}
    # written HIGH bucket first: order must not decide the tie
    (version_dir / fn.nvfp4_config_filename(E, I, H, device_name)).write_text(
        json.dumps({"512": tiles[512], "256": tiles[256]})
    )
    monkeypatch.setenv("FREETOKEN_MOE_CONFIG_DIR", str(tmp_path))
    fn._load_nvfp4_moe_configs.cache_clear()
    try:
        picked = {m: fn.nvfp4_moe_config(m, I, H, device_name, E)["num_stages"]
                  for m in (256, 383, 384, 385, 512, 900)}
    finally:
        fn._load_nvfp4_moe_configs.cache_clear()
    assert picked == {256: 256, 383: 256, 384: 256, 385: 512, 512: 512, 900: 512}


def test_m512_is_served_by_its_own_bucket_at_block_m_32():
    """soak ticket (moe_prefill_leftovers S2.4): before the "512" bucket existed, M=512
    rounded to "256" and ran at ``BLOCK_M=16``, where ``BLOCK_M=32`` measures 1.10x on two
    independent routings. The bucket must exist, must be the 256 entry with ONLY
    ``BLOCK_SIZE_M`` changed (that pair is what was measured; the 1024 bucket's wider tile
    was not), and must own the M window up to the midpoint with 1024.
    """
    from pathlib import Path

    from freetoken.moe import fused_nvfp4 as fn

    assert 512 in fn.PREFILL_M_BUCKETS
    device = "NVIDIA_GeForce_RTX_5080"
    configs = Path(fn.__file__).with_name("configs")
    tables = sorted(configs.glob(f"triton_*/nvfp4,*device_name={device}.json"))
    assert tables, "the 5080 prefill tables are the ones this ticket measured"
    for path in tables:
        table = json.loads(path.read_text())
        assert set(table["512"]) == set(table["256"]), path
        assert table["512"]["BLOCK_SIZE_M"] == 32, path
        assert table["256"]["BLOCK_SIZE_M"] == 16, path
        for key in fn.PREFILL_CONFIG_KEYS:
            if key != "BLOCK_SIZE_M":
                assert table["512"][key] == table["256"][key], f"{path}[{key}]"

    # The selection this fixes, through the real loader on the real table.
    fn._load_nvfp4_moe_configs.cache_clear()
    try:
        for m, want in ((385, 32), (512, 32), (768, 32), (384, 16), (256, 16), (1024, 16)):
            cfg = fn.nvfp4_moe_config(m, I, H, device, E)
            assert cfg["BLOCK_SIZE_M"] == want, (m, cfg)
    finally:
        fn._load_nvfp4_moe_configs.cache_clear()


def test_skip_buckets_env_restores_the_previous_bucket_boundary(monkeypatch):
    """The A/B hatch for the bucket change itself.

    ``FREETOKEN_NVFP4_PREFILL_BLOCK_M`` cannot express it -- it forces one BLOCK_M on every
    bucket, which would also drag 4096/8192 off their tuned 64 -- so a soak grading the 512
    bucket needs the skip list to reproduce the pre-change lookup exactly.
    """
    from freetoken.moe import fused_nvfp4 as fn

    device = "NVIDIA_GeForce_RTX_5080"
    fn._load_nvfp4_moe_configs.cache_clear()
    try:
        assert fn.nvfp4_moe_config(512, I, H, device, E)["BLOCK_SIZE_M"] == 32
        monkeypatch.setenv("FREETOKEN_NVFP4_PREFILL_SKIP_BUCKETS", "512")
        assert fn.nvfp4_moe_config(512, I, H, device, E) == fn.nvfp4_moe_config(
            256, I, H, device, E
        )
        assert fn.nvfp4_moe_config(8192, I, H, device, E)["BLOCK_SIZE_M"] == 64  # untouched
        # a typo degrades to the shipped table, not to the heuristic fallback
        monkeypatch.setenv("FREETOKEN_NVFP4_PREFILL_SKIP_BUCKETS", "five-twelve, ")
        assert fn.nvfp4_moe_config(512, I, H, device, E)["BLOCK_SIZE_M"] == 32
        # skipping everything falls back to the heuristic rather than raising
        monkeypatch.setenv(
            "FREETOKEN_NVFP4_PREFILL_SKIP_BUCKETS",
            ",".join(str(b) for b in fn.PREFILL_M_BUCKETS),
        )
        assert fn.nvfp4_moe_config(512, I, H, device, E) == fn._prefill_config_default(512)
    finally:
        fn._load_nvfp4_moe_configs.cache_clear()


def test_shipped_prefill_block_kb_is_kernel_legal():
    """Two hard constraints of ``_prefill_nvfp4_moe_kernel``'s K-loop, neither of which
    raises until a launch: the scale broadcast needs ``BLOCK_SIZE_KB % 8 == 0`` (one e4m3
    scale covers 8 packed bytes) and ``tl.dot`` needs ``K >= 16``, and the dot's K is
    ``BLOCK_SIZE_KB``. The same holds for the heuristic fallback."""
    import json
    from pathlib import Path

    from freetoken.moe import fused_nvfp4 as fn

    def check(cfg, where):
        kb = cfg["BLOCK_SIZE_KB"]
        assert kb % 8 == 0, f"{where}: BLOCK_SIZE_KB={kb} is not a multiple of 8"
        assert kb >= 16, f"{where}: BLOCK_SIZE_KB={kb} is below tl.dot's K>=16"

    configs = Path(fn.__file__).with_name("configs")
    for path in sorted(configs.glob("triton_*/nvfp4,*.json")):
        for bucket, cfg in json.loads(path.read_text()).items():
            check(cfg, f"{path.name}[{bucket}]")
    for m in (1, 16, 65, 8192):
        check(fn._prefill_config_default(m), f"_prefill_config_default({m})")
