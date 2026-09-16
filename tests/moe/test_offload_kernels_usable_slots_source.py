"""Source-level check for the expert-arena prerequisite (no torch import).

The gated (``FREETOKEN_EXPERT_ARENA=1``) LRU/eviction/index kernels must read their
slot range and usable-slot bound from device memory (a pointer, loaded in-kernel via
``tl.load``) instead of receiving ``class_begin``/``class_end`` as host scalars: a host
scalar becomes part of a captured CUDA graph node's frozen launch state, so a later
change is invisible to an already-captured graph, while a value loaded from a
persistent device tensor is re-read on every replay. This mirrors the AST-extraction
pattern in tests/engine/test_growable_kv_transaction_source.py, but only inspects
signatures/toggles textually -- it does not execute any Triton/CUDA code.
"""

from __future__ import annotations

import ast
from pathlib import Path

KERNELS = Path(__file__).parents[2] / "python/freetoken/moe/offload_kernels.py"
CACHE = Path(__file__).parents[2] / "python/freetoken/moe/offload_cache.py"

# The legacy kernels this step must NOT touch (default path, gate off).
_LEGACY_SIZED_KERNELS = {
    "_ensure_experts_sized_kernel",
    "_materialize_layer_sized_kernel",
    "_ensure_experts_hybrid_kernel",
}
# Their gated pointer-argument twins, added by this step.
_GATED_SIZED_KERNELS = {
    "_ensure_experts_sized_kernel_v2",
    "_materialize_layer_sized_kernel_v2",
    "_ensure_experts_hybrid_kernel_v2",
}


def _kernel_defs(tree: ast.Module) -> dict[str, ast.FunctionDef]:
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in _LEGACY_SIZED_KERNELS | _GATED_SIZED_KERNELS
    }


def _arg_names(fn: ast.FunctionDef) -> set[str]:
    return {a.arg for a in fn.args.args}


def test_gate_flag_exists_and_defaults_off():
    src = KERNELS.read_text()
    assert "FREETOKEN_EXPERT_ARENA" in src
    tree = ast.parse(src)
    assigns = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "FREETOKEN_EXPERT_ARENA"
            for t in node.targets
        )
    ]
    assert assigns, "FREETOKEN_EXPERT_ARENA module-level flag not found"
    # Must be derived from os.getenv(..., "0") (or equivalent falsy default) so the
    # legacy host-scalar kernels are the default path.
    call_src = ast.get_source_segment(src, assigns[0].value)
    assert "os.getenv" in call_src
    assert '"0"' in call_src or "'0'" in call_src


def test_legacy_sized_kernels_unchanged_signature_keeps_host_scalars():
    """The default (gate-off) path must be untouched: these still take
    class_begin/class_end as plain host scalars, exactly as before this step."""
    tree = ast.parse(KERNELS.read_text())
    defs = _kernel_defs(tree)
    for name in _LEGACY_SIZED_KERNELS:
        assert name in defs, f"legacy kernel {name} missing"
        args = _arg_names(defs[name])
        assert "class_begin" in args, name
        assert "class_end" in args, name


def test_gated_kernels_receive_pointers_not_host_scalars():
    """The v2 (FREETOKEN_EXPERT_ARENA=1) kernels must not take class_begin/class_end
    as separate host scalar params; they take a device-resident bounds pointer
    (and, where a cache-size bound is masked, a usable-slots pointer) instead."""
    tree = ast.parse(KERNELS.read_text())
    defs = _kernel_defs(tree)
    for name in _GATED_SIZED_KERNELS:
        assert name in defs, f"gated kernel {name} missing"
        args = _arg_names(defs[name])
        assert "class_begin" not in args, f"{name} still takes class_begin as a host scalar"
        assert "class_end" not in args, f"{name} still takes class_end as a host scalar"
        assert "bounds_ptr" in args, f"{name} must read its slot range from a device pointer"


def test_gated_ensure_kernels_mask_against_loaded_usable_bound():
    """The two ensure-experts kernels' victim-selection mask must compare against a
    usable value loaded in-kernel, not a compile-time/host cache_size scalar."""
    src = KERNELS.read_text()
    tree = ast.parse(src)
    defs = _kernel_defs(tree)
    for name in ("_ensure_experts_sized_kernel_v2", "_ensure_experts_hybrid_kernel_v2"):
        fn = defs[name]
        assert "usable_ptr" in _arg_names(fn), name
        body_src = ast.get_source_segment(src, fn)
        assert "tl.load(usable_ptr)" in body_src, name
        assert "off_c < usable" in body_src, name


def test_wrapper_functions_branch_on_gate_and_call_the_matching_kernel():
    src = KERNELS.read_text()
    tree = ast.parse(src)
    wrappers = {
        n.name: n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name in {
            "_ensure_experts_sized_gpu",
            "_materialize_layer_sized_gpu",
            "_ensure_experts_hybrid_gpu",
        }
    }
    assert len(wrappers) == 3
    for name, fn in wrappers.items():
        body_src = ast.get_source_segment(src, fn)
        assert "FREETOKEN_EXPERT_ARENA" in body_src, name
        assert "_kernel_v2[" in body_src, f"{name} must dispatch to its gated *_kernel_v2 twin"


def test_offload_cache_exposes_device_side_usable_slots_and_bounds():
    """Companion check on offload_cache.py: OffloadMoeCache must carry a device
    tensor for usable_slots and expose the per-layer bounds via a *_device method,
    with lru_slot_range (host ints) kept as the fallback/debug path."""
    tree = ast.parse(CACHE.read_text())
    cache_cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "OffloadMoeCache"
    )
    method_names = {
        n.name
        for n in cache_cls.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "lru_slot_range" in method_names
    assert "lru_slot_range_device" in method_names

    post_init = next(
        n for n in cache_cls.body if isinstance(n, ast.FunctionDef) and n.name == "__post_init__"
    )
    post_init_src = ast.get_source_segment(CACHE.read_text(), post_init)
    assert "self.slot_capacity" in post_init_src
    assert "self.usable_slots" in post_init_src
