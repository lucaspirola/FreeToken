"""Prefill (extend) attention cost at long context: the shared triton kernel vs alternatives.

One 8192-token chunk attending to a q8_0-quantized prefix of P tokens plus itself (causal),
per attention layer, for the two production geometries. Reports ms/layer and the effective
TFLOP/s against the causal FLOP count.

    python bench_extend.py [sweep] [fi] [gqa] [path]

  default : the current kernel (extend_launch_config) for every geometry and prefix
  sweep   : also every (BLOCK_M, BLOCK_N, num_warps, num_stages) the env override accepts
  fi      : also flashinfer on a bf16 copy of the prefix (dequant timed separately)
  gqa     : also the GQA head-packed split kernel (FREETOKEN_EXTEND_GQA=1) at several tiles
  path    : also the integrated flashinfer path (host_lens), per BENCH_FI_BLOCKS block size
"""
from __future__ import annotations

import itertools
import os
import sys

import torch
import triton.testing as tt

from freetoken.kernel.triton import attention as _attn
from freetoken.kernel.triton.attention import extend_paged_attention
from freetoken.kvcache.quant import BLOCK, resolve_kv_quant

GEOMS = {
    "ornith (16q/2kv, D256)": (16, 2, 256),
    "nemotron (32q/2kv, D128)": (32, 2, 128),
}
PREFIXES = [0, 32768, 65536, 122880]
CHUNK = 8192
dev = torch.device("cuda")
q8 = resolve_kv_quant("q8_0")


def flops(hq, d, p, c):
    # QK^T and PV over the visible keys: prefix fully, the chunk causally.
    visible = c * p + c * (c + 1) / 2
    return 4 * hq * d * visible


def setup(hq, hkv, d, p):
    torch.manual_seed(0)
    q = torch.randn(CHUNK, hq, d, device=dev, dtype=torch.bfloat16)
    ke = torch.randn(CHUNK, hkv, d, device=dev, dtype=torch.bfloat16)
    ve = torch.randn(CHUNK, hkv, d, device=dev, dtype=torch.bfloat16)
    slots = max(p, 1)
    kp = torch.randn(slots, hkv, d, device=dev, dtype=torch.bfloat16)
    vp = torch.randn(slots, hkv, d, device=dev, dtype=torch.bfloat16)
    kq, ks = q8.quantize(kp)
    vq, vs = q8.quantize(vp)
    meta = dict(
        qo_indptr=torch.tensor([0, CHUNK], dtype=torch.int32, device=dev),
        kv_indptr=torch.tensor([0, p], dtype=torch.int32, device=dev),
        kv_indices=torch.arange(slots, dtype=torch.int32, device=dev),
        prefix_lens=torch.tensor([p], dtype=torch.int32, device=dev),
    )
    return q, ke, ve, kq, ks, vq, vs, meta


def run_triton(q, ke, ve, kq, ks, vq, vs, meta, d, host_lens=None):
    return extend_paged_attention(
        q=q, k_cache=kq, v_cache=vq, max_q_len=CHUNK, sm_scale=d ** -0.5,
        k_extend=ke, v_extend=ve, k_scale=ks, v_scale=vs, host_lens=host_lens, **meta,
    )


def run_fi_path(args, d, p):
    """The integrated path: extend_paged_attention with host lengths (flashinfer)."""
    return run_triton(*args, d, host_lens=([CHUNK], [p], [p + CHUNK]))


def dequant(xq, xs):
    return (xq.float().unflatten(-1, (-1, BLOCK)) * xs.float().unsqueeze(-1)).flatten(-2).to(torch.bfloat16)


def main():
    do_sweep = "sweep" in sys.argv
    do_fi = "fi" in sys.argv
    print(f"device {torch.cuda.get_device_name()}  chunk {CHUNK}  kv q8_0")
    for name, (hq, hkv, d) in GEOMS.items():
        print(f"== {name}")
        for p in PREFIXES:
            args = setup(hq, hkv, d, p)
            q, ke, ve, kq, ks, vq, vs, meta = args
            f = flops(hq, d, p, CHUNK)
            ref = run_triton(*args, d)
            t = tt.do_bench(lambda: run_triton(*args, d), rep=200)
            line = f"  P={p:6d}: triton default {t:8.2f} ms ({f / t / 1e9:6.1f} TFLOP/s)"
            if "path" in sys.argv:
                for blk in os.getenv("BENCH_FI_BLOCKS", "16384").split():
                    os.environ["FREETOKEN_EXTEND_FI_BLOCK"] = blk
                    torch.cuda.reset_peak_memory_stats()
                    base = torch.cuda.memory_allocated()
                    out = run_fi_path(args, d, p)
                    peak = (torch.cuda.max_memory_allocated() - base) / 2**20
                    err = (out.float() - ref.float()).abs().max().item()
                    tp = tt.do_bench(lambda: run_fi_path(args, d, p), rep=200)
                    line += (f"\n      fi path blk{blk}: {tp:8.2f} ms ({f / tp / 1e9:6.1f} TFLOP/s)"
                             f" x{t / tp:4.2f}  max|d| {err:.3g}  peak +{peak:.0f} MiB")
            if do_fi:
                try:
                    import flashinfer
                    kfull = torch.cat([dequant(kq[:p], ks[:p]), ke]) if p else ke
                    vfull = torch.cat([dequant(vq[:p], vs[:p]), ve]) if p else ve
                    fi = lambda: flashinfer.single_prefill_with_kv_cache(  # noqa: E731
                        q, kfull, vfull, causal=True, sm_scale=d ** -0.5)
                    out = fi()
                    tf = tt.do_bench(fi, rep=200)
                    tdq = tt.do_bench(lambda: (dequant(kq[:p], ks[:p]), dequant(vq[:p], vs[:p])), rep=100) if p else 0.0
                    err = (out.float() - ref.float()).abs().max().item()
                    line += (f" | flashinfer bf16 {tf:8.2f} ms ({f / tf / 1e9:6.1f} TFLOP/s)"
                             f" + torch dequant {tdq:6.2f} ms, max|d| {err:.3g}")
                except Exception as e:  # report, keep going
                    line += f" | flashinfer failed: {type(e).__name__}: {str(e)[:120]}"
            if "gqa" in sys.argv:
                for bm, w in ((32, 4), (64, 4), (64, 8), (128, 8), (128, 4)):
                    os.environ.update(FREETOKEN_EXTEND_GQA="1", FREETOKEN_EXTEND_GQA_BLOCK_M=str(bm),
                                      FREETOKEN_EXTEND_GQA_NUM_WARPS=str(w))
                    try:
                        out = run_triton(*args, d)
                        err = (out.float() - ref.float()).abs().max().item()
                        tg = tt.do_bench(lambda: run_triton(*args, d), rep=100)
                        line += f"\n      gqa M{bm} w{w}: {tg:8.2f} ms ({f / tg / 1e9:6.1f} TFLOP/s) max|d| {err:.3g}"
                    except Exception as e:
                        line += f"\n      gqa M{bm} w{w}: failed {type(e).__name__}: {str(e)[:100]}"
                for k in ("FREETOKEN_EXTEND_GQA", "FREETOKEN_EXTEND_GQA_BLOCK_M", "FREETOKEN_EXTEND_GQA_NUM_WARPS"):
                    os.environ.pop(k, None)
            print(line, flush=True)
            if do_sweep and p == PREFIXES[-1]:
                best = (t, "default")
                for bm, bn, w, s in itertools.product((32, 64, 128), (32, 64, 128), (4, 8), (1, 2, 3)):
                    os.environ.update(FREETOKEN_EXTEND_BLOCK_M=str(bm), FREETOKEN_EXTEND_BLOCK_N=str(bn),
                                      FREETOKEN_EXTEND_NUM_WARPS=str(w), FREETOKEN_EXTEND_NUM_STAGES=str(s))
                    _attn._extend_launch_env_override.cache_clear()   # read once per process otherwise
                    try:
                        out = run_triton(*args, d)
                        err = (out.float() - ref.float()).abs().max().item()
                        ts = tt.do_bench(lambda: run_triton(*args, d), rep=100)
                        print(f"    M{bm} N{bn} w{w} s{s}: {ts:8.2f} ms  max|d| {err:.3g}", flush=True)
                        if err < 0.05 and ts < best[0]:
                            best = (ts, f"M{bm} N{bn} w{w} s{s}")
                    except Exception as e:
                        print(f"    M{bm} N{bn} w{w} s{s}: failed {type(e).__name__}", flush=True)
                for k in ("FREETOKEN_EXTEND_BLOCK_M", "FREETOKEN_EXTEND_BLOCK_N",
                          "FREETOKEN_EXTEND_NUM_WARPS", "FREETOKEN_EXTEND_NUM_STAGES"):
                    os.environ.pop(k, None)
                _attn._extend_launch_env_override.cache_clear()
                print(f"    best at P={p}: {best[1]} {best[0]:.2f} ms (default {t:.2f})")
            del args, q, ke, ve, kq, ks, vq, vs, ref
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
