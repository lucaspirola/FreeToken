# MoE prefill leftovers (handover item 2) — 5080 / Nemotron 3.5 Lightning NVFP4

Date 2026-09-05. Base `785a278`. Card RTX 5080 16 GB, WSL host 34 GiB. Every GPU job under
`scripts/gpu_lock.sh`, never piped, attended to exit. Synthetic banks, uniform routing,
3 warmup + 9 timed, median.

| # | leftover | verdict | number |
|---|---|---|---|
| a | gemm1 emits gemm2's k-planes, gemm2's prepass disappears | **SHIPPED, on by default** | **1.018x** at M=8192 (1.237x over the pre-`ca7e74b` kernel), **bit-exact at every M** |
| b | the M=256 GEMM bucket at `BLOCK_M=16` | **no change** — and the ticket's ceiling was the wrong one | `BLOCK_M=32` is a **12 % loss** at M=256; the bucket is at **69 % of the HBM roofline**, not 20 % of anything. New finding at **M=512: 1.10x** left on the table |
| c | the 1,024-routed-id extend guard under LFU | **documented only**, no code | §3 |

One new ticket falls out of (b): **M=512 rounds into the "256" bucket and is served at
`BLOCK_M=16`, where `BLOCK_M=32` measures 1.10x on two independent routings.** That is the
width the scheduler's interleave share actually produces under load. §2.4.

---

## 1. (a) The fused k-planes — SHIPPED, on by default

### The mechanism
`deinterleave_a` (`ca7e74b`) rewrites A into an even-k plane followed by an odd-k plane so
the kernel's two per-byte A gathers become unit-stride. It is a standalone read+write of A.
For gemm1 that is unavoidable — its A is the layer's hidden states. For **gemm2** it is
redundant: gemm2's A *is* gemm1's output buffer (`ic1.view(-1, two_i)`; the ReLU² epilogue
means the intermediate is never re-materialized), so gemm1's own store can emit the planes.

The obvious implementation — permute the store — costs a scattered `tl.store`. This one
permutes the **load** instead. `PLANAR_OUT` makes tile column `d` take weight row

    perm(d) = (d % (N // 2)) * 2 + d // (N // 2)

so C's store stays a plain contiguous run and the permutation lands on the B/scale/global
gathers, whose `n` axis is the strided one either way (`stride_pn == K // 2` — every `n` is a
different weight row, so a stride-2 set of rows is addressed like a contiguous one). Every
output element is still the same k-ordered dot of the same operands; only *which tile column
it lands in* moves. Which is why it is an equality, and measures as one.

Guards, because the permutation is only valid where gemm2 reads gemm1's buffer verbatim:
`fuse_planes = bool(act) and NVFP4_PREFILL_DEINTERLEAVE_A and NVFP4_PREFILL_FUSED_PLANES`.
A gated activation (`ACT == 0`) runs `_run_act` over gemm1's output *row-wise* and would be
silently corrupted by a permuted column order; with the deinterleave off, gemm2 wants the
plain interleaved A. Both arms are pinned by the new test.

### Measured
`benchmarks/bench_moe_prefill_gemm.py --m 256 1024 2048 4096 8192 --variant tree deint fused
prepass prepass2 --grid shipped --verify`. `tree` is the production kernel with the
deinterleave off, `deint` today's default, `fused` this change; `prepass2` times *only* the
gemm2 rewrite that disappears.

| M | tree | deint | **fused** | fused/deint | fused/tree | `prepass2` | max abs diff |
|---:|---:|---:|---:|---:|---:|---:|---|
| 256 | 1.273 ms | 1.091 | **1.078** | 1.012x | 1.181x | 0.010 ms | 0.000e+00 |
| 1024 | 2.940 | 2.596 | **2.533** | 1.025x | 1.161x | 0.029 | 0.000e+00 |
| 2048 | 5.109 | 4.465 | **4.334** | 1.030x | 1.179x | 0.113 | 0.000e+00 |
| 4096 | 9.168 | 7.793 | **7.660** | 1.017x | 1.197x | 0.224 | 0.000e+00 |
| **8192** | **16.969** | **13.965** | **13.708** | **1.018x** | **1.237x** | **0.445** | **0.000e+00** |

M=8192 reproduces to 0.05 % in a second, independently launched run (13.708 / 13.954 /
16.956), so the 1.018x is not stream noise. `%tl.dot` at M=8192 goes 59.3 → **60.4 %**.

**Read the `prepass2` column as a bound, not as the win.** The removed rewrite is 0.445 ms at
M=8192 and the arm recovers 0.250 ms of it — 56 %. So the permuted B/scale row set is *not*
quite free after all: it costs ~0.2 ms of the 0.445 it saves, presumably in L2/TLB locality
over a 2x-wider `n` span. It is still a strict win at every M, and the alternative (permuting
the store) would have paid more.

### Default state
**On.** `FREETOKEN_NVFP4_PREFILL_FUSED_PLANES=0` is the hatch, and the flag is inert unless
the deinterleave is also on and the activation is epilogue-fused. Bit-exactness makes an
accuracy gate unnecessary — the same argument (and the same evidence) as `ca7e74b`.

Test: `tests/moe/test_nvfp4_backends.py::test_fused_k_planes_are_bit_identical_to_the_gemm2_
prepass` — `torch.equal` at m ∈ {1, 8, 17, 33} against the prepass arm, each arm also checked
against the dense dequant reference, plus a gated (`silu`) arm asserting the flag is inert.

---

## 2. (b) The M=256 bucket — the ticket's denominator was wrong, and the win is at M=512

### 2.1 The right ceiling
The ticket reads "M=256 runs at 20 % of ceiling with +53 % padding waste at `BLOCK_M=16`".
The padding number is exact (the harness prints **+53.1 %**, 1,536 routed rows → 2,352
padded). The ceiling is not the right one.

At M=256 there are 256·6 = 1,536 routed rows over 128 experts — **~12 rows per expert** — and
both GEMMs must still read every expert bank once:

| term | bytes |
|---|---:|
| gate_up packed `128 · 1856 · 1344` | 319.3 MB |
| gate_up scale `128 · 1856 · 168` | 39.9 MB |
| down packed `128 · 2688 · 928` | 319.3 MB |
| down scale `128 · 2688 · 116` | 39.9 MB |
| **total, once** | **718.5 MB (685 MiB)** |

That is **0.748 ms at the card's 960 GB/s**, against a measured 1.078 ms (`fused`): M=256 is
already at **69.4 % of the HBM roofline**. The `tl.dot` ceiling implies 0.26 ms, which is
unreachable — at 12 rows per expert this bucket is weight-streaming bound like decode. Real
headroom is **≤1.44x**, not ~5x, and no tile can go below 0.748 ms. The harness now prints
that floor and a `%HBM` column beside `%tl.dot`, plus padded rows *and M-block count* at
`BLOCK_M ∈ {16, 32, 64}`, so the bucket is scored against the bound that binds it.

### 2.2 `BLOCK_M=32` at M=256: a 12 % loss
`--m 256 512 --variant deint fused --grid-json '{"BLOCK_SIZE_M":[16,32]}' --verify` (every
cell bit-exact against the shipped reference):

| M | tile | padded rows | M-blocks | deint | fused |
|---:|---|---:|---:|---:|---:|
| 256 | `BLOCK_M=16` | 2,352 (+53.1 %) | 147 | **1.091 ms** | **1.079** |
| 256 | `BLOCK_M=32` | 4,096 (+166.7 %) | 128 | 1.225 | 1.218 |
| 512 | `BLOCK_M=16` | 4,064 (+32.3 %) | 254 | 1.447 | 1.429 |
| 512 | **`BLOCK_M=32`** | 4,288 (+39.6 %) | 134 | **1.312** | **1.296** |

At M=256 the block count only falls 147 → 128 (−13 %) while padding goes +53 % → +167 %, and
the result is a **12 % loss**. That settles the open question in the hypothesis: the
duplicate whole-expert reads `BLOCK_M=16` pays *are* absorbed by L2 (an expert slice is
~2.5 MiB), so the block count buys almost nothing at this M and only the padding is real.
This is also why the original tuner, whose grid contained `BLOCK_M ∈ {16,32,64,128}`, picked
16. **The shipped "256" bucket is unchanged**, and no side config directory is left behind.

### 2.3 The `BLOCK_M`/M-block relationship, for the record
The harness's new column, at the shipped seed:

| M | blocks @16 | @32 | @64 | padding @16 |
|---:|---:|---:|---:|---:|
| 256 | 147 | 128 | 128 | +53.1 % |
| 512 | 254 | 134 | 128 | +32.3 % |
| 1024 | 449 | 258 | 130 | +16.9 % |
| 2048 | 831 | 444 | 256 | +8.2 % |
| 4096 | 1,594 | 836 | 454 | +3.8 % |
| 8192 | 3,136 | 1,603 | 833 | +2.1 % |

Below ~128 blocks nothing more is available (one block per expert is the floor), which is
why `BLOCK_M=32` and `64` are identical at M=256.

### 2.4 New ticket — M=512 is served by the wrong bucket
At M=512 the same `BLOCK_M=16 → 32` change halves the blocks (254 → 134) for only
+32 % → +40 % padding and **wins 1.103x** (`fused` 1.429 → 1.296 ms; `deint` 1.447 → 1.312).
Repeated at `--seed 7` on a different routing: **1.106x** (`fused` 1.445 → 1.306, `deint`
1.465 → 1.322). Two independent routings, same answer.

But `PREFILL_M_BUCKETS` is `(16, 64, 256, 1024, 4096, 8192)` and `nvfp4_moe_config` picks the
*nearest* bucket, so M=512 (|512−256| = 256 < |512−1024| = 512) lands in the "256" bucket and
is served at `BLOCK_M=16`. Under multi-lane load the scheduler's interleave share produces
exactly this width, so this is not hypothetical — it is the case the ticket was pointing at,
in the neighbouring bucket. **Not changed here** (the brief was the 256 bucket, and a bucket
boundary is a shipped-table change that wants its own end-to-end evidence). The cheap next
step is a "512" bucket at `BLOCK_M=32`, graded by a soak rather than the microbench, since
the microbench cannot see the scheduler's real chunk-width distribution.

`BLOCK_KB` / `num_stages` at this bucket were **not** swept (the 216-tile `--grid smallm` run
was skipped). That is the other lever and it is still open.

---

## 3. (c) The 1,024-routed-id extend guard is conservative under LFU — a warmup job, not a fix

`use_cached_extend` refuses whenever `topk_ids.numel() > _MAX_ENSURE_QUERY (1024)`, because
flashlib's `lru_ensure` dedups its query with a `[BLOCK_K, BLOCK_K]` block at
`BLOCK_K = next_pow2(query.numel())` and Triton caps a tensor at 1,048,576 elements — so a
query wider than 1,024 ids cannot compile and used to kill the engine mid-forward
(`misc_tickets` §3). But the serving profile runs **LFU**, and under LFU `ensure_experts`
never reaches `lru_ensure` at all: `offload_kernels.py:50` routes `cache._size_class_enabled
or cache.cache_policy_id == 1` to the in-repo `_ensure_experts_sized_kernel`, whose only
compile-time widths are `BLOCK_E = next_pow2(num_experts)` and `BLOCK_C =
next_pow2(cache_size)` — both fixed by the model and the cache size, neither a function of the
query, which the kernel simply loops over. So the guard is a flashlib-LRU constraint applied
to a policy that does not have it, and the *same* pathology that makes raising it
unattractive is a one-off: the last width that does compile under LRU (a 768-id query,
`BLOCK_K = 1024`) cost **22 minutes of Triton JIT** on this card. If the threshold is ever
raised, that compile must be a **warmup job — a startup pass that materializes the kernel into
the Triton cache before the server takes traffic — never a live request**, since a 22-minute
stall inside a forward is indistinguishable from a hang to every client and to the scheduler's
finishability invariant. None of which is urgent: `misc_tickets` §3 measured the cached path
as a **loss** at every width above the m=64 crossover (1.25x GPU-time loss at m=128, fetching
107.8 of 128 expert rows per layer anyway), so widening the gate buys access to a slower path.
The correct shape of the change, if it is ever wanted, is to make `_MAX_ENSURE_QUERY` a
property of the *policy* (unbounded for `cache_policy_id == 1`, 1,024 for flashlib LRU) rather
than a global constant — and to have a measurement saying the cached path wins somewhere above
170 tokens *before* doing so. **No code changed for this item.**

---

## 4. Shipped

* `python/freetoken/kernel/triton/nvfp4_fused_moe.py` — `PLANAR_OUT` constexpr arm in
  `_prefill_nvfp4_moe_kernel` (source-row permutation on the B/scale/global gathers; the C
  store is untouched and stays contiguous, and the `PLANAR_OUT=False` path is unchanged).
* `python/freetoken/moe/fused_nvfp4.py` — `NVFP4_PREFILL_FUSED_PLANES` (**on by default**,
  `FREETOKEN_NVFP4_PREFILL_FUSED_PLANES=0` disables), `_prefill_gemm(a_is_planar=,
  planar_out=)`, the `fuse_planes` guard in `fused_experts_nvfp4`.
* `benchmarks/bench_moe_prefill_gemm.py` — `fused` and `prepass2` variants, `--grid smallm`
  (unrun), the per-`BLOCK_M` padded-rows/M-blocks report, the weight-byte floor line and the
  `%HBM` column.
* `tests/moe/test_nvfp4_backends.py` —
  `test_fused_k_planes_are_bit_identical_to_the_gemm2_prepass`.

Decode kernels, the fp8 sibling and the shipped tile tables are untouched.

## 5. Not done

* **End to end.** Every number here is the microbenchmark. The 131K prefill re-measurement
  that `ca7e74b` got (6,124.7 → 6,577.8 tok/s) has not been repeated for this change; at
  1.018x on a term that is ~14 % of a long prefill the expected end-to-end move is ~0.3 %,
  i.e. below the run-to-run spread of `bench_long_context.py`, so it needs a paired A/B or
  nothing.
* **The 216-tile `--grid smallm` sweep** (`BLOCK_KB` / `num_stages` at the small-M bucket).
* **The M=512 bucket** (§2.4).
