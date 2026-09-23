# EXL3 weights in FreeToken: design note

Branch `exp/ornith-exl3` (worktree `~/ai/FreeToken-wt/exl3`), forked from `exp/reorg` at `eb60d81`.
Target: serve `~/ai/models/Ornith-1.5-35B-A3B-exl3-5.0bpw-hq` (exllamav3 1.5.1, "5.08 bpw", `-hq`)
on the RTX 5080 with the RAM saver (`--expert-residency mirror`) on.

Everything under "Verified" was checked against the downloaded files or against exllamav3
itself on 2026-09-24, not recalled. Scripts: `tasks/ornith-exl3/probes/`.

## 1. The EXL3 tensor layout

### Verified on this checkpoint

`quantization_config.json` lists every stored tensor (`tensor_storage`); all 124,156 entries
match the safetensors headers of the three shards in dtype, shape and byte count (0 mismatches).

One quantized linear `X` (in_features `K`, out_features `N`) is four tensors:

| tensor | dtype | shape | meaning |
|---|---|---|---|
| `X.trellis` | int16 | `[K/16, N/16, 16*bits]` | one 16x16 weight tile per `[k_tile, n_tile]`, `256*bits` bits |
| `X.suh` | fp16 | `[K]` | input-side signs x channel scales (applied before the input Hadamard) |
| `X.svh` | fp16 | `[N]` | output-side signs x scales (applied after the output Hadamard) |
| `X.mul1` | int32 | `[]` | codebook flag: present = "mul1" codebook; its value is the multiplier `0x83DCD12D` (= `mul1_multiplier` 2212286765 in the config) |

`X.mcg` (int32) would flag the "mcg" codebook, and neither flag means the original 3INST
codebook; this checkpoint only uses mul1 (`"codebook": "mul1"`). Older exports store packed
sign bitfields `su`/`sv` instead of `suh`/`svh`; none here, and the loader refuses them.

Per-tensor bits (`bits_per_weight` in `tensor_storage`, and `trellis.shape[-1] / 16`):

| module | count | bits | trellis |
|---|---|---|---|
| routed experts `layers.N.mlp.experts.E.{gate,up}_proj` | 2 x 10,240 | 5 | `[128, 32, 80]` |
| routed experts `...down_proj` | 10,240 | 5 | `[32, 128, 80]` |
| `shared_expert.{gate,up,down}_proj` | 3 x 40 | 7 | |
| `linear_attn.{in_proj_qkv,in_proj_z,out_proj}` | 3 x 30 | 7 | qkv `[128, 512, 112]` |
| `self_attn.{q,k,v,o}_proj` | 4 x 10 | 7 | q `[128, 512, 112]` (q carries the output gate) |
| `lm_head` | 1 | 6 | `[128, 15520, 96]` |

Stored unquantized: `embed_tokens` bf16 `[248320, 2048]`; all RMSNorm weights bf16; the routers
`mlp.gate` fp16 `[256, 2048]` and `shared_expert_gate` fp16 `[1, 2048]`; `linear_attn.in_proj_a/b`
fp16 `[32, 2048]`; `A_log`, `dt_bias`, `conv1d`. The index also carries an MTP head (`mtp.*`, 4 bits)
that FreeToken does not serve (the Ornith reader already drops `mtp.`), and **no vision tensors**,
so this checkpoint serves text only (`--text-model-only` / no encoders).

Every bit width is an integer (no `16*bits + 8` half-integer tiles), and all routed experts in all
40 layers are 5 bits, so one bank row size serves every layer.

### Tile decode (verified bit-exact against exllamav3's `reconstruct`)

`probes/perm_probe.py` decodes random trellis tiles in NumPy and compares with
`exllamav3_ext.reconstruct` for bits 2..8 and all three codebooks: **bit-exact in all 21 cases**.
The rules it encodes:

* View the tile's `16*bits` int16 as `8*bits` little-endian uint32 words; the bit stream is
  MSB-first over those words (stream bit `j` = bit `31 - j%32` of word `j/32`).
* Weight number `t` (0..255, stream order) is the 16-bit window of stream bits
  `[t*bits + bits - 16, t*bits + bits)` taken cyclically modulo `256*bits` (the first windows wrap
  to the end of the tile).
* mul1 codebook: `x = w * 0x83DCD12D mod 2^32`; `s = sum of x's four bytes`;
  `value = fp16( (1024 + s) * h(0x1eee) + h(0xc931) )` with one rounding (hfma). In fp32 the FMA is
  exact for every `s`, so an fp32 FMA then one fp16 rounding reproduces it.
  mcg: `x = w * 0xCBAC1FED`; 3INST: `x = w * 89226354 + 64248484`; both then
  `x = (x & 0x8fff8fff) ^ 0x3b603b60` and `value = lo_half(x) + hi_half(x)` in fp16.
* Stream position `t` sits at tile row `r` and column `c` with
  `t = 32*(c & 7) + 16*((r>>2)&1) + 8*((r>>1)&1) + 4*(c>>3) + 2*(r>>3) + (r&1)`
  (the tensor-core fragment order; derived by matching and then checked on every tile).
* The decoded tile `W_hat[k_tile*16 + r, n_tile*16 + c]` is in the *rotated* basis.

### The linear forward

`H` is the 128x128 Sylvester Hadamard matrix (checked equal to exllamav3's `get_hadamard(128)`),
scaled by `1/sqrt(128)`, applied blockwise over every 128 contiguous channels:

    y = H_blocks( H_blocks(x * suh) @ W_hat ) * svh

exactly `LinearEXL3.get_weight_tensor()`: `W = diag(suh) . H . W_hat . H . diag(svh)`. EXL3 needs
`K % 128 == 0` and `N % 128 == 0`, true for every tensor here.

## 2. Kernel strategy: FreeToken-native Triton kernels, exllamav3 as the oracle

Chosen: write EXL3 kernels in Triton inside FreeToken (`kernel/triton/exl3.py`), with a NumPy/torch
decoder as the reference, and test them against exllamav3 outputs. Not chosen for now: vendoring
exllamav3's CUDA `exl3_gemm`/`exl3_mgemm`/`exl3_moe` and compiling them with
`torch.utils.cpp_extension.load` the way `kernel/gguf.py` vendors llama.cpp's.

Why:

* **G1, self-containment.** exllamav3's kernels are not a leaf: `exl3_gemm` pulls in its device
  context (`DevCtx`, a global lock buffer), a cooperative-launch autotuner with process-global state,
  its own CUDA-graph recorder (`Graph*` parameter patching), and 118 template compilation units
  (the local build is a 475 MB `.so`). Vendoring a subset means maintaining a fork of a fast-moving
  tree (the local checkout already carries K2-Horizon patches); the GGUF precedent vendored a
  small, stable, torch-op-only subset, which this is not.
* **Bank addressing.** FreeToken's expert kernels read `[slots, row]` banks with `topk_ids` already
  rewritten to slot ids, inside CUDA graphs. exllamav3's MoE entry points take pointer tables built
  per call from its own module objects and cooperative grids sized by its autotuner. A Triton kernel
  takes the bank tensor and the slot id directly, the same contract `fused_nvfp4`/`fused_gguf` use.
* **Graph safety.** Triton launches are plain launches with fixed shapes; nothing to patch at replay.
* **Correctness first.** The decoder is 30 lines and proven bit-exact; the kernels are checked
  against it and against exllamav3 layer outputs.

Cost, stated plainly: the first Triton kernels will be slower than exllamav3's hand-tuned CUDA,
especially the M=1 decode GEMV (exllamav3's QTIP-style GEMV plus autotuning). The follow-up, if
the measured decode rate matters, is to vendor exllamav3's `exl3_gemv` for dense decode only,
behind the same `LinearKernel` interface (`--quant-backend linear.exl3=...`), which the kernel
table already supports; nothing else changes.

Kernels (all fp32 accumulation):

1. `had_rows`: `xh[p] = H_blocks(x[src[p]] * suh_of(p))` for rows `p`, with per-row suh (a routed
   pair's expert) or one suh (dense). `tl.dot` against the +-1 matrix, exact in fp16, scaled after.
2. `exl3_gemm`: `y[p, n] = svh_of(p)[n] * H_blocks( xh[p] @ W_hat_of(p) )[n]`, the decode of each
   `[BK, 128]` weight block done in registers inside the K loop, the output Hadamard fused into the
   epilogue (BN = 128 = one Hadamard block, so no fp32 temporary). Rows are grouped per expert with
   `moe_align_block_size` for prefill; a dense linear is the one-expert case.
3. Decode: the same kernel with BM=16; a split-K GEMV is the first optimization if needed.

## 3. Experts as bank rows for the mirror pool

The MoE kernel (`layers/quantization/moe/exl3.py`) declares six banks, all `raw_row=True`
(the checkpoint's own bytes, no repack), per expert:

| bank | dtype | per-expert shape | bytes (Ornith) | source |
|---|---|---|---|---|
| `gate_up_trellis` | int16 | `[2, H/16, I/16, 16*bits]` | 1,310,720 | gate trellis, then up trellis |
| `gate_up_suh` | fp16 | `[2, H]` | 8,192 | gate suh, up suh |
| `gate_up_svh` | fp16 | `[2, I]` | 2,048 | gate svh, up svh |
| `down_trellis` | int16 | `[I/16, H/16, 16*bits]` | 655,360 | down trellis |
| `down_suh` | fp16 | `[I]` | 1,024 | down suh |
| `down_svh` | fp16 | `[H]` | 4,096 | down svh |

Row = 1,981,440 bytes; 10,240 rows = 18.90 GiB for the whole model (so the whole-model residency
does not fit this 28 GiB host with the rest of the process; the saver is required here).

Verified on the shards: **every expert's 12 tensors are one contiguous extent in one shard**, in
the same order for all 10,240 experts (down suh, svh, mul1, trellis; gate ...; up ...), 1,981,452
bytes including the three 4-byte `mul1` scalars. So the pool's source reads one region per expert
(the pool's aligned O_DIRECT reader already coalesces a record's pieces into regions) and scatters
nine pieces into the six banks; the `mul1` scalars are checked once at scan time and not stored.

Plumbing, generic rather than Ornith-specific:

* `models/exl3_banks.py`: `Exl3ExpertSpec` (the model's key template, e.g.
  `model.language_model.layers.{layer}.mlp.experts.{expert}.{proj}.{kind}`), the expert piece reader
  for `build_expert_banks` (whole-model residency) and the mirror `source`
  (`quant_format`, `shapes`, `records`, `shard_fds`, `fd_size`: the same protocol S12c gave GGUF).
  Both produce identical row bytes by construction (same offsets table), and a CPU test asserts
  byte equality between them and the raw file bytes.
* A model opts in by exporting `exl3_expert_spec(model_path, config)`; Ornith does.
* The residency builder's format gate learns one more sourced format through a small hook
  (`moe/mirror_sources.py`), keeping the NVFP4 branch untouched.

The bank shapes need the expert bit width before any bank exists; it comes from the installed
`Exl3Config` (read from `quantization_config.json`), which refuses a checkpoint whose routed experts
do not all share one bit width (one pointer/stride per bank, like the mixed-GGUF refusal).

## 4. Attention and the other dense tensors

* A new quant dialect `layers/quantization/configs/exl3.py` claims `quant_method == "exl3"`, reads
  the per-tensor storage from `quantization_config.json` next to `config.json`, and gives each
  module the scheme `EXL3(bits, codebook)`. The QuantKind enum gains `EXL3` (one line).
* `layers/quantization/linear/exl3.py` declares `trellis`/`suh`/`svh` and runs the kernels above.
* **Fused modules.** FreeToken fuses `q/k/v -> qkv_proj`, `in_proj_qkv/z -> in_proj_qkvz`,
  `shared_expert.gate/up -> gate_up_proj`. EXL3 parts have different `suh` (input scales differ per
  weight), so the fused layer keeps `trellis` concatenated along the N tiles, `svh` concatenated,
  and `suh` stacked `[parts, K]`; every part boundary is a multiple of 128 here, so each 128-column
  output block belongs to one part and the kernel picks that part's rotated input. Parts with
  different bit widths are refused (not present in this checkpoint). The dense reader asks the
  dialect how to combine parts (`QuantConfig.combine_parts`, default: the old `cat(dim=0)`), which
  is the one hook added to the shared reader path.
* `in_proj_ba` (fp16 `in_proj_b`/`in_proj_a`) and the routers stay unquantized; the loader casts
  them to bf16 as it does for any stored dtype.
* `lm_head` (6 bits) is an EXL3 linear; in prefill it only sees the last token of each request.
* `embed_tokens` is bf16 and unchanged.

## 5. Test plan

CPU (server venv, no model loaded):

1. Decoder: the NumPy reference vs exllamav3 reconstruct outputs stored as a small fixture
   (`tests/fixtures/exl3/`), bits 2..8 x 3 codebooks.
2. Bank rows byte-exact: the piece reader's packed row and the mirror source's row equal the raw
   file bytes of that expert, on a synthetic EXL3 checkpoint (and, in a probe, on 4 real experts).
3. Config/dialect: schemes per module, fused-part combination, refusal of mixed bits.

GPU (under the host lock, no server):

4. Triton decode vs NumPy decode, bit-exact.
5. Each layer kind vs exllamav3 on real tensors: dense linear (7-bit qkv, 6-bit lm_head slice),
   fused qkv, MoE layer (routed + shared) on real layer-0 experts; max abs error and relative error.
6. Whole-model logits on a short prompt: FreeToken (transient unit on :1921, RAM saver on) vs
   exllamav3's own model forward; max abs error and top-1 agreement per position.

## 6. Edits to shared files (G1 budget)

New modules carry the format. Shared files get only: the `EXL3` enum member and scheme builder,
three package imports, one `LEGACY_FORMAT` entry, one `_BANK_SCHEMAS` entry and its byte formula,
the `combine_parts` hook (default behaviour unchanged), the sourced-format hook in the residency
builder, and `"exl3"` next to `"gguf"` in the arena/mirror gates. Each is listed in its commit.
