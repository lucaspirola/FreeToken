# K2-Horizon-MoVA-36B-A4B -> EXL3 4.0 bpw -> FreeToken

Queued by the owner on 2026-09-23 after the S12c and fastpack merges, before the GGUF Ornith GPU proof.

## Owner's words (quoted, not paraphrased)
- "hire a gpu in hugging face, quantize IFM/K2-Horizon-MoVA-36B-A4B  to EXL3 using their dataset, targeting 4-bits. download the converted EXL3 version only, kill the gpu server (of course! i am not rich), and add support for it in our freetoken."
- "if exl3 does not support it, that's one more thing for your bucket"
- Provider: "look for vast.ai or spheron.ai prices as well. But the job is the job: look and you'll find credentials for both of them. also, at this prices per hour, maybe a H100 might be better, evaluate."
- Calibration: the owner chose "exllamav3's built-in mix". The card's datasets, IFM/K2-Horizon-Pretrain-Data and -Midtrain-Data, return "Repository not found" even with the owner's token.
- Output: "push to my huggingface account, of course!"

## Facts (2026-09-23)
- Architecture `k2_horizon` (custom code, `modeling_k2_horizon.py`):
  - 48 layers; layers 0-2 are dense MLP (intermediate 6144).
  - MoE: 100 experts, top-8, plus 1 shared expert; moe_intermediate 768; sigmoid router with a selection-only bias; router_scaling 2.5; norm_topk.
  - Attention is MoVA (Mixture of Values): the V projection is replaced by 64 routed value experts per layer, top-4. Each is Linear(2560 -> 8 x 128), with SiLU applied to each expert's output before the weighted sum.
  - Attention also has a softplus(beta=ln 2) output gate (`gate_proj`).
  - GQA with 32 query heads and 8 KV heads of dim 128; RoPE theta 1e7 on the full head; no qk-norm.
  - RMSNorm is grouped (`layernorm_num_groups` = 2).
  - Vocabulary 250,624; untied embeddings; 512K context.
  - BF16 size is 69.8 GiB in 48 shards.
- exllamav3 v1.5.1 (2026-09-20) has no `k2_horizon` architecture, so we have to write it (MoVA is new).
- exllamav3 convert: `-b 4.0`; `-hq` recommended for MoE; `--cal_data` is available, but the built-in mix is chosen; the work dir needs room for a full copy of the output.
- FreeToken has no EXL3 kernels today and no k2_horizon model.

## Provider prices (single GPU, looked up 2026-09-23)
| provider | GPU | $/h | note |
|---|---|---|---|
| HF Jobs | RTX PRO 6000 96 GB | 2.75 | the `hf` CLI is already authenticated (user pirola); pushes straight to HF |
| HF Jobs | A100 80 GB | 2.50 | |
| HF Jobs | H200 141 GB | 5.00 | |
| vast.ai (verified, on-demand) | RTX PRO 6000 S 96 GB | 1.49 | 43 Gbps down |
| vast.ai | H100 SXM 80 GB | 2.27 | Czechia; next offer 5.47 |
| vast.ai | RTX 5090 32 GB | 0.54 | |
| Spheron (list price) | H100 80 GB | 2.64 | |
| Spheron | RTX PRO 6000 96 GB | 2.35 | |
| Spheron | A100 80 GB | 1.48 | |

Credentials: none found on disk for vast.ai or Spheron. The only candidate is the owner's Bitwarden vault, which is locked.

## exllamav3 support (2026-09-23, /home/lucas/ai/tools/exllamav3 branch k2-horizon, commit 4160552, local only)
- `architecture/k2_horizon.py` and `modules/arch_specific/k2_horizon.py` (MoVAValues: 64 value experts, each its own quantizable EXL3 linear; router in fp32 with a selection-only bias; softplus(beta=ln 2) gate; grouped RMSNorm; dots-style MoE router with a selection-only bias). Python only; no kernel change.
- Unquantized vs the HF fp32 reference, layers 0-4 (262 tokens): cos min >= 0.999999 on tokens without a routing flip; logits cos 0.999996, top-1 100%.
- 4.0 bpw -hq, 5-layer model, 100 calibration rows, fp16 head: logits cos min 0.999627, top-1 100%. Value-expert routing: 90% exact / 97.5% overlap. Converter errors (rfn) 0.004-0.010.
- Untested locally: the 6-bit lm_head step with default reference states (needs 14-16 GB host RAM), and layers 5-47.
- Estimates (from a measured 77 s per MoE layer on the 5080 at 250 rows): 5080 ~70 min, RTX PRO 6000 ~30-35 min, RTX 5090 ~40 min, H100 SXM ~55-75 min. The H100 is the worst value because the trellis quantization is compute-bound on SMs. Host needs >= 64 GB RAM and ~130 GB disk.
- Decision [agent]: HF Jobs RTX PRO 6000 ($2.75/h, 256 GB RAM, 475 GB disk). Reasons: the `hf` CLI is authenticated; no vast/Spheron credentials were found; the output lands in the owner's HF account directly. Hard cap: `--timeout 3h` = $8.25 max; expected ~1-1.5 h including the 70 GB download = ~$3-4.
- Output repo: pirola/K2-Horizon-MoVA-36B-A4B-exl3-4.0bpw-hq, created PRIVATE (the owner said "push to my huggingface account"; visibility was not stated, so private is the default [agent]). The recipe is in its `conversion/` folder (patch + run.sh).
- Job 6ab3773d52d0dbd7f1d82f2e launched 2026-09-23 (`hf jobs logs 6ab3773d52d0dbd7f1d82f2e`).
- Owner asked 2026-09-23: "why did you cape at 3 hours? we can lose it all by doing that (server gets killed, nothing gets saved) right?" Yes. A timeout kill loses /work (the input, the exllamav3 checkpoints and the output), and the script's ERR trap does not run on a kill. The cap exists to bound the bill if the job hangs; a running job's timeout cannot be changed. Measured on the job: the 70 GB download took 35 s, and the converter printed "Estimated remaining time: 16 minutes" after 4 of 48 layers (06:56 UTC). So 3 h is about 4x the expected time. For any future paid run [agent practice]: sync the converter's work dir to the HF repo periodically so a killed job can resume with `-r`, and set the timeout from a measured ETA with a wide margin.
- Owner sanity check 2026-09-23: "k2 horizon dont even run in llama.cpp stock, did you check their model page?" I had not checked llama.cpp; I had only checked exllamav3. IFM/K2-Horizon-MoVA-36B-A4B-GGUF says the files "require a version of `llama.cpp` containing K2 Horizon architecture support. PR to llama.cpp is in progress", and names the fork MBZUAI-IFM/llama.cpp branch model/K2Horizon. Supported engines: vLLM and SGLang (IFM recipes) plus that fork.
  - The card says router numerics matter: SGLang needs the `xllm_source_router_gemm_partitions` override, "which preserves the checkpoint's router numerics". So the router stays fp32, and routing is compared against the reference on all 48 layers.
  - The fork + the official GGUF are a second independent reference for checking FreeToken's output.
  - Tool calls: the `k2_horizon` parser (default format xml; json and xml_typed via chat_template_kwargs).
