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
