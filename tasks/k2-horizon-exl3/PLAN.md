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

## FreeToken design (Opus specialist, 2026-09-23, read-only; [V] = verified, [I] = inferred)
- **Kernels: copy exllamav3's EXL3 CUDA sources unmodified into `kernel/csrc/exl3/`** and build them on first use, as the GGUF kernels are. Build only the K in {4, 6} with the mul1 codebook, plus fork-only bindings.
  - Rejected: importing exllamav3_ext. FreeToken's venv is torch 2.11.0+cu130, pinned by sglang-kernel [V]; the locally built ext targets torch 2.14/cu132 [V].
  - Dequant-to-bf16 is used only as a test oracle.
- **Experts bank into fixed-stride rows [V]:** each expert carries its own trellis/suh/svh.
  - MoE row 2,969,088 B (gate_up 1,979,392 + down 989,696): 45 x 100 experts = 12.44 GiB.
  - Value-expert row 1,317,888 B: 45 x 64 = 3.53 GiB.
  - exllamav3's `exl3_moe` (prefill) and `exl3_moe_coop` (decode) take per-expert pointer tables. The arena's virtual addresses are fixed, so the slot-pointer tables are static and graph-safe.
- **Context on the 16 GB card [I]:** q8_0 KV costs 104,448 B/token [V].
  - MoVA experts on the GPU (option A): ~32K context with ~1,260 extra MoE slots.
  - MoVA on the arena (option B, engine change): ~64K at q8_0, ~128K with q4_0 KV.
  - **256K and above do not fit** (25.5 GiB at q8_0). The model's native context is 512K. This limit is for the owner to know.
- **Router: numerics-sensitive.**
  - The EXL3 file stores the MoE router weight as fp16 (converted from bf16, lossy for weights below 6.1e-5, max diff 3e-8); the bias stays fp32. `v_router` stays bf16.
  - Plan: fp32 compute. Study the effect by comparing fp32, a single bf16 GEMM, and the 2-partition bf16 sum (the likely meaning of `xllm_source_router_gemm_partitions`; unconfirmed until SGLang's model code is read), using the original BF16 router weights fetched by range reads (~45 MB).
- **Phases:**
  - P0 layout check (check_experts EXL3). Run it on the real repo first: the bit allocation may vary per layer, and if so the rows fall into size classes (the S12b mechanism).
  - P1 bf16 model `models/k2_horizon/*` vs HF fp32 on the 5-layer checkpoint.
  - P2 EXL3 dense linears vs exllamav3 goldens.
  - P3 EXL3 experts on the arena (slot-shuffle test, graph capture, eviction churn).
  - P4 full serve, with growable KV, parsers, acceptance vs goldens and the llama.cpp fork.
  - P5 optional: MoVA on the arena, cached-extend for EXL3, CPU decode.
- **Serving:**
  - Tool-call parser `k2_horizon` (xml/json/xml_typed; `<ifm|...>` tags; adapted from Glm47Detector).
  - Force-open reasoning parser for `<ifm|think*>`.
  - The template raises if an earlier assistant turn has no thinking field.
- **Risks:**
  - fp16 activations in deep layers (EXL3 kernels take fp16; FreeToken's residual is bf16).
  - Autotune and lazy allocation must be warmed before graph capture.
  - One shared barrier buffer means one compute stream.
  - The prefill full-layer stream moves 13.4 GB per forward.
- Owner 2026-09-23: "validation run on the rtx pro 6000 approved" (the ~$1.50-2 run: BF16 original vs EXL3 on all 48 layers, plus golden reference outputs saved to the repo).
- Owner 2026-09-23: "why don't collect the needle questions at higher tokens? because of our own local limitation ?" -> yes, I had sized them to the local ~32K. Now changed: BF16 to 64K (bf16 KV 196 KB/token; 128K+ does not fit in 96 GB); EXL3 with an FP16 cache to 256K; exllamav3 quantized-cache arms K8V8 / K6V5 / K4V4 to 256K, scored against the FP16 arm. The extended run's cost (~$3-5 estimated) is above the approved ~$1.50-2 and is being put to the owner.
- **Owner challenge 2026-09-23**: "I challenge you to create something that will make it fast in our rtx5080, with 256k kv 8-bits, or at least 6-bits k and 5-bits V (usually quality is still maintained, right?)".
  - Facts: KV is 25.5 GiB at q8_0 and ~18 GiB at K6/V5 for 256K. That is more than the 16 GB card, and competes in host RAM with 12-16 GiB of expert banks. So resident 256K KV is impossible; it needs tiered KV (VRAM/RAM/NVMe) and sparse retrieval-style decode attention.
  - Design by an Opus specialist in progress, with a physics budget and accuracy plan; it fits G2 (the smart KV manager).
- Owner 2026-09-23: "approved the longer validation" (~$3-5; exact estimate to be reported once the script is timed).
- **Conversion DONE 2026-09-23.** Job 6ab3773d52d0dbd7f1d82f2e: COMPLETED, running 2,843 s (47.4 min), ~$2.17 at $2.75/h. The job ended by itself; `hf jobs ps` shows no running jobs.
  - Final bitrate 4.11 bpw excluding the head, with `-hq`.
  - Repo pirola/K2-Horizon-MoVA-36B-A4B-exl3-4.0bpw-hq (private): 20 files, 19.12 GiB, 3 safetensors shards (7.87 / 7.68 / 3.52 GiB), plus quantization_config.json, the tokenizer, the modeling code and `conversion/{patch, run.sh, convert.log}`.
  - Local download is deferred until the control measurement ends, to keep the host quiet.
- **Published 2026-09-23** at the owner's request ("make ... public, link it as a quantization of the original model"). The repo is public; the Hub shows `base_model:quantized:IFM/K2-Horizon-MoVA-36B-A4B`.
  - Our own model card replaced IFM's README, which the converter had copied in. It says: community conversion, not affiliated with IFM; stock exllamav3 cannot load it; the patch is in `conversion/`; full-model validation is pending.
  - The public patch has its Claude-Session line removed, but the first upload (commit 3bb057f) still carries it in the repo history.
- **Prior art the owner found: vcruz305/K2-Horizon-MoVA-36B-A4B-EXL3** (created 2026-09-22):
  - Packs 2.0 / 2.5 / 4.0 / 5.0 / 6.5 / 8.0 bpw, made with "SAGE" (their own mixed-precision method) on exllamav3 1.5.0, with calibration 64 x 1024 (ours: 250 x 2048).
  - Their scores vs the BF16 original running on IFM's code with fp32 activations, over 10,240 held-out positions (top-1 / mean KLD):
    - BF16 in exllamav3: 97.71% / 0.0030
    - 8.0 bpw: 96.43% / 0.0047
    - 6.5 bpw: 90.68% / 0.0335
    - 5.0 bpw: 85.83% / 0.0923
    - 4.0 bpw: 84.81% / 0.1082
    - 2.5 bpw: 83.71% / 0.1326 (13.27 GB, fits 16 GB)
    - 2.0 bpw: 81.66% / 0.1604
  - So this model loses a lot between 8 and 5 bpw and then flattens; that is consistent with the router's sensitivity.
  - **Their port is unpublished** ("will be linked here when they are published"), so ours is currently the only loadable public port.
  - Their metrics are directly comparable to what our validation run computes (top-1 and KL vs BF16 on held-out text). Add the BF16-in-exllamav3 ceiling to our run if it is cheap.
- Owner 2026-09-23: "clear the claude session likn in the repo". Done: `super_squash_history`, 5 commits -> 1 (a9bd0ca). The current files carry no claude.ai link (verified by download + grep).
- Owner 2026-09-23 on the vcruz305 numbers: "agreed with the 2.5bpw, but my reading is that we might just drop it: a 15% loss defeats the purpose, doesn't it?"
  - My reading [agent]: top-1 disagreement is not a 15% accuracy loss, but a mean KLD of 0.108 at 4 bpw is high for EXL3.
  - Their curve is flat from 5.0 to 2.5 bpw (85.8 -> 83.7%) after a cliff from 8.0 (96.4%). That points to a structural error floor (routing flips, fp16 router weights or activations, MoVA), not weight precision.
  - Decide after OUR validation run, whose per-layer routing agreement and KL can locate the floor. If it is the router, keeping routers/value experts in higher precision may remove it.
  - Owner's call pending: continue after the validation, or drop.
- Owner 2026-09-23: "most likely vcruz305 gave us a warning of where to look and maybe dodge a problem. are we ready for that? let's beat this guy? you tell me."
  - **Target:** beat vcruz305 4.0 bpw (84.81% top-1, mean KLD 0.108, p99 0.909 vs BF16 with fp32 activations, held-out text) at <= ~20 GB. Stretch: approach their 8 bpw (96.4%) at ~4.5 bpw.
  - **Plan** [agent]:
    1. The validation run, now with a per-layer error-growth and routing diagnostic (which layer and component makes the error).
    2. Fix the router storage in our patch: the MoE router is stored fp16 today; store it bf16/fp32 as in the checkpoint.
    3. exllamav3's own per-tensor sensitivity pipeline (`doc/optimize.md`): sc_trace (self-sampled in-domain trace) -> sc_measure (noise injection into the UNQUANTIZED model, KLD) -> sc_optimize (greedy per-tensor bit allocation) -> `convert.py -rcp recipe.yaml [-cd cal_trace.safetensors]`. Upstream warns it is "untested on sparse models", and MoVA is new to it.
    4. Evaluate on wikitext (comparable with vcruz305) AND on a disjoint self-sampled trace (the deployment metric; optimize.md argues raw web text misjudges reasoning models).
  - **Readiness:** the patch, the conversion pipeline (47 min, ~$2.2) and the validation script exist. Unknowns: sc_* on MoE + MoVA; whether the floor is intrinsic to the model.
  - **Extra spend estimate** beyond the approved validation: ~$10-15 (sensitivity measurement on BF16 ~1-2 h + one recipe conversion + one validation). Awaiting the owner's approval.
- Owner 2026-09-23: "you have my go to beat vcruz205" -> the beat plan is approved at the ~$10-15 estimate given (cap $15 [agent: my stated figure]).

## Validation run result (job 6ab389d351992417dfcd650b, rtx-pro-6000, 31m26s, ~$1.4; COMPLETED, nothing left running)

Outputs: `validation/summary.{md,json}` in pirola/K2-Horizon-MoVA-36B-A4B-exl3-4.0bpw-hq.

- **Headline, our 4.0 bpw -hq (4.11 bpw excluding the head, 19.12 GiB)**, measured against the BF16 reference with fp32 activations (vcruz305's method), wikitext-2 test, 8 x 2048 = 16,384 positions:
  - top-1 **93.90%**, mean KL **0.0170**, p99 KL 0.176. PPL 7.200 vs BF16 7.153 (+0.7%).
  - vcruz305 4.0 bpw: 84.81% / 0.108 / 0.909. Ours also beats their 6.5 bpw (90.68% / 0.0335); their 8.0 bpw is 96.43% / 0.0047.
  - Caveats [agent]: their held-out text is unnamed and has 10,240 positions (ours: wikitext-2 test, 16,384), and our KL is truncated to the reference's top-64 (a slight underestimate; top-1 is exact). Only a run of both packs through the same script is apples to apples.
  - Their "structural floor" is **not in our port**. Hypothesis: it is a port error in theirs (their unquantized ceiling is already 97.71%). Ours was not measured: the ceiling arm was skipped because VRAM was still held.
- **Per-prompt top-1 vs BF16:**
  - math 95–98%, code 93–94%, multilingual 93%, thinking 88–93%, reasoning 84–88%.
  - **Tool-call XML is the weak spot: 75.7%, KL 0.43, p99 3.08.**
- **Per layer:**
  - Teacher-forced relative error is 0.6–1.7% in every layer, with no bad layer.
  - The accumulated error grows smoothly to 11% by layers 42–45, then recovers to 9.9% at layer 47.
  - Routing exact-match of the whole top-k set: MoE 55–76%, MoVA 70–93%. The mismatch margins are ~1e-3, i.e. near-tie flips, not a broken router.
  - BF16 router GEMM flips vs fp32: 0/512 rows in every layer, so router storage precision is not the problem.
- **What failed (the harness, not the model):**
  1. The needle scores are invalid. The thinking preamble used up `max_new_tokens`, so the code was cut off (e.g. "I find: \"The secret verification code for site Bravo"). All "plant=False" results come from truncation.
  2. KV-quant arms: k8v8 reload OOMed because the fp16 arm had not been freed; k6v5 and k4v4 hit an AssertionError on reload. No KV-quant numbers.
  3. The BF16 needles OOMed at 64K (88,743 tokens) on the transformers path.
  4. The BF16-in-exllamav3 ceiling arm was skipped (0.3 GB free).
- **Consequence for the beat plan [agent]:** the target is already beaten at the same bpw.
  - The router-precision fix is moot (0 flips).
  - The remaining lever is tool-call quality: the sc_* recipe with a self-sampled trace that contains tool calls.
  - The fixed long-context/KV arms are still needed for the 256K design.
  - Proposed to the owner before spending more.
- Owner 2026-09-23: "local overlay is off, but wait for my go. approved both fixed rerun of needle and a new conversion tuned for tool calls."
  - Guard3 (S12c) stays held until the owner's go.
  - Approved spend [agent estimates given]: the fixed needle/KV rerun (~$1.5) and the tool-call-tuned conversion (~$8-10).

## Rerun result (job 6ab3969351992417dfcd68cd, rtx-pro-6000, 23m08s, ~$1.06; COMPLETED)

Outputs: `validation/rerun/summary.{md,json}`. Fixes vs the first run: cache detach between arms, 512 tokens at low reasoning effort, haystack sizes calibrated to real token counts.

- **Needles on our 4.0 bpw, 8K-256K real tokens (8,075 / 31,893 / 64,115 / 128,349 / 255,229): the Bravo code is correct in every KV arm at every size (fp16, k8v8, k6v5, k4v4).**
  - Distractor (site Delta, never planted): correct at every size.
    - The scorer marked 256K false only because the reasoning quoted the planted Alpha code while concluding "I was not given it." The model's answer is correct.
  - Answer text identical to fp16:
    - k8v8 and k6v5 from 32K up;
    - k4v4 from 32K to 128K.
    - At 8K every quantized arm skipped fp16's short reasoning (same answer).
    - At 256K k4v4's text diverges (same answer).
  - First-answer-token KL vs fp16:
    - k8v8 1e-5–0.14;
    - k6v5 3e-4–0.42;
    - k4v4 0.02–0.95.
  - For the 256K design (KV256K.md): K6/V5 keeps the answers, and K8/V8 is near-exact.
- **Ceiling, BF16 weights run through exllamav3** (same wikitext 16,384 positions, fp32-activation reference): **97.36% top-1, KL 0.0049**, PPL 7.157. vcruz305's ceiling: 97.71% / 0.0030.
  - The two ports' numerics agree, so the gap at 4.0 bpw (ours 93.90% vs their 84.81%) comes from the quantization, not from the port.
  - Our 4.0 bpw keeps 96.4% of the ceiling's top-1.
