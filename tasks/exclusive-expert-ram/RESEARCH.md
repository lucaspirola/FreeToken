# Prior art: expert prefetch and cache policy, ranked for FreeToken (2026-09-24)

This is a bounded literature pass (~30 min) that decides what the next two levers (layer-ahead
prefetch, eviction beyond LFU) should be.

## Our design constraints

Every idea below is judged against these:

- **Single lane.** Decode runs one token per step.
- **Memory layout.** The whole model sits in pinned RAM. A unified GPU slot cache (LFU) holds the
  resident experts, and a mirror pool in host RAM holds the complement.
- **One graph.** The decode step is one CUDA graph, captured once.
- **Byte equality.** Outputs must equal the whole-model run.
- **Generic.** The mechanism must work for every MoE model, with no per-model training.

## Where the time goes

Box numbers: ft-g5 RTX 5080, node-level nsys, Nemotron-3.5-Lightning, 8K decode.

- The in-graph miss fetch (`fast_index_copy_kinds`, SM-driven zero-copy) costs 631 us of a
  5593 us graph on average. The median step spends 276-295 us; the 90th percentile spends 1.26 ms.
- The fetch is serial: it overlaps nothing.
- Every Nemotron MoE block is preceded by a Mamba or attention block. Summing min(fetch, preceding
  block) over the layers gives **326 us/step (5.8% of the graph)**. That is the most that
  one-block-ahead prefetch can hide with a perfect predictor (`results/gap-g5/hide-bound.txt`).
- "Speculating Experts" derives the same bound: ΔT = Σ min(t_copy, t_compute).

## Ranked for our design

### 1. Cross-layer gate prefetch, exact mode (adopt)

**How it works.** Run MoE layer j's own norm and router on the residual one block early. Fetch the
predicted misses on a graph branch while the intervening block computes. The real
`ensure_experts` then fetches whatever the prediction missed.

**Prior art:**
- Fate reuses the next layer's gate on the previous gate input and reports 97% decode prefetch
  accuracy. That figure needs a confidence-expanded set; the plain top-k is ~76-79% on the
  shallow layers (layers 0-3).
- AdapMoE uses the next layer's gate: ~90% on most layers, lower on the early ones.
- DALI and HOBBIT prefetch the same way.
- "Speculating Experts" predicts at d=1 with ~90% hits on GPT-OSS. On Qwen3-30B-A3B the early
  layers reach only 60-70% recall@k.
- ProMoE measured this skip/gate reuse at only 66.9% on Qwen2 and chose a learned MLP (84.7%).

**Why it fits us:**
- It is generic: every MoE layer has a router, so a model only exposes `route_ids(norm(x))`.
- It needs no training.

**Exact, not approximate.** "Speculating Experts" executes the *predicted* experts, which cost
GSM8K 0.950 → 0.576 on Qwen3-30B-A3B. That breaks byte equality, so we keep the ProMoE and
ST-MoE rule instead: a misprediction is fetched on demand, so the outputs never change.

**What to watch:**
- *Wasted fetches.* A wrong prediction still transfers bytes and evicts a resident expert.
  ProMoE uses low-priority speculative tasks that are cancelled once the gate is known. Our
  graph cannot cancel, so we measure the waste (`wasted` in the probe).
- *SM contention.* Our fetch is an SM copy kernel, so on a parallel graph branch it shares SMs
  with the block it overlaps.
- *Bookkeeping order.* The mirror's per-step snapshot (`prev_slot_of_id`, `begin_layer`) must be
  taken by the first admission of the layer in the step, i.e. by the prefetch.

**Measurement.** exp/layer-ahead 5c2377c adds `FREETOKEN_ROUTE_PROBE`, which reports recall, miss
coverage and wasted fetches at d=1 and d=2 for Nemotron before anything is built. The expected
gain is miss_coverage × 326 us, minus the cost of the wasted fetches.

### 2. Per-layer slot budgets from predictability (adopt with item 1)

**Prior art:**
- AdapMoE allocates cache per layer with a DP/knapsack over each layer's sensitivity and
  prefetch accuracy. Early layers, which are harder to prefetch, get more slots; 1.36× end to
  end together with gating and prefetch.
- Fate's "shallow-favoring" cache holds all shallow-layer experts, for a 99.08% hit rate.

**Why it fits us.** Our unified LFU has no per-layer budget. Once prefetch exists, a layer whose
misses are covered by the prediction needs fewer resident slots, and a layer the prediction
cannot cover needs more.

**Inputs.** Generic: the per-layer probe coverage plus the per-layer miss counters that already
exist (`stat_missing_layer`). Per-layer routing entropy (the owner's idea) is the
prediction-free fallback.

### 3. Eviction beyond LFU: decay / LFRU (test cheaply, expect little)

**Prior art:**
- "When Does Trace-Driven Evaluation Mislead MoE Expert Caching?" (128 experts, 40% residency):
  the best causal policy is LFRU (frequency/age) at a 18.01% miss fraction, vs 19.37% for a
  static set.
- 84-97% of the gap to Belady is knowledge of which resident expert is used furthest in the
  future, and a trained next-use predictor did *worse* than LRU.
- The same paper shows that trace replay inflates recency policies by 27-29%, so it must be
  measured live, which our arms do.
- A Markov-chain policy beats LRU by 2.3-3.8 pp (Zenodo artifact).
- MoE-Infinity and EdgeMoE use LFU; SeqMoE's probabilistic Belady needs a sequence-model predictor.

**Expectation for us.** At our 94.7% hit rate, decaying LFU is a cheap A/B (one counter-halving
kernel every N steps, inside the graph) with a small expected gain.

### 4. Compute the misses on the CPU instead of fetching them (known, not first)

**Prior art:**
- Fiddler: for small batches the CPU beats the weight transfer (1.26× single batch).
- ktransformers and llama.cpp / ik_llama keep routed experts on the CPU (`-ot exps=CPU`).
  ktransformers keeps the GPU part in one CUDA graph with `cudaLaunchHostFunc` host nodes.

**Where we stand.** FreeToken already has `decode_target=cpu|hybrid` with host nodes. For us it
could serve the tail steps (1+ ms of fetch), but it changes kernels and reduction order, so byte
equality would need checking.

### 5. Rejected: they change outputs or need per-model training

- HOBBIT loads low-precision copies of the less critical missed experts.
- AdapMoE's adaptive gating activates 25% fewer experts.
- Pre-gated MoE retrains the gate to select the next layer's experts.
- ProMoE (learned MLP), ExpertFlow (T5-style routing-path predictor) and SeqMoE (sequence model,
  96.97% hit at 45% residency) all need predictors trained per model. They are worth revisiting
  only if the cross-layer gate's recall proves too low.
- SeqMoE's "graph-compatible, synchronization-free" runtime is the closest system to ours. Only
  the abstract was read.

## Sources

- Pre-gated MoE — https://arxiv.org/pdf/2308.12066
- Fate (cross-layer gate) — https://arxiv.org/html/2502.12224v1
- AdapMoE — https://arxiv.org/html/2408.10284 , https://github.com/PKU-SEC-Lab/AdapMoE
- ProMoE — https://arxiv.org/html/2410.22134v2
- HOBBIT — https://arxiv.org/html/2411.01433v1
- MoE-Infinity — https://arxiv.org/html/2401.14361v2
- Fiddler — https://arxiv.org/html/2402.07033v2
- ExpertFlow — https://arxiv.org/pdf/2410.17954
- Speculating Experts — https://arxiv.org/html/2603.19289
- ST-MoE spatio-temporal prefetch — https://arxiv.org/html/2606.15453v1
- DALI — https://arxiv.org/pdf/2602.03495
- SeqMoE — https://arxiv.org/abs/2609.12978
- Trace-driven caching evaluation — https://arxiv.org/html/2608.07911v1
- Markov-guided expert cache replacement — https://zenodo.org/records/22711487
- ktransformers — https://github.com/kvcache-ai/ktransformers , llama.cpp discussion https://github.com/ggml-org/llama.cpp/discussions/8721
- llama.cpp MoE offload guide — https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide
