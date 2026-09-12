# GPU expert-cache tuning outcome

Preferred reboot profile: [resume-1m-static-4096-cuda-telemetry-mr091.sh](resume-1m-static-4096-cuda-telemetry-mr091.sh).
Fallback: [resume-1m-static-4096-cuda-telemetry.sh](resume-1m-static-4096-cuda-telemetry.sh),
which retains memory ratio 0.85. The generic 4096 launcher remains unchanged.

The preferred `.91` profile keeps one lane, 1,048,576-token static q8_0 KV/context,
4096-token prefill, six linear-state slots, Triton attention/Mamba-2 decode, MoE
offload LFU, 1/50/50 GiB session-spill RAM/disk/total limits, 8 GiB host reserve, and
`--cuda-memory-telemetry`. Its only tuning difference from the fallback is
`--memory-ratio 0.91` rather than `0.85`.

Bounded evidence is in
`/home/lucas/ai/hidden-state-routing/analysis/experiments/gpu-expert-tuning/`:
`baseline-085-v1`, the matching `.91` and `.9175` four-request trials, and
`long-context-091-v1`. The `.91` matched trial observed 677 MiB sampled CUDA free
margin; the one long-prefill request used 128,026 input tokens and observed about 751
MiB sampled free margin. These are sampled bounds, not an allocator or driver guarantee.
The work does not qualify sustained decode or full-1M context operation.

Both scripts are self-contained launchers: they contain their complete non-secret argv
and set the PATH/CUDA architecture environment used by `systemd-run`. They refuse an
active FreeToken unit or occupied port, verify the project, uv executable, model, and
spill paths, require 28 GiB `MemAvailable` and 60 GiB spill free after the previous model
has been stopped, and start the transient `freetoken-serve.service`. No other launcher is
required. They intentionally do not stop an existing service themselves.
