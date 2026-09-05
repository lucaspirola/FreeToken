# Ornith RTX 2000 Ada elastic multi-agent validation

Date: 2026-08-31  
Host: NVIDIA RTX 2000 Ada Generation (sm_89), 16 GiB, 70 W, WSL 2  
Host RAM: 103 GiB; WSL CUDA pin budget approximately 41 GiB

The RTX 5080 fork added exact elastic CUDA graphs, GDN-state sizing, growable
shared KV accounting, coalesced teardown, and cold-session infrastructure. These
control paths are architecture-neutral, but were live-gated on Ada with both
production model/KV pairs before acceptance.

Both runs used four independent 4,096-token prompts, 256 forced output tokens,
an initial capacity of two requests, a four-request ceiling, 4K prefill chunks,
a 262,144-token shared virtual KV ceiling, and LFU expert caching. Q4/INT4 used
64K physical KV growth; Q6/Q8 used 128K. Every agent recovered only its own
passcode with no foreign prompt data.

| Model / KV | Expansion | GDN slots | MoE slots | Simultaneous decode | Wall | Result |
|---|---|---|---|---:|---:|---|
| Q4_K_M / INT4 | 2 → 4 → 2 | 13 → 25 → 13 | 6,232 → 5,685 → 6,232 | 92.40 tok/s | 25.06 s | PASS |
| Q6_K / Q8_0 | 2 → 4 → 2 | 13 → 25 → 13 | 4,177 → 3,773 → 4,177 | 53.91 tok/s | 31.92 s | PASS |

Exact three-agent graph capture, live-demand sizing, and the two-second
intermediate-shrink grace are shared machinery and remain unchanged. These gates
exercise the endpoints—two and four—while their unit coverage verifies the
intermediate demand tiers and coalescing rules.

## Pageable expert applicability

A live Q6/Q8 startup with `--moe-pageable-gpu` reported that every expert bank
fits the CUDA pin budget and retained the direct pinned-RAM-to-GPU route for all
40 layers. Its 4K retrieval gate remained coherent, but no pageable staging was
activated. Therefore the RTX 5080's measured low-miss pageable-layer order is
intentionally not translated or enabled on this high-RAM host: doing so would
insert an unnecessary pageable-to-pinned copy without freeing a constrained
resource.

The session spill/restore, explicit session-close barrier, VMM cache compaction,
and pageable staging implementations remain available unchanged. They are
covered by exact tensor/state and scheduler tests; pageable staging activates
only if a future host's post-reserve RAM or WSL pin quota actually requires it.
