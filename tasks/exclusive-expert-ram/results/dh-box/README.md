# Decode-level headroom probe, ft-dev (Linux, Nemotron NVFP4; box numbers)

Branch exp/decode-headroom a2f08c0 with `FREETOKEN_DECODE_MEM_PROBE=1`: each decode window
(release of prefill headroom -> next prefill reserve or KV grow) logs driver free minimum,
allocator reserved rise and allocated-peak rise; each KV commit logs its overhead.

* mirror-np (8K, 80K): every decode window free drop 0.00 GiB, reserved rise 0, alloc peak rise 0.
* mirror-1m: 67 windows 0.00 GiB; 2 windows at 1M context (window closed by the KV shrink after
  agent teardown) drop 0.02 GiB, reserved rise 0.02, alloc peak 0.01. 45 KV commits, overhead 0.00 GiB.
* Decode with the probe on: 1M 68.5/69.9 tok/s, 8K 161.8/162.1 tok/s (p1/p2).

Reading: on Linux decode itself needs <= ~20 MiB beyond what prefill release leaves; the 0.25 GiB
VMM cushion is a WSL cuMemSetAccess mapping need that the commit path already funds. The
default-vs-128 MiB decode target A/B (dh2.sh) was stopped before it ran (wb-schedule took
priority); the WSL-side cushion probe belongs in the local queue.
