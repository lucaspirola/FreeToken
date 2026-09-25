# Prefill chunk A/B on exp/reorg-next (next-measure f352594), 2026-09-25

`chunk-local.sh`, RTX 5080 (WSL), ratio 1.00, 0 MiB on the GPU and MemAvailable >= 23 GiB before
every arm, one arm at a time on :1920. Prompts 8K/32K/80K/256K, two passes (pass 2 of record).
Arms: `--max-prefill-length` 8192 / 12288 / 16384 x {Nemotron 3.5 Lightning, Ornith 1.5} x
{whole, saver}. Table: `chunks-compare.txt` (`compare_chunks.py`).

Result: no chunk size above 8192 earns a rule.
* Nemotron: prefill is flat within noise at 32K-256K (98-103%), except one whole-16384 p2 dip
  (84%/94% at 32K/80K); decode is 0-6% lower at 12288/16384.
* Ornith: 16384 prefills 2-3.5% faster at 32K-256K but decodes 3-7% slower (the arena holds
  ~560 fewer expert slots during prefill and the transient grows 1.02 -> 1.95 GiB).
* The measured transient scales with the chunk (Nemotron 0.59/0.88/1.18 GiB saver, Ornith
  1.02/1.53/1.99 GiB); no OOM line in any arm.
The default stays 8192.
