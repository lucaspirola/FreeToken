# Mirror reserve minimum on ft-dev (box)

exp/mirror-reserve-min f16672a:
* `tests/moe/test_mirror_reserve_env.py` + `tests/moe/test_mirror_reserve_min.py`: 18 passed.
* Live: Ornith-1.5-35B-A3B-NVFP4 (E=256), `FREETOKEN_MIRROR_RESERVE_ROWS=256`, measure.sh arm
  rmin256. The server now refuses at startup (`refusal.txt`) instead of dying in the warmup
  prefill with "bounded expert mirror lost coverage before prefill" (exp/lfu-halve
  results/lfuab-box, lfuo* arms). 512 and 768 rows served in the same A/B.
