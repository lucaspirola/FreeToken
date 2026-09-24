# Ornith EXL3 + RAM saver with DMA writebacks (scratch/exl3-dma d5b57a3), ft-dev

Box numbers: ft-dev (RTX 5080, PCIe gen 4, EPYC 7402P), Ornith-1.5-35B-A3B EXL3 5.0bpw hq, saver
(auto pool), ratio 1.00, q8_0 KV, `/root/step5.sh` settings, `probe_decode.py` 2 passes.
`scratch/exl3-dma` = exp/ornith-exl3 (edbbc62) + the reorg-headroom picks (as scratch/exl3-headroom)
+ the DMA-writeback commit from exp/mirror-dma-wb (773d9f8). Not merged anywhere.

Same code, same session, ring on vs off (`FREETOKEN_MIRROR_WB_STAGE_MB=0`, the SM-store path):

| decode tok/s | 8K p1 | 32K p1 | 80K p1 | 128K p1 | 8K p2 | 32K p2 | 80K p2 | 128K p2 |
|---|---|---|---|---|---|---|---|---|
| SM stores (dmaoff) | 40.5 | 58.8 | 55.8 | 58.7 | 119.2 | 116.2 | 76.6 | 81.6 |
| **DMA writebacks (dma)** | **63.6** | **96.0** | **91.5** | **88.8** | 118.1 | 115.2 | **102.4** | **99.0** |

The control matches the earlier v3 run (`s5-v3`, 40.5/58.7/55.8/58.7 and 118.9/115.8/76.6/81.5).
Prefill is unchanged. Pass 2 at 8K/32K was never writeback-bound (the working set fits).

* 32-row ring (60.5 MiB, the row cap at 1.98 MB rows). 21,749 of 35,866 writebacks staged
  (61%); the rest are ring-full fallbacks and prefill-buffer writebacks. A larger ring cap is the
  obvious next lever for this model; not tried.
* Coverage faults 0, starved 0. Both chats at temperature 0 are identical between dma and dmaoff.
* GPU tests on this branch (`gpu-tests.txt`): 368 passed (mirror suites incl. the DMA byte-exact
  tests, and tests/kernels/test_exl3.py).
