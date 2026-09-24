# nsys, 8K decode, whole vs mirror, exp/dt-dma 583afc8, owner's RTX 5080 (WSL)

Produced by `nsys-local.sh` on 2026-09-24 from `dt-measure` at 583afc8. Each arm got three
warm-up requests, then one traced request: 8K prompt, 512 decode tokens, passes 1 and 2.
`steps-graph-level.txt` comes from `ck4dma-g5/ck4dma-nsys/nsys_steps.py` and covers pass 1;
the `.sqlite` exports were not committed because they can be regenerated from the `.nsys-rep`.

| pass 1, per decode step | whole | mirror |
|---|---|---|
| period, median | 5510.8 us (180.5 tok/s) | 5253.1 us (188.2 tok/s) |
| graph GPU time | 4896.3 us | 4786.8 us |
| GPU busy / idle | 4930.1 / 608.7 us | 4819.2 / 495.5 us |
| D2H memcpy | 1.0 per step, 0.8 us | 13.1 per step, 231.1 us, 9.47 MB (write-backs on side streams) |
| cudaMemcpyAsync calls | 20.0 per step | 32.1 per step |

On 583afc8 the pool's write-back D2H (9.5 MB per step) runs on side streams and overlaps the
decode graph, so the mirror step is not longer than the whole step here.

On ba7d1a6 the gap was +0.5 ms per token (`recheck-local`). The g5 trace attributed it to
write-backs between graphs and to memcpy nodes inside the graph. mirror-dma-wb's VRAM staging
ring removes that cost.

Probe decode under the tracer, pass 1 / pass 2:
* whole: 181.6 / 170.6 tok/s;
* mirror: 184.3 / 185.4 tok/s.
