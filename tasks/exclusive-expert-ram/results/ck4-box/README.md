# ck4 on the rented box ft-ck (RTX 5080, native Linux), commit 9dac3b5

Run 2026-09-24 02:57Z to 04:34Z by `checkpoint-box.sh ck4`: four arms, in order ck4-whole,
ck4-mirror-1m, ck4-mirror and ck4-whole-close. Settings: port 1920, memory ratio 1.00, q8q8 KV,
/root/venv (torch 2.11+cu130). The box runs a Threadripper PRO 3945WX (Zen 2) with 251 GiB RAM,
driver 610.43.02, and its GPU sits on a **PCIe gen 4 x16** link. The owner's RTX 5080 has a gen 5
x16 link and a Core Ultra 7 265K. `ck4-run.log` is the full run log.
`ck4-needles-compare.txt` and `ck4-records-compare.txt` are the script comparisons, and
`ck4-pcie-bw.txt` is the copy-bandwidth probe (`../../pcie_bw.py`).

## Verdict

**The correctness and RAM gates pass. The decode gate fails as measured on this box.** The
failure follows the box's PCIe gen 4 link, which gives half the owner's bandwidth. It does not
point to a code regression. This box cannot certify decode for ck4. That needs a gen 5 host,
meaning the owner's machine or a rented box with a gen 5 x16 link.

| Gate (owner's) | Result | |
|---|---|---|
| 1M arm RAM 12.26 ± 0.6 GiB | ck4-mirror-1m 12.06 GiB (the 713K arm is 12.80, and the gate applies to the 1M arm only) | PASS |
| coverage faults 0 | 0 / 0 (mirror-1m / mirror) | PASS |
| starved write-backs 0 | 0 / 0 | PASS |
| one graph capture | captures=1 on all 4 arms, 0 tracebacks, 0 OOMs (R3 PASS on all 4) | PASS |
| needles/recall identical to ck4-whole | 14 needle answers + 3 recall sizes (21K/120K/240K): 0 differences | PASS |
| decode within 9% of ck4-whole, pass by pass | 1 of 4 comparable points is inside (table below) | FAIL (environment, see below) |
| R6 memlock | RLIMIT_MEMLOCK 8192 KiB inside the Vast container on all 4 arms | environment limit of Vast, not judged |

Box stability: ck4-whole-close matches ck4-whole within 0.1% at every point (8K 173.2/173.4,
32K 168.5/167.7, 80K 156.5/161.1 for p1/p2). The box did not drift during the run.

### Decode against the box's own ck4-whole (tok/s)

| point | ck4-whole | ck4-mirror | ratio | ck4-mirror-1m | ratio |
|---|---|---|---|---|---|
| 8K p1 | 173.3 | 131.5 | 75.9% | 131.4 | 75.8% |
| 8K p2 | 173.5 | 146.8 | 84.6% | 147.7 | 85.1% |
| 80K p1 | 156.6 | 119.6 | 76.4% | | |
| 80K p2 | 161.2 | 147.2 | 91.3% ok | | |
| 713K p1 / p2 | none (whole arm stops at 80K) | 56.6 / 77.7 | | | |
| 1M p1 / p2 | none | | | 47.5 / 61.4 | |

The whole arm has no 713K or 1M point, so the long-context numbers can only be compared with
the owner's record: 1M 47.5/61.4 against 76.9/72.9 (61.8%/84.2%), and 713K 56.6/77.7 against
91.7/107.6. The owner's machine gives the same pool-vs-whole reference (ck3, commit 5f0a213):
mirror/whole at 8K was 97.4%/92.9% and at 80K p2 98.3%, all inside the gate. The pool costs
3–7% there and 15–24% here.

## Cause of the decode gap: PCIe gen 4

Measured on the box (`ck4-pcie-bw.txt`): pinned H2D 28.3 GB/s and D2H 28.2 GB/s (256 MiB). The
owner's gen 5 link carries the pool gather at 52.9 GB/s (plan notes). One expert row is 5.36 MiB
(8.88 GiB / 1696 rows). A row therefore costs **0.199 ms here and 0.106 ms on the owner's
machine, a ratio of 1.88.**

The pool arm moves experts over PCIe (miss admissions H2D, write-backs D2H). If that traffic is
what the arm pays for, then the extra ms/token each host pays over its own whole-model decode,
divided by that host's per-row time, should give the same experts per token on both hosts.
The owner's side uses ck3 (whole and mirror), and the long points use the 8K whole decode as
the base.

| point | extra ms/token box | extra ms/token owner | box/owner | experts/token box | experts/token owner |
|---|---|---|---|---|---|
| 1M p1 | 15.28 | 7.84 | 1.95 | 76.7 | 73.7 |
| 1M p2 | 10.52 | 5.86 | 1.79 | 52.8 | 55.2 |
| 713K p2 | 7.11 | 3.57 | 1.99 | 35.6 | 33.6 |
| 8K p2 | 1.05 | 0.40 | 2.63 | 5.3 | 3.8 |

At long context the box pays 1.8–2.0 times the owner's extra time, which tracks the 1.88
bandwidth ratio. The implied miss traffic agrees within 5%. A CPU-bound gap would scale with
the single-thread gap between Zen 2 and the 265K. That gap is about 1.3–1.5×, which is what the
whole arm shows: 173 vs 190 tok/s at 8K, a 9% gap before any pool traffic. The stats back this
up. The pool behaved as well as on the owner's machine or better: free-eviction rate 0.646 vs
0.617, swaps 80009 vs 88441, write-backs 29071 vs 34408, 0 starved, 0 faults. The same
admissions simply take twice as long on the wire. (Excluded as known outliers: the owner's ck3
713K p1 at 34.1, which carries the delivery-stall signature, and ck3-whole 80K p1 at 44.6.)

Not fully explained: the first pass after start (8K p1 and 80K p1) runs about 24% below whole,
where the per-token model predicts about 10%. The journal shows the pool restoring coverage and
the arena resizing (2144 -> 2096 -> 2136 slots) around those requests. Extra PCIe work during
the first requests is the likely reason, but this run cannot confirm it because dmon logged no
PCIe counters. The second passes, which are the steady state, fit the bandwidth model.

## What would settle decode

Run the mirror arms on a gen 5 x16 host. On the owner's machine that means `checkpoint1.sh ck4`
with the production server stopped by the owner. A rented box would need
`nvidia-smi --query-gpu=pcie.link.gen.max` = 5 checked before renting.
