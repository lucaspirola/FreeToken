"""fast_index_copy_multi on the EXL3 expert banks: pinned host -> GPU (admission) and GPU -> host
(writeback) of n experts, per blocks_per_bank. Row sizes are Ornith 5.0bpw's six banks."""
import torch
import triton.testing as tt

from freetoken.kernel.fast_index_copy import fast_index_copy_multi_jit
from freetoken.kernel.pinned import device_ptr

ROWS = {"gu_tr": 2048 * 1024 * 5 // 8, "gu_suh": 2 * 2048 * 2, "gu_svh": 1024 * 2,
        "dn_tr": 512 * 2048 * 5 // 8, "dn_suh": 512 * 2, "dn_svh": 2048 * 2}
E, S = 64, 32
host = {k: torch.empty((E, v), dtype=torch.uint8).pin_memory() for k, v in ROWS.items()}
gpu = {k: torch.empty((S, v), dtype=torch.uint8, device="cuda") for k, v in ROWS.items()}
hp = torch.tensor([device_ptr(host[k]) for k in ROWS], dtype=torch.int64, device="cuda")
gp = torch.tensor([gpu[k].data_ptr() for k in ROWS], dtype=torch.int64, device="cuda")
fb = torch.tensor(list(ROWS.values()), dtype=torch.int64, device="cuda")
per_expert = sum(ROWS.values())
print(f"expert bytes {per_expert / 1e6:.2f} MB over {len(ROWS)} banks")
for n in (0, 1, 2, 4, 8):
    di = torch.arange(n, dtype=torch.int32, device="cuda")
    si = torch.arange(3, 3 + n, dtype=torch.int32, device="cuda")
    cnt = torch.tensor([n], dtype=torch.int64, device="cuda")
    # the mirror launches with a fixed-length index list and a device count
    di_full = torch.zeros(8, dtype=torch.int32, device="cuda"); di_full[:n] = di
    si_full = torch.zeros(8, dtype=torch.int32, device="cuda"); si_full[:n] = si
    line = []
    for bpb in (8, 16, 32, 64, 128):
        h2d = tt.do_bench(lambda: fast_index_copy_multi_jit(gp, hp, fb, di_full, si_full, cnt, blocks_per_bank=bpb), rep=100) * 1e3
        d2h = tt.do_bench(lambda: fast_index_copy_multi_jit(hp, gp, fb, si_full, di_full, cnt, blocks_per_bank=bpb), rep=100) * 1e3
        gbs = lambda t: n * per_expert / t / 1e3 if n else 0
        line.append(f"bpb{bpb}: H2D {h2d:6.1f}us ({gbs(h2d):4.1f} GB/s) D2H {d2h:6.1f}us ({gbs(d2h):4.1f})")
    print(f"n={n}: " + " | ".join(line))
