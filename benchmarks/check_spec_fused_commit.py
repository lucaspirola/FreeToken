"""Weightless check: the fused (all-layers-on-the-head-axis) speculative commit == the
per-layer one, and what it costs.

No weights, ~40 s of GPU. Synthetic mixers and a synthetic LinearStatePool at the
Nemotron-3.5-Lightning geometry; compares SpecScanCapture._commit_fused against
._commit_per_layer at several (m, n) and times both.

    PYTHONPATH=python scripts/gpu_lock.sh .venv/bin/python -u \
      benchmarks/check_spec_fused_commit.py
"""
import torch

from freetoken.models.nemotron_h.spec_scan import SpecScanCapture, _plan

dev = torch.device("cuda")
L, SLOTS, H, P, N, G = 23, 6, 64, 64, 128, 8
CONV_DIM, KM1 = 4096 + 2 * G * N, 3
torch.manual_seed(0)


class Pool:
    track_chunk_size = 128

    def __init__(self):
        self.recurrent_states = torch.randn(L, SLOTS, H, P, N, device=dev) * 0.1
        self.conv_states = (torch.randn(L, SLOTS, CONV_DIM, KM1, device=dev) * 0.1).to(torch.bfloat16)

    def local_index(self, layer_id):
        return layer_id

    def copy_from(self, src, dst):
        self.conv_states[:, dst].copy_(self.conv_states[:, src])
        self.recurrent_states[:, dst].copy_(self.recurrent_states[:, src])


class Mixer:
    num_heads, head_dim, state_size, n_groups = H, P, N, G
    dt_limit = (0.0, float("inf"))

    def __init__(self, layer_id):
        self.layer_id = layer_id
        self.A = -torch.exp(torch.randn(H, device=dev)).float()
        self.D = torch.randn(H, device=dev).float()
        self.dt_bias = torch.randn(H, device=dev).float()


pool = Pool()
MIXERS = [Mixer(li) for li in range(L)]
for m in (9, 17):
    for n in (1, 2, 5, m - 1):
        cap = SpecScanCapture(m)
        for li in range(L):
            cap.record(
                MIXERS[li],
                torch.randn(m, H, P, device=dev, dtype=torch.bfloat16),
                torch.randn(m, H, device=dev, dtype=torch.bfloat16),
                torch.randn(m, G, N, device=dev, dtype=torch.bfloat16),
                torch.randn(m, G, N, device=dev, dtype=torch.bfloat16),
                torch.randn(m, CONV_DIM, device=dev, dtype=torch.bfloat16),
            )
        live, a, b = 1, 2, 3
        pool.copy_from(live, a)
        pool.copy_from(live, b)
        plan = _plan(pool, cap.layers, a)
        assert plan is not None
        cap._commit_fused(pool, plan, n)
        cap._commit_per_layer(pool, b, n)
        rec = (pool.recurrent_states[:, a] - pool.recurrent_states[:, b]).abs().max().item()
        conv = (pool.conv_states[:, a].float() - pool.conv_states[:, b].float()).abs().max().item()
        ref = pool.recurrent_states[:, b].abs().max().item()
        print(f"m={m} n={n}: rec |d|max={rec:.3e} (scale {ref:.3e})  conv |d|max={conv:.3e}", flush=True)
        assert conv == 0.0, "conv window must be bit-exact"
        assert rec <= 1e-4 * max(ref, 1e-3), "fused scan disagrees with the per-layer scan"

# launch count, eager
import time
cap = SpecScanCapture(9)
for li in range(L):
    cap.record(MIXERS[li],
               torch.randn(9, H, P, device=dev, dtype=torch.bfloat16),
               torch.randn(9, H, device=dev, dtype=torch.bfloat16),
               torch.randn(9, G, N, device=dev, dtype=torch.bfloat16),
               torch.randn(9, G, N, device=dev, dtype=torch.bfloat16),
               torch.randn(9, CONV_DIM, device=dev, dtype=torch.bfloat16))
plan = _plan(pool, cap.layers, 2)
for name, fn in (("fused", lambda: cap._commit_fused(pool, plan, 5)),
                 ("per-layer", lambda: cap._commit_per_layer(pool, 3, 5))):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        fn()
    host = (time.perf_counter() - t0) / 20 * 1e3
    torch.cuda.synchronize()
    total = (time.perf_counter() - t0) / 20 * 1e3
    print(f"{name}: host {host:.3f} ms/commit, wall {total:.3f} ms/commit", flush=True)
print("FUSED_COMMIT_OK")
