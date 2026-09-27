"""Host (CPU) time of one fused MoE prefill call vs its device time, per decode group size: is the
call launch-bound at small token counts?"""
import time, torch
import freetoken.moe.fused_exl3 as fe
from freetoken.models.exl3_banks import exl3_bank_shapes
DEV = torch.device("cuda")
H, I, E, TOPK, BITS = 2048, 512, 256, 8, 5
torch.manual_seed(0)
B = []
for name, (shape, dtype) in exl3_bank_shapes(H, I, BITS).items():
    if dtype == torch.int16:
        B.append(torch.randint(-(1 << 15), 1 << 15, (E, *shape), dtype=torch.int32, device=DEV).to(torch.int16))
    else:
        B.append((torch.randn((E, *shape), device=DEV) * (0.5 if name.endswith("suh") else 0.05)).half())
B = tuple(B)
for M in (512, 8192):
    x = torch.randn(M, H, dtype=torch.bfloat16, device=DEV)
    w = torch.softmax(torch.randn(M, TOPK, device=DEV), -1)
    ids = torch.stack([torch.randperm(E, device=DEV)[:TOPK] for _ in range(M)]).to(torch.int32)
    for g in (32, 16, 8):
        fe.PREFILL_DECODE_GROUP = g
        run = lambda: fe.fused_experts_exl3(x, B, w, ids, bits=BITS, codebook="mul1", is_prefill=True, num_experts=E)
        run(); torch.cuda.synchronize()
        # host cost: launch 5 calls back to back behind a long GPU sleep so the queue never drains
        torch.cuda._sleep(int(2e9)); t0 = time.perf_counter()
        for _ in range(5):
            run()
        host = (time.perf_counter() - t0) / 5 * 1000
        torch.cuda.synchronize()
        print(f"M={M} group {g}: host {host:.3f} ms per call", flush=True)
