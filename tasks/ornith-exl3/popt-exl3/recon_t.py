"""Prototype: reconstruct_experts in bitstream order (codes t = 32 cl + i of each 16x16 tile, word
offsets and shifts functions of i only), permuted to (r, c) in registers. Bitwise vs production
reconstruct_experts, and timing, at Ornith gate_up / down shapes (16 experts per launch)."""
import torch, triton, triton.language as tl
import freetoken.kernel.triton.exl3 as k3
import freetoken.moe.fused_exl3 as fe
from freetoken.models.exl3_banks import exl3_bank_shapes


@triton.jit
def _recon_t_kernel(tr_ptr, tr_expert_stride, out_ptr, out_expert_stride, e_lo,
                    part_of_nb_ptr, word_off_ptr, ntiles_ptr, nstart_ptr,
                    BITS: tl.constexpr, CB: tl.constexpr, KT: tl.constexpr):
    pid_k = tl.program_id(0)
    pid_n = tl.program_id(1)
    g = tl.program_id(2).to(tl.int64)
    part = tl.load(part_of_nb_ptr + pid_n)
    n_tiles = tl.load(ntiles_ptr + part)
    nt0 = (pid_n * 128 - tl.load(nstart_ptr + part)) // 16
    WORDS: tl.constexpr = 8 * BITS
    OFF0: tl.constexpr = 257 * BITS - 16
    i = tl.arange(0, 32)
    b0 = OFF0 % 32 + i * BITS
    j0 = b0 // 32
    j1 = (b0 + 15) // 32
    sh = ((j1 + 1) * 32 - (b0 + 16)).to(tl.uint32)
    cl = tl.arange(0, 8)
    wa = (BITS * cl[:, None] + OFF0 // 32 + j0[None, :]) % WORDS
    wb = (BITS * cl[:, None] + OFF0 // 32 + j1[None, :]) % WORDS
    nt = tl.arange(0, 8)
    n = tl.num_programs(1) * 128
    for kt in tl.static_range(KT):
        k_tile = pid_k * KT + kt
        base = tr_ptr + (e_lo + g) * tr_expert_stride + tl.load(word_off_ptr + part) + (k_tile * n_tiles + nt0 + nt).to(tl.int64) * WORDS
        a = tl.load(base[:, None, None] + wa[None, :, :]).to(tl.uint32, bitcast=True)
        b = tl.load(base[:, None, None] + wb[None, :, :]).to(tl.uint32, bitcast=True)
        v = k3._decode_words(a, b, sh[None, None, :], CB)  # [nt, cl, i]
        v = tl.reshape(v, (8, 8, 2, 2, 2, 2, 2))  # nt, cl, i4, i3, i2, i1, i0
        v = tl.permute(v, (5, 2, 3, 6, 0, 4, 1))  # r = 8 i1 + 4 i4 + 2 i3 + i0; c = 16 nt + 8 i2 + cl
        w = tl.reshape(v, (16, 128))
        rr = k_tile * 16 + tl.arange(0, 16)
        tl.store(out_ptr + g * out_expert_stride + rr[:, None].to(tl.int64) * n + (pid_n * 128 + tl.arange(0, 128))[None, :], w)


def recon_t(bank, parts, lo, hi, out, KT=1, nw=4):
    _recon_t_kernel[(parts.k // (16 * KT), parts.n // 128, hi - lo)](
        k3._words(bank), bank.stride(0) // 2, out, out.stride(0), lo,
        parts.part_of_nb, parts.word_off, parts.ntiles, parts.nstart,
        BITS=parts.bits, CB=k3.CODEBOOKS[parts.codebook], KT=KT, num_warps=nw)
    return out


def med(fn, reps=20):
    fn(); torch.cuda.synchronize(); ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); fn(); b.record(); b.synchronize(); ts.append(a.elapsed_time(b))
    return sorted(ts)[reps // 2]


DEV = torch.device("cuda")
H, I, E = 2048, 512, 256
for BITS in (2, 3, 4, 5, 6, 8):
    for cb in k3.CODEBOOKS:
        if BITS != 5 and cb != "mul1":
            pass
        torch.manual_seed(BITS)
        sh = exl3_bank_shapes(H, I, BITS)
        gu, dn = fe.expert_parts(H, I, BITS, cb, DEV)
        for nm, parts, key in (("gu", gu, "gate_up_trellis"), ("dn", dn, "down_trellis")):
            G = 16
            tr = torch.randint(-(1 << 15), 1 << 15, (G, *sh[key][0]), dtype=torch.int32, device=DEV).to(torch.int16)
            ref = k3.reconstruct_experts(tr, parts, 0, G)
            out = torch.empty_like(ref)
            recon_t(tr, parts, 0, G, out)
            ok = torch.equal(ref, out)
            line = f"bits {BITS} {cb:5s} {nm} equal={ok}"
            if BITS == 5 and cb == "mul1":
                t0 = med(lambda: k3.reconstruct_experts(tr, parts, 0, G, out=ref))
                line += f"  prod {t0 * 1000:.1f} us"
                for KT in (1, 2, 4):
                    for nw in (2, 4, 8):
                        line += f"  t{KT}/{nw} {med(lambda: recon_t(tr, parts, 0, G, out, KT, nw)) * 1000:.1f}"
            print(line, flush=True)
