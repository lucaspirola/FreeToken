import sys, torch, numpy as np
sys.path.insert(0, "/home/lucas/ai/tools/exllamav3")
from exllamav3.ext import exllamav3_ext as ext

def stream_words(tile_u16, K):
    # tile_u16: np.uint16 array of 16*K entries; returns 256 16-bit words per stream position t
    u32 = tile_u16.view(np.uint32).astype(np.uint64)  # little-endian pairs
    nbits = 256 * K
    # bit j (MSB-first over uint32 words)
    bits = np.zeros(nbits, dtype=np.uint64)
    for j in range(nbits):
        bits[j] = (u32[j // 32] >> np.uint64(31 - j % 32)) & np.uint64(1)
    words = np.zeros(256, dtype=np.uint64)
    for t in range(256):
        b0 = t * K + K - 16
        w = 0
        for q in range(16):
            w = (w << 1) | int(bits[(b0 + q) % nbits])
        words[t] = w
    return words

def decode(words, cb):
    x = words.astype(np.uint64)
    if cb == 2:
        x = (x * 0x83DCD12D) & 0xFFFFFFFF
        s = (x & 0xff) + ((x >> 8) & 0xff) + ((x >> 16) & 0xff) + ((x >> 24) & 0xff)
        h = (1024 + s).astype(np.float64)
        kinv = float(np.array([0x1eee], dtype=np.uint16).view(np.float16)[0])
        kb = float(np.array([0xc931], dtype=np.uint16).view(np.float16)[0])
        return (h * kinv + kb).astype(np.float16)
    if cb == 1:
        x = (x * 0xCBAC1FED) & 0xFFFFFFFF
    else:
        x = ((x * 89226354) + 64248484) & 0xFFFFFFFF
    x = (x & 0x8fff8fff) ^ 0x3b603b60
    x = x.astype(np.uint32)
    lo = (x & 0xffff).astype(np.uint16).view(np.float16)
    hi = (x >> 16).astype(np.uint16).view(np.float16)
    return (lo + hi).astype(np.float16)  # fp16 add

torch.manual_seed(0)
perm = None
for K in (5, 6, 7, 4, 3, 8, 2):
  for cb in (2, 1, 0):
    k, n = 32, 256
    tr = torch.randint(-32768, 32767, (k // 16, n // 16, 16 * K), dtype=torch.int16, device="cuda")
    w = torch.empty((k, n), dtype=torch.half, device="cuda")
    ext.reconstruct(w, tr, K, cb == 1, cb == 2)
    torch.cuda.synchronize()
    W = w.cpu().numpy()
    trn = tr.cpu().numpy().view(np.uint16)
    ok = True
    for kt in range(k // 16):
        for nt in range(n // 16):
            vals = decode(stream_words(trn[kt, nt], K), cb)
            blk = W[kt*16:(kt+1)*16, nt*16:(nt+1)*16]
            if perm is None:
                cands = [set(range(256)) for _ in range(256)]
                for kt2 in range(k // 16):
                    for nt2 in range(n // 16):
                        v2 = decode(stream_words(trn[kt2, nt2], K), cb).view(np.uint16)
                        b2 = W[kt2*16:(kt2+1)*16, nt2*16:(nt2+1)*16].reshape(-1).view(np.uint16)
                        for t in range(256):
                            cands[t] &= set(np.nonzero(b2 == v2[t])[0].tolist())
                perm = np.array([min(c) if len(c) == 1 else -1 for c in cands])
                print("derived", (perm >= 0).sum(), "unique of 256")
                assert (perm >= 0).all()
                assert len(set(perm.tolist())) == 256
            got = blk.reshape(-1)[perm]
            if not np.array_equal(got.view(np.uint16), vals.view(np.uint16)):
                ok = False
                print("mismatch", K, cb, kt, nt, (got.view(np.uint16) != vals.view(np.uint16)).sum())
                break
        if not ok: break
    print("K", K, "cb", cb, "bit-exact" if ok else "MISMATCH")
np.save("tile_perm.npy", perm)
print(perm.reshape(16,16))
