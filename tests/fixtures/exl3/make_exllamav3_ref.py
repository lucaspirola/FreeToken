"""Write exllamav3_ref.npz: exllamav3's own EXL3 decode and linear forward on seeded random tensors.

Runs in an exllamav3 venv (not FreeToken's): ``python make_exllamav3_ref.py exllamav3_ref.npz``.
Written with exllamav3 v1.5.1 (the version that exported the Ornith checkpoint), on CUDA.

* ``rec_trellis_<bits>_<cb>`` / ``rec_w_<bits>_<cb>``: ``ext.reconstruct`` of a [2, 8] tile grid.
* ``lin_{trellis,suh,svh,x,y_gemv,y_recon}_<bits>_<cb>``: a 256 -> 128 ``LinearEXL3``; ``y_gemv``
  is its small-batch kernel (``forward``), ``y_recon`` its reconstruct + hgemm path.
"""
import sys

import numpy as np
import torch
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.quant.exl3 import LinearEXL3

MULT = {"mcg": 0xCBAC1FED, "mul1": 0x83DCD12D}


def i32(v):
    return torch.tensor(v - (1 << 32) if v >= 1 << 31 else v, dtype=torch.int32, device="cuda")


def main(path):
    g = torch.Generator().manual_seed(1234)
    from exllamav3.version import __version__

    out = {"exllamav3_version": np.array(__version__)}
    for bits in range(2, 9):
        for cb in ("3inst", "mcg", "mul1"):
            tr = torch.randint(-32768, 32768, (2, 8, 16 * bits), dtype=torch.int32, generator=g).to(torch.int16)
            w = torch.empty((32, 128), dtype=torch.half, device="cuda")
            ext.reconstruct(w, tr.cuda(), bits, cb == "mcg", cb == "mul1")
            out[f"rec_trellis_{bits}_{cb}"] = tr.numpy()
            out[f"rec_w_{bits}_{cb}"] = w.cpu().numpy()

            k, n = 256, 128
            tr = torch.randint(-32768, 32768, (k // 16, n // 16, 16 * bits), dtype=torch.int32, generator=g).to(torch.int16)
            suh = (torch.randn(k, generator=g) * 0.5).half()
            svh = (torch.randn(n, generator=g) * 0.02).half()
            x = torch.randn(5, k, generator=g).half()
            flags = {cb: i32(MULT[cb])} if cb in MULT else {}
            lin = LinearEXL3(None, k, n, suh=suh.cuda(), svh=svh.cuda(), trellis=tr.cuda(), **flags)
            y_gemv = lin.forward(x.cuda(), {})
            y_recon = lin.forward(x.cuda(), {"reconstruct": True})
            tag = f"{bits}_{cb}"
            for name, t in (("trellis", tr), ("suh", suh), ("svh", svh), ("x", x), ("y_gemv", y_gemv), ("y_recon", y_recon)):
                out[f"lin_{name}_{tag}"] = t.cpu().numpy()
    np.savez_compressed(path, **out)
    print("saved", len(out), "arrays to", path)


if __name__ == "__main__":
    main(sys.argv[1])
