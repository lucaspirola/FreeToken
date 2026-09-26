"""exl3 worker: exact VRAM ledger around the growable-KV arena (diagnostic, no behaviour change except a sync per
forward). Wraps longctx_decode2.main. Prints LEDGER lines: driver free, torch allocated/reserved, arena committed bytes,
KV mapped bytes, and resid = total - free - reserved - arena - kv (VRAM outside the torch allocator and the VMM pools),
at startup, at every headroom/grow boundary, and after any forward where alloc/reserved/resid moved by > 1 MiB."""
import sys, torch
sys.path.insert(0, "/root/K/longctx")
import longctx_decode2 as L
from freetoken.engine import growable_kv as G
from freetoken.engine.engine import Engine

last = {}
def led(tag, eng, force=True):
    ctl = eng.growable_kv; moe = ctl.moe
    torch.cuda.synchronize()
    f, t = torch.cuda.mem_get_info()
    a, r = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
    ab = ctl._arena_transition_bytes(moe.cache_size)
    kv = ctl.kv_cache.mapped_bytes_for_pages(int(ctl.kv_cache.committed_pages))
    res = t - f - r - ab - kv
    cur = dict(a=a, r=r, res=res)
    moved = any(abs(cur[k] - last.get(k, cur[k])) > (1 << 20) for k in cur)
    if force or moved or not last:
        print(f"LEDGER {tag} slots={moe.cache_size} free={f} alloc={a} resv={r} arena={ab} kv={kv} resid={res} "
              f"live_budget={getattr(eng, chr(95)+"growable_live_budget", None)} headroom={ctl.headroom_bytes()}", flush=True)
    last.update(cur)

def wrap(cls, name, before=True, after=True):
    orig = getattr(cls, name)
    def w(self, *a, **k):
        eng = self.engine if hasattr(self, "engine") else self
        if before: led(f"{name}:before", eng)
        out = orig(self, *a, **k)
        if after: led(f"{name}:after", eng)
        return out
    setattr(cls, name, w)

for n in ("reserve_prefill_headroom", "release_prefill_headroom"):
    wrap(G.GrowableKvController, n)
wrap(G.GrowableKvController, "_grow_runtime_kv_arena", after=True)
wrap(Engine, "_log_startup_geometry", before=False)
nfw = [0]
orig_fb = Engine.forward_batch
def fb(self, *a, **k):
    out = orig_fb(self, *a, **k)
    nfw[0] += 1
    if getattr(self, "growable_kv", None) is not None and getattr(self, "_growable_moe_ceiling", None) is not None:
        led(f"forward#{nfw[0]}", self, force=False)
    return out
Engine.forward_batch = fb

if __name__ == "__main__":
    sep = sys.argv.index("--")
    L.main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[sep + 1:])
