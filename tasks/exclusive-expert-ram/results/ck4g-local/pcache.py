#!/usr/bin/env python3
"""Page-cache residency per file (mincore), for attributing a cgroup's file charge.
  pcache.py OUT.tsv DIR...   -> path, size, cached_bytes for every regular file >= 4 KiB."""
import ctypes, mmap, os, sys
libc = ctypes.CDLL("libc.so.6", use_errno=True)
libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
PG = mmap.PAGESIZE
def cached(path, size):
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOATIME)
    except OSError:
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError:
            return -1
    try:
        addr = libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
        if addr in (None, ctypes.c_void_p(-1).value): return -1
        n = (size + PG - 1) // PG
        vec = ctypes.create_string_buffer(n)
        r = libc.mincore(ctypes.c_void_p(addr), size, vec)
        libc.munmap(ctypes.c_void_p(addr), size)
        if r != 0: return -1
        return sum(b & 1 for b in vec.raw) * PG
    except OSError:
        return -1
    finally:
        os.close(fd)
out = open(sys.argv[1], "w")
tot = 0
for d in sys.argv[2:]:
    for root, dirs, files in os.walk(d):
        for f in files:
            p = os.path.join(root, f)
            try:
                st = os.lstat(p)
            except OSError:
                continue
            if not os.path.isfile(p) or os.path.islink(p) or st.st_size < 4096: continue
            c = cached(p, st.st_size)
            if c > 0:
                out.write(f"{p}\t{st.st_size}\t{c}\n"); tot += c
out.close()
print(f"{sys.argv[1]}: {tot/2**30:.2f} GiB cached", file=sys.stderr)
