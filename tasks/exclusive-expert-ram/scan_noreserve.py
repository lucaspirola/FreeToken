import re,sys
# Flags every prefill that starts while the arena is at the decode level (a release with no
# reserve since), which is the dyn-g5 bug: release -> KV shrink (flag cleared) -> prefill.
for path in sys.argv[1:]:
    j=re.sub(r"\x1b\[[0-9;]*m","",open(path,errors="replace").read()).splitlines()
    start=max(i for i,l in enumerate(j) if "ServerArgs(model_path" in l) if any("ServerArgs(model_path" in l for l in j) else 0
    decode_level=False; bad=0; shrink_after_release=False; prefills=0
    for i,l in enumerate(j[start:],start):
        if "Prefill headroom released to decode" in l: decode_level=True; shrink_after_release=False
        elif "Prefill headroom reserved" in l: decode_level=False
        elif "Released growable KV" in l and decode_level: shrink_after_release=True
        elif "Committed growable KV" in l: decode_level=False
        elif "Prefill batch" in l:
            prefills+=1
            if decode_level:
                bad+=1
                if bad<=3: print(f"  {path.split('/')[-1]}:{i+1} prefill at decode level (shrink after release: {shrink_after_release})")
                decode_level=False
    print(f"{path.split('/')[-1]}: {prefills} prefill batches, {bad} started at the decode level")
