import ctypes
import time
import torch
H,I=2688,1856
src=torch.empty((I,H//2),dtype=torch.uint8)
dst=torch.empty_like(src,pin_memory=True)
gpu=torch.empty_like(src,device='cuda')
src.fill_(17)
for label,copy in [('torch_cpu',lambda:dst.copy_(src)),('memmove_cpu',lambda:ctypes.memmove(dst.data_ptr(),src.data_ptr(),src.numel()))]:
    start=time.perf_counter()
    for _ in range(128):copy()
    print(label,'ms/row',(time.perf_counter()-start)*1000/128,flush=True)
    assert torch.equal(dst,src)
a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
a.record()
for _ in range(128):gpu.copy_(dst,non_blocking=True)
b.record();b.synchronize()
print('H2D CUDA ms/row',a.elapsed_time(b)/128,'torch_threads',torch.get_num_threads(),flush=True)
