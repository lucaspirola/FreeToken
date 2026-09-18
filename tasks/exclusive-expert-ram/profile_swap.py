import cProfile, pstats, time, torch
from freetoken.distributed import set_tp_info
from freetoken.moe.exclusive_pool import ExclusiveExpertPool
from freetoken.moe.offload_cache import OffloadMoeCache
set_tp_info(rank=0,size=1)
p=ExclusiveExpertPool('/home/lucas/ai/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4',23,128,128,hidden_size=2688,intermediate_size=1856)
c=OffloadMoeCache(num_layers=23,num_experts=128,cache_size=128,device=torch.device('cuda'),quant_format='nvfp4',prefill_overlap=False,cache_policy='lfu')
c.attach_exclusive_pool(p)
prof=cProfile.Profile(); prof.enable()
for iteration in range(2):
    for layer in range(3):
        c.materialize_layer(layer); c.copy_missing()
prof.disable(); pstats.Stats(prof).strip_dirs().sort_stats('tottime').print_stats(18)
print('Completed six native full-layer transitions; bounded host bytes',p.pool_bytes)
