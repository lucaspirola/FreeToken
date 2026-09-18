import torch
from test_exclusive_integration import _mini_checkpoint, _init_tp
from freetoken.moe.exclusive_pool import ExclusiveExpertPool, _TENSOR_MAP
from freetoken.moe.offload_cache import OffloadMoeCache
_init_tp()
path, tensors, _ = _mini_checkpoint()
pool = ExclusiveExpertPool(path, 2, 8, 8, hidden_size=32, intermediate_size=32)
cache = OffloadMoeCache(num_layers=2,num_experts=8,cache_size=10,device=torch.device('cuda'),quant_format='nvfp4',prefill_overlap=False,cache_policy='lfu',slot_capacity=10,arena_step_slots=1)
cache.direct_device_banks = True
cache.attach_exclusive_pool(pool)
def verify(layer, experts):
    routed = torch.tensor([experts], dtype=torch.int32, device='cuda')
    cache.ensure_experts(layer, routed)
    cache.copy_missing()
    for expert, slot in zip(experts, routed.flatten().tolist()):
        assert pool.host_slot(layer, expert) == -1
        for suffix, bank, scalar in _TENSOR_MAP:
            got = cache.bank_caches[bank][slot].cpu()
            native = tensors[f'backbone.layers.{layer}.mixer.experts.{expert}.{suffix}']
            expected = native.to(torch.float16).expand_as(got) if scalar else native
            assert torch.equal(got.view(torch.uint8), expected.contiguous().view(torch.uint8)), (layer, expert, bank)
    gpu = set(cache.id_of_slot.cpu().tolist()) - {-1}
    host = set(pool.id_of_host_slot) - {-1}
    assert gpu.isdisjoint(host), (gpu,host)
verify(0, [1,5,3,7])
refill = pool.refill
def no_refill(*args):
    raise AssertionError('GPU hit attempted disk refill')
pool.refill = no_refill
verify(0, [7,3,5,1])
pool.refill = refill
for layer in [1,0,1]:
    verify(layer, list(range(8)))
cache.set_usable_slots(8)
verify(0,list(range(8)))
cache.materialize_layer(1)
cache.copy_missing()
for expert in range(8):
    assert torch.equal(cache.bank_caches['gate_up_packed'][expert].cpu(), tensors[f'backbone.layers.1.mixer.experts.{expert}.up_proj.weight'])
    assert pool.host_slot(1,expert) == -1
print('PASS: native bytes, GPU hit without RAM row/refill, repeated eviction, arena shrink, full-layer prefill')
