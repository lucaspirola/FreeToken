import torch, json, os, shutil
import pytest

from freetoken.distributed import set_tp_info, try_get_tp_info

def _init_tp():
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)

def _mini_checkpoint(layers=2, experts=8, H=32, I=32):
    """Fabricate a native-layout NVFP4 checkpoint: 6 tensors per expert."""
    tensors = {}
    for L in range(layers):
        for e in range(experts):
            for proj, O, IN in (("up_proj", I, H), ("down_proj", H, I)):
                tensors[f"backbone.layers.{L}.mixer.experts.{e}.{proj}.weight"] = torch.randint(0, 255, (O, IN // 2), dtype=torch.uint8)
                tensors[f"backbone.layers.{L}.mixer.experts.{e}.{proj}.weight_scale"] = (torch.randn(O, IN // 16) * 10).to(torch.float8_e4m3fn)
                tensors[f"backbone.layers.{L}.mixer.experts.{e}.{proj}.weight_scale_2"] = torch.tensor(0.01 + 0.001 * e, dtype=torch.float32)
    from safetensors.torch import save_file
    path = "/tmp/excl-test/mini"
    os.makedirs(path, exist_ok=True)
    idx = {}
    shards = {}
    names = list(tensors)
    for i in range(0, len(names), 24):  # split shards so O_DIRECT reads span files
        shard = f"model-{i//24+1:05d}.safetensors"
        save_file({k: tensors[k] for k in names[i:i+24]}, os.path.join(path, shard))
        for k in names[i:i+24]:
            idx[k] = shard
    types = ["moe", "moe"]  # backbone layers 0..layers-1 are all MoE
    json.dump({"layers_block_type": types}, open(os.path.join(path, "config.json"), "w"))
    json.dump({"weight_map": idx}, open(os.path.join(path, "model.safetensors.index.json"), "w"))
    return path, tensors, types

def test_exclusive_pool_refill_matches_native():
    from freetoken.moe.exclusive_pool import ExclusiveExpertPool
    path, tensors, types = _mini_checkpoint()
    moe_ids = [i for i, k in enumerate(types) if k == "moe"]  # [1, 3]
    pool = ExclusiveExpertPool(path, 2, 8, 8, hidden_size=32, intermediate_size=32)
    # fill past capacity to force eviction, then verify a cold expert
    for e in range(8):
        pool.refill(0, e)
    slot = pool.refill(1, 3)
    assert pool.host_slot(1, 3) == slot
    key = f"backbone.layers.{moe_ids[1]}.mixer.experts.3.up_proj.weight"
    assert torch.equal(pool.banks["gate_up_packed"][slot], tensors[key])
    gkey = f"backbone.layers.{moe_ids[1]}.mixer.experts.3.up_proj.weight_scale_2"
    expected = tensors[gkey].to(torch.float16).expand(32).contiguous()
    assert torch.equal(pool.banks["gate_up_global"][slot], expected)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cache_attach_and_copy_missing_exclusive():
    from freetoken.moe.exclusive_pool import ExclusiveExpertPool
    from freetoken.moe.offload_cache import OffloadMoeCache
    path, tensors, types = _mini_checkpoint()
    moe_ids = [i for i, k in enumerate(types) if k == "moe"]
    _init_tp()
    pool = ExclusiveExpertPool(path, 2, 8, 8, hidden_size=32, intermediate_size=32)
    cache = OffloadMoeCache(
        num_layers=2, num_experts=8, cache_size=10,
        device=torch.device("cuda"), quant_format="nvfp4",
        prefill_overlap=False,
    )
    cache.attach_exclusive_pool(pool)
    topk = torch.tensor([[1, 5], [3, 7]], dtype=torch.int32, device="cuda")
    orig = topk.clone()  # ensure_experts rewrites ids to slots in place
    cache.ensure_experts(1, topk)
    count = int(cache.num_indices.item())
    admitted = dict(zip(cache.src_indices[:count].tolist(), cache.evict_slots[:count].tolist()))
    assert topk.flatten().tolist() == [admitted[e] for e in orig.flatten().tolist()]
    cache.copy_missing()
    torch.cuda.synchronize()
    for i in range(4):
        expert = int(orig.reshape(-1)[i])
        gpu_slot = int(topk.reshape(-1)[i])
        for suffix, bank in [
            ("up_proj.weight", "gate_up_packed"), ("up_proj.weight_scale", "gate_up_scale"),
            ("up_proj.weight_scale_2", "gate_up_global"), ("down_proj.weight", "down_packed"),
            ("down_proj.weight_scale", "down_scale"), ("down_proj.weight_scale_2", "down_global"),
        ]:
            key = f"backbone.layers.{moe_ids[1]}.mixer.experts.{expert}.{suffix}"
            native = tensors[key]
            got = cache.bank_caches[bank][gpu_slot]
            expected = (native.to(torch.float16).expand(got.shape) if "global" in bank else native.reshape(got.shape)).contiguous().to(got.device)
            assert torch.equal(got, expected), (expert, bank, suffix)
