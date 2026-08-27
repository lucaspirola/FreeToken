import pytest
import torch


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_logits_top_k_top_p_never_draws_filtered_tokens():
    from freetoken.kernel.triton.sampling import top_k_top_p_sampling_from_logits

    # With p=0.5 the first token alone crosses the nucleus threshold in both rows.
    logits = torch.tensor(
        [[9.0, 3.0, 2.0, 1.0, 0.0], [0.0, 8.0, 3.0, 2.0, 1.0]],
        device="cuda",
    ).repeat_interleave(128, dim=0)
    temperatures = torch.ones(logits.size(0), device="cuda")
    top_k = torch.tensor([2, 4], device="cuda", dtype=torch.int32).repeat_interleave(128)
    top_p = torch.full((logits.size(0),), 0.5, device="cuda")

    got = top_k_top_p_sampling_from_logits(
        logits, temperatures, top_k, top_p, max_top_k=4
    ).view(2, 128)
    assert torch.equal(got[0], torch.zeros(128, device="cuda", dtype=got.dtype))
    assert torch.equal(got[1], torch.ones(128, device="cuda", dtype=got.dtype))


def test_logits_top_k_respects_per_row_k_without_top_p():
    from freetoken.kernel.triton.sampling import top_k_top_p_sampling_from_logits

    logits = torch.tensor(
        [[5.0, 4.0, 3.0, 2.0], [2.0, 3.0, 4.0, 5.0]], device="cuda"
    ).repeat_interleave(128, dim=0)
    temperatures = torch.ones(logits.size(0), device="cuda")
    top_k = torch.tensor([1, 3], device="cuda", dtype=torch.int32).repeat_interleave(128)

    got = top_k_top_p_sampling_from_logits(
        logits, temperatures, top_k, max_top_k=3
    ).view(2, 128)
    assert torch.equal(got[0], torch.zeros(128, device="cuda", dtype=got.dtype))
    assert set(got[1].tolist()) <= {1, 2, 3}
