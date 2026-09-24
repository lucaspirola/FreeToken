# exp/reorg 05a110a on ft-dev: tests/moe + tests/engine + tests/scheduler + tests/kernels (box, 2026-09-24)

`reorgtest.sh`, under /root/gpu.lock. Full log: `tests.txt`.

**Result: 1575 passed, 2 failed, 14 skipped.** Neither failure comes from the merge.

1. **`tests/kernels/test_pinned_tensor.py::test_host_device_ptr_is_identity_under_uva`**
   - It fails the same way on exp/dt-dma 583afc8 (isolated runs, same box: 1 failed, 7 passed on
     both trees).
   - The test assumes that under UVA, `cudaHostGetDevicePointer` on unregistered pageable memory
     returns the pointer unchanged. This box's driver validates the pointer and returns
     `cudaErrorInvalidValue`. The assumption does not hold on this driver.
2. **`tests/kernels/test_sampling.py::test_logits_top_k_top_p_never_draws_filtered_tokens`**
   - It passes in isolation on both trees (2 passed).
   - In the suite it picks up failure 1's error. `pinned_tensor.cpp` does not clear the runtime's
     last error after the failed `cudaHostGetDevicePointer`, so torch's next launch check reports
     "invalid argument".
