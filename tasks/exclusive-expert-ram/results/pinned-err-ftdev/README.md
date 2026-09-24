# exp/pinned-err a658c8c on ft-dev (box, RTX 5080, driver 595.58.03, 2026-09-24)

`pintest.sh`: tests/moe + tests/engine + tests/scheduler + tests/kernels under /root/gpu.lock, extension
rebuilt in place (`setup.py build_ext --inplace`).

**1577 passed, 14 skipped, 0 failed** (`summary.txt`, `tests.txt`). exp/reorg 05a110a on the same box had 1575
passed, 2 failed (`../reorg-merge-ftdev/`):

- `test_host_device_ptr_is_identity_under_uva`: driver 595.58 rejects `cudaHostGetDevicePointer` on a
  plain host pointer. The test now accepts either the identity or a clean `RuntimeError`, then launches a
  torch kernel and checks its result in the same test.
- `test_sampling`: collateral. The failed call left the runtime's sticky last error set, and torch's next
  post-launch `cudaGetLastError` reported it as its own failure. `pinned_tensor.cpp` now clears the last
  error (`check_cuda`) before raising, at all 7 checked call sites.
