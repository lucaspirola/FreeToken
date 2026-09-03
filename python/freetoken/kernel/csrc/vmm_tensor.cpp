#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

void check_cu(CUresult result, const char* operation) {
  if (result == CUDA_SUCCESS) {
    return;
  }
  const char* name = nullptr;
  const char* message = nullptr;
  cuGetErrorName(result, &name);
  cuGetErrorString(result, &message);
  throw std::runtime_error(
      std::string(operation) + " failed: " + (name ? name : "CUDA_ERROR") +
      " (" + (message ? message : "unknown error") + ")");
}

void check_cuda(cudaError_t result, const char* operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(
        std::string(operation) + " failed: " + cudaGetErrorString(result));
  }
}

at::ScalarType parse_dtype(const std::string& name) {
  if (name == "uint8") return at::kByte;
  if (name == "int8") return at::kChar;
  if (name == "float16") return at::kHalf;
  if (name == "bfloat16") return at::kBFloat16;
  if (name == "float32") return at::kFloat;
  // NVFP4 expert banks carry an fp8 per-group scale tensor; the MoE device bank cache
  // is VMM-backed whenever elastic capacity is on, so the mapping needs this dtype.
  if (name == "float8_e4m3fn") return at::kFloat8_e4m3fn;
  if (name == "float8_e5m2") return at::kFloat8_e5m2;
  throw std::invalid_argument("unsupported VMM tensor dtype: " + name);
}

size_t allocation_granularity(int device) {
  check_cuda(cudaSetDevice(device), "cudaSetDevice");
  check_cu(cuInit(0), "cuInit");
  CUmemAllocationProp prop{};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device;
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;
  size_t granularity = 0;
  check_cu(
      cuMemGetAllocationGranularity(
          &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM),
      "cuMemGetAllocationGranularity");
  return granularity;
}

struct Mapping {
  size_t offset;
  size_t size;
};

struct VMMState {
  CUdeviceptr base = 0;
  size_t reserved_bytes = 0;
  size_t granularity = 0;
  int device = 0;
  std::vector<Mapping> mappings;
  std::mutex mutex;

  ~VMMState() {
    if (!base) return;
    cudaSetDevice(device);
    for (auto it = mappings.rbegin(); it != mappings.rend(); ++it) {
      cuMemUnmap(base + it->offset, it->size);
    }
    cuMemAddressFree(base, reserved_bytes);
  }
};

class VMMAllocation {
 public:
  VMMAllocation(
      const std::vector<int64_t>& shape,
      const std::string& dtype_name,
      int device,
      int64_t reserved_bytes,
      const std::vector<std::pair<int64_t, int64_t>>& initial_ranges) {
    TORCH_CHECK(!shape.empty(), "VMM tensor shape must not be empty");
    TORCH_CHECK(reserved_bytes > 0, "reserved_bytes must be positive");
    check_cuda(cudaSetDevice(device), "cudaSetDevice");
    check_cu(cuInit(0), "cuInit");

    CUdevice cu_device;
    check_cu(cuDeviceGet(&cu_device, device), "cuDeviceGet");
    int supported = 0;
    check_cu(
        cuDeviceGetAttribute(
            &supported,
            CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
            cu_device),
        "cuDeviceGetAttribute(VMM)");
    TORCH_CHECK(supported, "CUDA device does not support virtual memory management");

    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;

    size_t granularity = 0;
    check_cu(
        cuMemGetAllocationGranularity(
            &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM),
        "cuMemGetAllocationGranularity");
    const size_t requested = static_cast<size_t>(reserved_bytes);
    const size_t aligned = ((requested + granularity - 1) / granularity) * granularity;

    auto state = std::make_shared<VMMState>();
    state->reserved_bytes = aligned;
    state->granularity = granularity;
    state->device = device;
    check_cu(
        cuMemAddressReserve(&state->base, aligned, granularity, 0, 0),
        "cuMemAddressReserve");

    const auto dtype = parse_dtype(dtype_name);
    int64_t numel = 1;
    for (int64_t dim : shape) {
      TORCH_CHECK(dim >= 0, "negative VMM tensor dimension");
      TORCH_CHECK(dim == 0 || numel <= INT64_MAX / dim, "VMM tensor numel overflow");
      numel *= dim;
    }
    const size_t tensor_bytes = static_cast<size_t>(numel) * c10::elementSize(dtype);
    TORCH_CHECK(
        tensor_bytes <= aligned,
        "tensor requires ", tensor_bytes, " bytes but reservation has ", aligned);

    state_ = std::move(state);
    commit_ranges(initial_ranges);
    tensor_ = torch::from_blob(
        reinterpret_cast<void*>(state_->base),
        shape,
        [keepalive = state_](void*) mutable { keepalive.reset(); },
        torch::TensorOptions().dtype(dtype).device(torch::kCUDA, device));
  }

  torch::Tensor tensor() const { return tensor_; }
  int64_t granularity() const { return static_cast<int64_t>(state_->granularity); }
  int64_t reserved_bytes() const { return static_cast<int64_t>(state_->reserved_bytes); }

  int64_t mapped_bytes() const {
    std::lock_guard<std::mutex> lock(state_->mutex);
    size_t total = 0;
    for (const auto& mapping : state_->mappings) total += mapping.size;
    return static_cast<int64_t>(total);
  }

  void commit_ranges(const std::vector<std::pair<int64_t, int64_t>>& ranges) {
    std::lock_guard<std::mutex> lock(state_->mutex);
    check_cuda(cudaSetDevice(state_->device), "cudaSetDevice");

    std::vector<Mapping> requested;
    requested.reserve(ranges.size());
    for (const auto& range : ranges) {
      TORCH_CHECK(range.first >= 0 && range.second > 0, "invalid VMM mapping range");
      const size_t offset = static_cast<size_t>(range.first);
      const size_t size = static_cast<size_t>(range.second);
      TORCH_CHECK(
          offset % state_->granularity == 0 && size % state_->granularity == 0,
          "VMM ranges must be aligned to ", state_->granularity, " bytes");
      TORCH_CHECK(
          offset <= state_->reserved_bytes && size <= state_->reserved_bytes - offset,
          "VMM mapping range exceeds reservation");
      requested.push_back({offset, size});
    }
    std::sort(requested.begin(), requested.end(), [](const Mapping& a, const Mapping& b) {
      return a.offset < b.offset;
    });
    for (size_t i = 0; i < requested.size(); ++i) {
      if (i) {
        TORCH_CHECK(
            requested[i - 1].offset + requested[i - 1].size <= requested[i].offset,
            "overlapping ranges in one VMM commit");
      }
      for (const auto& mapped : state_->mappings) {
        TORCH_CHECK(
            requested[i].offset + requested[i].size <= mapped.offset ||
                mapped.offset + mapped.size <= requested[i].offset,
            "VMM range overlaps an existing mapping");
      }
    }

    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = state_->device;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;
    CUmemAccessDesc access{};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = state_->device;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    std::vector<Mapping> added;
    try {
      for (const auto& range : requested) {
        CUmemGenericAllocationHandle handle{};
        check_cu(cuMemCreate(&handle, range.size, &prop, 0), "cuMemCreate");
        try {
          check_cu(
              cuMemMap(state_->base + range.offset, range.size, 0, handle, 0),
              "cuMemMap");
          check_cu(
              cuMemSetAccess(state_->base + range.offset, range.size, &access, 1),
              "cuMemSetAccess");
        } catch (...) {
          cuMemUnmap(state_->base + range.offset, range.size);
          cuMemRelease(handle);
          throw;
        }
        check_cu(cuMemRelease(handle), "cuMemRelease");
        added.push_back(range);
      }
    } catch (...) {
      for (auto it = added.rbegin(); it != added.rend(); ++it) {
        cuMemUnmap(state_->base + it->offset, it->size);
      }
      throw;
    }
    state_->mappings.insert(state_->mappings.end(), added.begin(), added.end());
  }

  void uncommit_ranges(const std::vector<std::pair<int64_t, int64_t>>& ranges) {
    std::lock_guard<std::mutex> lock(state_->mutex);
    check_cuda(cudaSetDevice(state_->device), "cudaSetDevice");

    std::vector<Mapping> requested;
    requested.reserve(ranges.size());
    for (const auto& range : ranges) {
      TORCH_CHECK(range.first >= 0 && range.second > 0, "invalid VMM unmapping range");
      const size_t offset = static_cast<size_t>(range.first);
      const size_t size = static_cast<size_t>(range.second);
      TORCH_CHECK(
          offset % state_->granularity == 0 && size % state_->granularity == 0,
          "VMM ranges must be aligned to ", state_->granularity, " bytes");
      TORCH_CHECK(
          offset <= state_->reserved_bytes && size <= state_->reserved_bytes - offset,
          "VMM unmapping range exceeds reservation");
      requested.push_back({offset, size});
    }
    std::sort(requested.begin(), requested.end(), [](const Mapping& a, const Mapping& b) {
      return a.offset < b.offset;
    });
    for (size_t i = 1; i < requested.size(); ++i) {
      TORCH_CHECK(
          requested[i - 1].offset + requested[i - 1].size <= requested[i].offset,
          "overlapping ranges in one VMM uncommit");
    }

    // CUDA only permits unmapping a complete physical mapping. Require each requested range
    // to exactly match one commit_ranges allocation; growth therefore creates one mapping per
    // shrink step. Prevalidate the whole operation before the first destructive unmap.
    for (const auto& range : requested) {
      const auto found = std::find_if(
          state_->mappings.begin(), state_->mappings.end(),
          [&range](const Mapping& mapped) {
            return mapped.offset == range.offset && mapped.size == range.size;
          });
      TORCH_CHECK(
          found != state_->mappings.end(),
          "VMM uncommit range must exactly match an existing mapping");
    }

    for (const auto& range : requested) {
      check_cu(cuMemUnmap(state_->base + range.offset, range.size), "cuMemUnmap");
    }

    std::vector<Mapping> remaining;
    for (const auto& mapped : state_->mappings) {
      const auto removed = std::any_of(
          requested.begin(), requested.end(), [&mapped](const Mapping& range) {
            return mapped.offset == range.offset && mapped.size == range.size;
          });
      if (!removed) remaining.push_back(mapped);
    }
    state_->mappings = std::move(remaining);
  }

 private:
  std::shared_ptr<VMMState> state_;
  torch::Tensor tensor_;
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("allocation_granularity", &allocation_granularity);
  py::class_<VMMAllocation>(m, "VMMAllocation")
      .def(py::init<
           const std::vector<int64_t>&,
           const std::string&,
           int,
           int64_t,
           const std::vector<std::pair<int64_t, int64_t>>&>())
      .def_property_readonly("tensor", &VMMAllocation::tensor)
      .def_property_readonly("granularity", &VMMAllocation::granularity)
      .def_property_readonly("reserved_bytes", &VMMAllocation::reserved_bytes)
      .def_property_readonly("mapped_bytes", &VMMAllocation::mapped_bytes)
      .def("commit_ranges", &VMMAllocation::commit_ranges)
      .def("uncommit_ranges", &VMMAllocation::uncommit_ranges);
}
