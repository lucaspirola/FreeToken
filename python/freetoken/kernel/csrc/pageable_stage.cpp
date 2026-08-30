#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include <cuda_runtime_api.h>
#include <torch/extension.h>

namespace {

class PageableGather {
 public:
  PageableGather(std::vector<int64_t> sources,
                 std::vector<int64_t> destinations,
                 std::vector<int64_t> row_bytes,
                 int64_t count_ptr,
                 int64_t source_ids_ptr,
                 int64_t capacity,
                 int64_t source_rows)
      : count_(reinterpret_cast<const int64_t*>(count_ptr)),
        source_ids_(reinterpret_cast<const int32_t*>(source_ids_ptr)),
        capacity_(capacity),
        source_rows_(source_rows) {
    TORCH_CHECK(sources.size() == destinations.size() &&
                    sources.size() == row_bytes.size(),
                "pageable gather bank descriptor lengths differ");
    TORCH_CHECK(capacity > 0, "pageable gather capacity must be positive");
    TORCH_CHECK(source_rows > 0, "pageable gather source rows must be positive");
    for (size_t i = 0; i < sources.size(); ++i) {
      TORCH_CHECK(row_bytes[i] > 0, "pageable gather row size must be positive");
      sources_.push_back(reinterpret_cast<const uint8_t*>(sources[i]));
      destinations_.push_back(reinterpret_cast<uint8_t*>(destinations[i]));
      row_bytes_.push_back(static_cast<size_t>(row_bytes[i]));
    }
  }

  void launch(int64_t stream) {
    const cudaError_t err = cudaLaunchHostFunc(
        reinterpret_cast<cudaStream_t>(stream), &PageableGather::callback, this);
    TORCH_CHECK(err == cudaSuccess, "cudaLaunchHostFunc(pageable gather) failed: ",
                cudaGetErrorString(err));
  }

  std::vector<int64_t> stats() const {
    return {calls_, rows_, nanoseconds_};
  }

  void reset_stats() {
    calls_ = 0;
    rows_ = 0;
    nanoseconds_ = 0;
  }

 private:
  static void CUDART_CB callback(void* data) {
    reinterpret_cast<PageableGather*>(data)->run();
  }

  void run() {
    const auto started = std::chrono::steady_clock::now();
    const int64_t n = std::clamp<int64_t>(*count_, 0, capacity_);
    for (size_t bank = 0; bank < sources_.size(); ++bank) {
      const uint8_t* source = sources_[bank];
      uint8_t* destination = destinations_[bank];
      const size_t bytes = row_bytes_[bank];
      for (int64_t row = 0; row < n; ++row) {
        const int64_t source_row = source_ids_[row];
        if (source_row < 0 || source_row >= source_rows_) continue;
        std::memcpy(destination + static_cast<size_t>(row) * bytes,
                    source + static_cast<size_t>(source_row) * bytes, bytes);
      }
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - started);
    ++calls_;
    rows_ += n;
    nanoseconds_ += elapsed.count();
  }

  std::vector<const uint8_t*> sources_;
  std::vector<uint8_t*> destinations_;
  std::vector<size_t> row_bytes_;
  const int64_t* count_;
  const int32_t* source_ids_;
  int64_t capacity_;
  int64_t source_rows_;
  int64_t calls_ = 0;
  int64_t rows_ = 0;
  int64_t nanoseconds_ = 0;
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  pybind11::class_<PageableGather, std::shared_ptr<PageableGather>>(m, "PageableGather")
      .def(pybind11::init<std::vector<int64_t>, std::vector<int64_t>,
                          std::vector<int64_t>, int64_t, int64_t, int64_t,
                          int64_t>())
      .def("launch", &PageableGather::launch)
      .def("stats", &PageableGather::stats)
      .def("reset_stats", &PageableGather::reset_stats);
}
