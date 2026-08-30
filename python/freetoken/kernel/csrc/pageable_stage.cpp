#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <cuda_runtime_api.h>
#include <torch/extension.h>

namespace {

class ParallelCopyPool {
 public:
  ParallelCopyPool() {
    const unsigned detected = std::max(1u, std::thread::hardware_concurrency());
    int requested = std::min(4u, detected);
    if (const char* value = std::getenv("FREETOKEN_PAGEABLE_GATHER_THREADS")) {
      requested = std::max(1, std::atoi(value));
    }
    thread_count_ = std::min<int>(requested, detected);
    for (int i = 1; i < thread_count_; ++i) {
      const int worker_id = i - 1;
      workers_.emplace_back([this, worker_id] { worker_loop(worker_id); });
    }
  }

  ~ParallelCopyPool() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
      ++generation_;
    }
    start_.notify_all();
    for (auto& worker : workers_) worker.join();
  }

  int thread_count() const { return thread_count_; }

  void copy(const std::vector<const uint8_t*>& sources,
            const std::vector<uint8_t*>& destinations,
            const std::vector<size_t>& row_bytes,
            const int32_t* source_ids, int64_t rows, int64_t source_rows) {
    // The descriptors below are shared by the persistent workers. A server uses
    // one staging stream, but serialize here as well so multiple engines/streams
    // in the same process cannot overwrite an in-flight job.
    std::unique_lock<std::mutex> submit_lock(submit_mutex_);
    const int64_t tasks = rows * static_cast<int64_t>(sources.size());
    if (tasks <= 1 || workers_.empty()) {
      run_serial(sources, destinations, row_bytes, source_ids, rows, source_rows);
      return;
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      sources_ = &sources;
      destinations_ = &destinations;
      row_bytes_ = &row_bytes;
      source_ids_ = source_ids;
      rows_ = rows;
      source_rows_ = source_rows;
      task_count_ = tasks;
      next_task_.store(0, std::memory_order_relaxed);
      // The submitting thread consumes work too. Do not make it wait for more
      // background workers than the job can use (decode commonly stages one
      // row from two banks: exactly one worker can help the caller).
      active_workers_ = std::min<int64_t>(workers_.size(), tasks - 1);
      remaining_workers_ = active_workers_;
      ++generation_;
    }
    start_.notify_all();
    consume_tasks();
    std::unique_lock<std::mutex> lock(mutex_);
    done_.wait(lock, [this] { return remaining_workers_ == 0; });
  }

 private:
  static void run_serial(const std::vector<const uint8_t*>& sources,
                         const std::vector<uint8_t*>& destinations,
                         const std::vector<size_t>& row_bytes,
                         const int32_t* source_ids, int64_t rows,
                         int64_t source_rows) {
    for (size_t bank = 0; bank < sources.size(); ++bank) {
      for (int64_t row = 0; row < rows; ++row) {
        const int64_t source_row = source_ids[row];
        if (source_row < 0 || source_row >= source_rows) continue;
        std::memcpy(destinations[bank] + static_cast<size_t>(row) * row_bytes[bank],
                    sources[bank] + static_cast<size_t>(source_row) * row_bytes[bank],
                    row_bytes[bank]);
      }
    }
  }

  void consume_tasks() {
    while (true) {
      const int64_t task = next_task_.fetch_add(1, std::memory_order_relaxed);
      if (task >= task_count_) return;
      const int64_t bank = task / rows_;
      const int64_t row = task - bank * rows_;
      const int64_t source_row = source_ids_[row];
      if (source_row < 0 || source_row >= source_rows_) continue;
      const size_t bytes = (*row_bytes_)[bank];
      std::memcpy((*destinations_)[bank] + static_cast<size_t>(row) * bytes,
                  (*sources_)[bank] + static_cast<size_t>(source_row) * bytes,
                  bytes);
    }
  }

  void worker_loop(int worker_id) {
    uint64_t seen_generation = 0;
    while (true) {
      {
        std::unique_lock<std::mutex> lock(mutex_);
        start_.wait(lock, [this, &seen_generation] {
          return stopping_ || generation_ != seen_generation;
        });
        if (stopping_) return;
        seen_generation = generation_;
        if (worker_id >= active_workers_) continue;
      }
      consume_tasks();
      {
        std::lock_guard<std::mutex> lock(mutex_);
        --remaining_workers_;
        if (remaining_workers_ == 0) done_.notify_one();
      }
    }
  }

  int thread_count_ = 1;
  std::vector<std::thread> workers_;
  std::mutex submit_mutex_;
  std::mutex mutex_;
  std::condition_variable start_;
  std::condition_variable done_;
  bool stopping_ = false;
  uint64_t generation_ = 0;
  int remaining_workers_ = 0;
  int active_workers_ = 0;
  std::atomic<int64_t> next_task_{0};
  int64_t task_count_ = 0;
  const std::vector<const uint8_t*>* sources_ = nullptr;
  const std::vector<uint8_t*>* destinations_ = nullptr;
  const std::vector<size_t>* row_bytes_ = nullptr;
  const int32_t* source_ids_ = nullptr;
  int64_t rows_ = 0;
  int64_t source_rows_ = 0;
};

ParallelCopyPool& copy_pool() {
  static ParallelCopyPool pool;
  return pool;
}

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

  int64_t threads() const { return copy_pool().thread_count(); }

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
    copy_pool().copy(sources_, destinations_, row_bytes_, source_ids_, n,
                     source_rows_);
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
      .def("threads", &PageableGather::threads)
      .def("reset_stats", &PageableGather::reset_stats);
}
