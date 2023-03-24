// Copyright 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>
#include "../doctest.h"
#include "client_backend.h"
#include "gmock/gmock.h"

namespace triton { namespace perfanalyzer { namespace clientbackend {

// Holds information (either the raw data or a shared memory label) for an
// inference input
//
struct TestRecordedInput {
  TestRecordedInput(int32_t data_in, size_t size_in)
      : shared_memory_label(""), data(data_in), size(size_in)
  {
  }

  TestRecordedInput(std::string label_in, size_t size_in)
      : shared_memory_label(label_in), data(0), size(size_in)
  {
  }

  std::string shared_memory_label;
  int32_t data;
  size_t size;
};

/// Mock class of an InferInput
///
class MockInferInput : public InferInput {
 public:
  MockInferInput(
      const BackendKind kind, const std::string& name,
      const std::vector<int64_t>& dims, const std::string& datatype)
      : InferInput(kind, name, datatype), dims_(dims)
  {
  }

  const std::vector<int64_t>& Shape() const override { return dims_; }

  Error Reset() override
  {
    recorded_inputs_.clear();
    return Error::Success;
  }

  Error AppendRaw(const uint8_t* input, size_t input_byte_size) override
  {
    if (input) {
      int32_t val = *reinterpret_cast<const int32_t*>(input);
      recorded_inputs_.push_back(TestRecordedInput(val, input_byte_size));
    }
    ++append_raw_calls_;
    return Error::Success;
  }

  Error SetSharedMemory(
      const std::string& name, size_t byte_size, size_t offset = 0)
  {
    recorded_inputs_.push_back(TestRecordedInput(name, byte_size));
    ++set_shared_memory_calls_;
    return Error::Success;
  }

  const std::vector<int64_t> dims_{};
  std::vector<TestRecordedInput> recorded_inputs_{};
  std::atomic<size_t> append_raw_calls_{0};
  std::atomic<size_t> set_shared_memory_calls_{0};
};

/// Mock class of an InferResult
///
class MockInferResult : public InferResult {
 public:
  MockInferResult(const InferOptions& options) : req_id_{options.request_id_} {}

  Error Id(std::string* id) const override
  {
    *id = req_id_;
    return Error::Success;
  }
  Error RequestStatus() const override { return Error::Success; }
  Error RawData(
      const std::string& output_name, const uint8_t** buf,
      size_t* byte_size) const override
  {
    return Error::Success;
  }

 private:
  std::string req_id_;
};

/// Class to track statistics of MockClientBackend
///
class MockClientStats {
 public:
  enum class ReqType { SYNC, ASYNC, ASYNC_STREAM };

  struct SeqStatus {
    std::set<uint64_t> used_seq_ids;
    std::map<uint64_t, uint32_t> live_seq_ids_to_length;
    std::map<uint64_t, uint32_t> seq_ids_to_count;
    uint32_t max_live_seq_count = 0;
    std::vector<uint64_t> seq_lengths;

    bool IsSeqLive(uint64_t seq_id)
    {
      return (
          live_seq_ids_to_length.find(seq_id) != live_seq_ids_to_length.end());
    }
    void HandleSeqStart(uint64_t seq_id)
    {
      used_seq_ids.insert(seq_id);
      live_seq_ids_to_length[seq_id] = 0;
      if (live_seq_ids_to_length.size() > max_live_seq_count) {
        max_live_seq_count = live_seq_ids_to_length.size();
      }
    }
    void HandleSeqEnd(uint64_t seq_id)
    {
      uint32_t len = live_seq_ids_to_length[seq_id];
      seq_lengths.push_back(len);
      auto it = live_seq_ids_to_length.find(seq_id);
      live_seq_ids_to_length.erase(it);
    }

    void HandleSeqRequest(uint64_t seq_id)
    {
      live_seq_ids_to_length[seq_id]++;

      if (seq_ids_to_count.find(seq_id) == seq_ids_to_count.end()) {
        seq_ids_to_count[seq_id] = 1;
      } else {
        seq_ids_to_count[seq_id]++;
      }
    }

    void Reset()
    {
      // Note that live_seq_ids_to_length is explicitly not reset here.
      // This is because we always want to maintain the true status of
      // live sequences

      used_seq_ids.clear();
      max_live_seq_count = 0;
      seq_lengths.clear();
      seq_ids_to_count.clear();
    }
  };

  std::atomic<size_t> num_infer_calls{0};
  std::atomic<size_t> num_async_infer_calls{0};
  std::atomic<size_t> num_async_stream_infer_calls{0};
  std::atomic<size_t> num_start_stream_calls{0};

  std::atomic<size_t> num_active_infer_calls{0};

  std::atomic<size_t> num_append_raw_calls{0};
  std::atomic<size_t> num_set_shared_memory_calls{0};
  // Struct tracking shared memory method calls
  //
  struct SharedMemoryStats {
    std::atomic<size_t> num_unregister_all_shared_memory_calls{0};
    std::atomic<size_t> num_register_system_shared_memory_calls{0};
    std::atomic<size_t> num_register_cuda_shared_memory_calls{0};
    std::atomic<size_t> num_register_cuda_memory_calls{0};
    std::atomic<size_t> num_register_system_memory_calls{0};
    std::atomic<size_t> num_create_shared_memory_region_calls{0};
    std::atomic<size_t> num_map_shared_memory_calls{0};
    std::atomic<size_t> num_close_shared_memory_calls{0};
    std::atomic<size_t> num_unlink_shared_memory_region_calls{0};
    std::atomic<size_t> num_unmap_shared_memory_calls{0};

    // bool operator==(const SharedMemoryStats& lhs, const SharedMemoryStats&
    // rhs)
    bool operator==(const SharedMemoryStats& rhs) const
    {
      if (this->num_unregister_all_shared_memory_calls ==
              rhs.num_unregister_all_shared_memory_calls &&
          this->num_register_system_shared_memory_calls ==
              rhs.num_register_system_shared_memory_calls &&
          this->num_register_cuda_shared_memory_calls ==
              rhs.num_register_cuda_shared_memory_calls &&
          this->num_register_cuda_memory_calls ==
              rhs.num_register_cuda_memory_calls &&
          this->num_register_system_memory_calls ==
              rhs.num_register_system_memory_calls &&
          this->num_create_shared_memory_region_calls ==
              rhs.num_create_shared_memory_region_calls &&
          this->num_map_shared_memory_calls ==
              rhs.num_map_shared_memory_calls &&
          this->num_close_shared_memory_calls ==
              rhs.num_close_shared_memory_calls &&
          this->num_unlink_shared_memory_region_calls ==
              rhs.num_unlink_shared_memory_region_calls &&
          this->num_unmap_shared_memory_calls ==
              rhs.num_unmap_shared_memory_calls) {
        return true;
      }
      return false;
    }
  };

  /// Determines how long the backend will delay before sending a "response".
  /// If a single value vector is passed in, all responses will take that long.
  /// If a list of values is passed in, then the mock backend will loop through
  /// the values (and loop back to the start when it hits the end of the vector)
  ///
  void SetDelays(std::vector<size_t> times)
  {
    response_delays_.clear();
    for (size_t t : times) {
      response_delays_.push_back(std::chrono::milliseconds{t});
    }
  }

  /// Determines the return status of requests.
  /// If a single value vector is passed in, all responses will return that
  /// status. If a list of values is passed in, then the mock backend will loop
  /// through the values (and loop back to the start when it hits the end of the
  /// vector)
  ///
  void SetReturnStatuses(std::vector<bool> statuses)
  {
    response_statuses_.clear();
    for (bool success : statuses) {
      if (success) {
        response_statuses_.push_back(Error::Success);
      } else {
        response_statuses_.push_back(Error("Injected test error"));
      }
    }
  }

  std::chrono::milliseconds GetNextDelay()
  {
    std::lock_guard<std::mutex> lock(mtx_);

    auto val = response_delays_[response_delays_index_];
    response_delays_index_++;
    if (response_delays_index_ == response_delays_.size()) {
      response_delays_index_ = 0;
    }
    return val;
  }

  Error GetNextReturnStatus()
  {
    std::lock_guard<std::mutex> lock(mtx_);

    auto val = response_statuses_[response_statuses_index_];
    response_statuses_index_++;
    if (response_statuses_index_ == response_statuses_.size()) {
      response_statuses_index_ = 0;
    }
    return val;
  }

  bool start_stream_enable_stats_value{false};

  std::vector<std::chrono::time_point<std::chrono::system_clock>>
      request_timestamps;
  SeqStatus sequence_status;
  SharedMemoryStats memory_stats;

  // Each entry in the top vector is a list of all inputs for an inference
  // request. If there are multiple inputs due to batching and/or the model
  // having multiple inputs, all of those from the same request will be in the
  // same second level vector
  std::vector<std::vector<TestRecordedInput>> recorded_inputs{};

  void CaptureRequest(
      ReqType type, const InferOptions& options,
      const std::vector<InferInput*>& inputs,
      const std::vector<const InferRequestedOutput*>& outputs)
  {
    std::lock_guard<std::mutex> lock(mtx_);
    auto time = std::chrono::system_clock::now();
    request_timestamps.push_back(time);

    // Group all values across all inputs together into a single vector, and
    // then record it
    std::vector<TestRecordedInput> request_inputs;
    for (const auto& input : inputs) {
      auto recorded_inputs =
          static_cast<const MockInferInput*>(input)->recorded_inputs_;
      request_inputs.insert(
          request_inputs.end(), recorded_inputs.begin(), recorded_inputs.end());
    }
    recorded_inputs.push_back(request_inputs);

    UpdateCallCount(type);
    UpdateSeqStatus(options);
    AccumulateInferInputCalls(inputs);
  }

  void CaptureStreamStart()
  {
    std::lock_guard<std::mutex> lock(mtx_);
    num_start_stream_calls++;
  }


  void Reset()
  {
    std::lock_guard<std::mutex> lock(mtx_);
    num_infer_calls = 0;
    num_async_infer_calls = 0;
    num_async_stream_infer_calls = 0;
    num_start_stream_calls = 0;
    request_timestamps.clear();
    sequence_status.Reset();
  }

 private:
  std::vector<std::chrono::milliseconds> response_delays_{
      std::chrono::milliseconds{0}};
  std::vector<Error> response_statuses_{Error::Success};
  std::atomic<size_t> response_delays_index_{0};
  std::atomic<size_t> response_statuses_index_{0};

  std::mutex mtx_;

  void UpdateCallCount(ReqType type)
  {
    if (type == ReqType::SYNC) {
      num_infer_calls++;
    } else if (type == ReqType::ASYNC) {
      num_async_infer_calls++;
    } else {
      num_async_stream_infer_calls++;
    }
  }

  void UpdateSeqStatus(const InferOptions& options)
  {
    // Seq ID of 0 is reserved for "not a sequence"
    //
    if (options.sequence_id_ != 0) {
      // If a sequence ID is not live, it must be starting
      if (!sequence_status.IsSeqLive(options.sequence_id_)) {
        REQUIRE(options.sequence_start_ == true);
      }

      // If a new sequence is starting, that sequence ID must not already be
      // live
      if (options.sequence_start_ == true) {
        REQUIRE(sequence_status.IsSeqLive(options.sequence_id_) == false);
        sequence_status.HandleSeqStart(options.sequence_id_);
      }

      sequence_status.HandleSeqRequest(options.sequence_id_);

      // If a sequence is ending, it must be live
      if (options.sequence_end_) {
        REQUIRE(sequence_status.IsSeqLive(options.sequence_id_) == true);
        sequence_status.HandleSeqEnd(options.sequence_id_);
      }
    }
  }

  void AccumulateInferInputCalls(const std::vector<InferInput*>& inputs)
  {
    for (const auto& input : inputs) {
      const MockInferInput* mock_input =
          static_cast<const MockInferInput*>(input);
      num_append_raw_calls += mock_input->append_raw_calls_;
      num_set_shared_memory_calls += mock_input->set_shared_memory_calls_;
    }
  }
};

/// Mock implementation of ClientBackend interface
///
class MockClientBackend : public ClientBackend {
 public:
  MockClientBackend(std::shared_ptr<MockClientStats> stats) { stats_ = stats; }

  MOCK_METHOD(
      Error, ModelConfig,
      (rapidjson::Document*, const std::string&, const std::string&),
      (override));

  Error Infer(
      InferResult** result, const InferOptions& options,
      const std::vector<InferInput*>& inputs,
      const std::vector<const InferRequestedOutput*>& outputs) override
  {
    stats_->num_active_infer_calls++;

    stats_->CaptureRequest(
        MockClientStats::ReqType::SYNC, options, inputs, outputs);

    std::this_thread::sleep_for(stats_->GetNextDelay());

    local_completed_req_count_++;
    stats_->num_active_infer_calls--;

    return stats_->GetNextReturnStatus();
  }

  Error AsyncInfer(
      OnCompleteFn callback, const InferOptions& options,
      const std::vector<InferInput*>& inputs,
      const std::vector<const InferRequestedOutput*>& outputs) override
  {
    stats_->num_active_infer_calls++;

    stats_->CaptureRequest(
        MockClientStats::ReqType::ASYNC, options, inputs, outputs);

    LaunchAsyncMockRequest(options, callback);

    return stats_->GetNextReturnStatus();
  }

  Error AsyncStreamInfer(
      const InferOptions& options, const std::vector<InferInput*>& inputs,
      const std::vector<const InferRequestedOutput*>& outputs)
  {
    stats_->num_active_infer_calls++;

    stats_->CaptureRequest(
        MockClientStats::ReqType::ASYNC_STREAM, options, inputs, outputs);

    LaunchAsyncMockRequest(options, stream_callback_);

    return stats_->GetNextReturnStatus();
  }

  Error StartStream(OnCompleteFn callback, bool enable_stats)
  {
    stats_->CaptureStreamStart();
    stats_->start_stream_enable_stats_value = enable_stats;
    stream_callback_ = callback;
    return stats_->GetNextReturnStatus();
  }

  Error ClientInferStat(InferStat* infer_stat) override
  {
    infer_stat->completed_request_count = local_completed_req_count_;
    return Error::Success;
  }

  Error UnregisterAllSharedMemory() override
  {
    stats_->memory_stats.num_unregister_all_shared_memory_calls++;
    return Error::Success;
  }

  Error RegisterSystemSharedMemory(
      const std::string& name, const std::string& key,
      const size_t byte_size) override
  {
    stats_->memory_stats.num_register_system_shared_memory_calls++;
    return Error::Success;
  }

  Error RegisterCudaSharedMemory(
      const std::string& name, const cudaIpcMemHandle_t& handle,
      const size_t byte_size) override
  {
    stats_->memory_stats.num_register_cuda_shared_memory_calls++;
    return Error::Success;
  }

  Error RegisterCudaMemory(
      const std::string& name, void* handle, const size_t byte_size) override
  {
    stats_->memory_stats.num_register_cuda_memory_calls++;
    return Error::Success;
  }

  Error RegisterSystemMemory(
      const std::string& name, void* memory_ptr,
      const size_t byte_size) override
  {
    stats_->memory_stats.num_register_system_memory_calls++;
    return Error::Success;
  }

  Error CreateSharedMemoryRegion(
      std::string shm_key, size_t byte_size, int* shm_fd) override
  {
    stats_->memory_stats.num_create_shared_memory_region_calls++;
    return Error::Success;
  }

  Error MapSharedMemory(
      int shm_fd, size_t offset, size_t byte_size, void** shm_addr) override
  {
    stats_->memory_stats.num_map_shared_memory_calls++;
    return Error::Success;
  }

  Error CloseSharedMemory(int shm_fd) override
  {
    stats_->memory_stats.num_close_shared_memory_calls++;
    return Error::Success;
  }

  Error UnlinkSharedMemoryRegion(std::string shm_key) override
  {
    stats_->memory_stats.num_unlink_shared_memory_region_calls++;
    return Error::Success;
  }

  Error UnmapSharedMemory(void* shm_addr, size_t byte_size) override
  {
    stats_->memory_stats.num_unmap_shared_memory_calls++;
    return Error::Success;
  }

 private:
  void LaunchAsyncMockRequest(const InferOptions options, OnCompleteFn callback)
  {
    std::thread([this, options, callback]() {
      std::this_thread::sleep_for(stats_->GetNextDelay());
      local_completed_req_count_++;

      InferResult* result = new MockInferResult(options);
      callback(result);

      stats_->num_active_infer_calls--;
    })
        .detach();
  }

  // Total count of how many requests this client has handled and finished
  size_t local_completed_req_count_ = 0;

  std::shared_ptr<MockClientStats> stats_;
  OnCompleteFn stream_callback_;
};

/// Mock factory that always creates a MockClientBackend instead
/// of a real backend
///
class MockClientBackendFactory : public ClientBackendFactory {
 public:
  MockClientBackendFactory(std::shared_ptr<MockClientStats> stats)
  {
    stats_ = stats;
  }

  Error CreateClientBackend(std::unique_ptr<ClientBackend>* backend) override
  {
    std::unique_ptr<MockClientBackend> mock_backend(
        new MockClientBackend(stats_));
    *backend = std::move(mock_backend);
    return Error::Success;
  }

 private:
  std::shared_ptr<MockClientStats> stats_;
};

}}}  // namespace triton::perfanalyzer::clientbackend
