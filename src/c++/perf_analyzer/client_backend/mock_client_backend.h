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

#include <chrono>
#include "client_backend.h"

namespace triton { namespace perfanalyzer { namespace clientbackend {

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

    void HandleSeqRequest(uint64_t seq_id) { live_seq_ids_to_length[seq_id]++; }
  };

  size_t num_infer_calls{0};
  size_t num_async_infer_calls{0};
  size_t num_async_stream_infer_calls{0};
  size_t num_start_stream_calls{0};

  std::vector<std::chrono::time_point<std::chrono::system_clock>>
      request_timestamps;
  SeqStatus sequence_status;

  void CaptureRequest(
      ReqType type, const InferOptions& options,
      const std::vector<InferInput*>& inputs,
      const std::vector<const InferRequestedOutput*>& outputs)
  {
    std::lock_guard<std::mutex> lock(mtx_);
    auto time = std::chrono::system_clock::now();
    request_timestamps.push_back(time);

    UpdateCallCount(type);
    UpdateSeqStatus(options);
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
  }

 private:
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
};

/// Mock implementation of ClientBackend interface
///
class MockClientBackend : public ClientBackend {
 public:
  MockClientBackend(std::shared_ptr<MockClientStats> stats) { stats_ = stats; }

  Error Infer(
      InferResult** result, const InferOptions& options,
      const std::vector<InferInput*>& inputs,
      const std::vector<const InferRequestedOutput*>& outputs) override
  {
    stats_->CaptureRequest(
        MockClientStats::ReqType::SYNC, options, inputs, outputs);
    return Error::Success;
  }

  Error AsyncInfer(
      OnCompleteFn callback, const InferOptions& options,
      const std::vector<InferInput*>& inputs,
      const std::vector<const InferRequestedOutput*>& outputs) override
  {
    stats_->CaptureRequest(
        MockClientStats::ReqType::ASYNC, options, inputs, outputs);
    InferResult* result = new MockInferResult(options);

    callback(result);
    return Error::Success;
  }

  Error AsyncStreamInfer(
      const InferOptions& options, const std::vector<InferInput*>& inputs,
      const std::vector<const InferRequestedOutput*>& outputs)
  {
    stats_->CaptureRequest(
        MockClientStats::ReqType::ASYNC_STREAM, options, inputs, outputs);
    InferResult* result = new MockInferResult(options);
    stream_callback_(result);
    return Error::Success;
  }

  Error StartStream(OnCompleteFn callback, bool enable_stats)
  {
    stats_->CaptureStreamStart();
    stream_callback_ = callback;
    return Error::Success;
  }

  Error ClientInferStat(InferStat* a) override { return Error::Success; }

 private:
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
