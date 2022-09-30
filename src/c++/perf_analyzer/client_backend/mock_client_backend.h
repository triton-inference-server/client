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

#include "client_backend.h"
#include <chrono>

namespace triton { namespace perfanalyzer { namespace clientbackend {

/// Mock class of an InferResult
///
class MockInferResult : public InferResult {
  public: 
    MockInferResult(const InferOptions& options): req_id_{options.request_id_} {}

    Error Id(std::string* id) const override { *id = req_id_; return Error::Success; }
    Error RequestStatus() const override { return Error::Success; }
    Error RawData(const std::string& output_name, const uint8_t** buf, size_t* byte_size) const override { return Error::Success; }
  
  private:
    std::string req_id_;
};

/// Class to track statistics of MockClientBackend
/// 
class MockClientStats {
  public:
    enum class ReqType {SYNC, ASYNC, ASYNC_STREAM};

    struct SeqStatus {
      bool live = false;
    };

    size_t num_infer_calls{0};
    size_t num_async_infer_calls{0};
    size_t num_async_stream_infer_calls{0};
    size_t num_start_stream_calls{0};

    std::vector<std::chrono::time_point<std::chrono::system_clock>> request_timestamps;
    std::map<uint64_t, SeqStatus> sequence_statuses;

    void CaptureRequest(ReqType type, 
                        const InferOptions& options,
                        const std::vector<InferInput*>& inputs,
                        const std::vector<const InferRequestedOutput*>& outputs) {

      std::lock_guard<std::mutex> lock(mtx_);
      auto time = std::chrono::system_clock::now();
      request_timestamps.push_back(time);

      UpdateCallCount(type);
      UpdateSeqStatus(options);
    }   

    void Reset() {
      num_infer_calls = 0;
      num_async_infer_calls = 0;
      num_async_stream_infer_calls = 0;
      num_start_stream_calls = 0;
      request_timestamps.clear();
    }

  private:    
    std::mutex mtx_;

    void UpdateCallCount(ReqType type) {
      if (type == ReqType::SYNC) {
        num_infer_calls++;
      }
      else if (type == ReqType::ASYNC) {
        num_async_infer_calls++;
      }
      else {
        num_async_stream_infer_calls++;
      }
    }

    void UpdateSeqStatus(const InferOptions& options) {
      if (options.sequence_id_ != 0) {
        SeqStatus& status = sequence_statuses[options.sequence_id_];

        // FIXME one more check around starting a live seq
        // change to assert(?)
        if (status.live == false) {
          CHECK(options.sequence_start_ == true);
          status.live = true;
        }
        if (options.sequence_end_) {
          CHECK(status.live == true);
          status.live = false;
        }
      }
    }
};

/// Mock implementation of ClientBackend interface
///
class MockClientBackend : public ClientBackend {
  public:
    MockClientBackend(std::shared_ptr<MockClientStats> stats) {
      stats_ = stats;
    }

    Error Infer(InferResult** result, 
                    const InferOptions& options, 
                    const std::vector<InferInput*>& inputs, 
                    const std::vector<const InferRequestedOutput*>& outputs) override 
    {
      stats_->CaptureRequest(MockClientStats::ReqType::SYNC, options, inputs, outputs);
      return Error::Success;
    }

    Error AsyncInfer(OnCompleteFn callback, 
                         const InferOptions& options,
                         const std::vector<InferInput*>& inputs,
                         const std::vector<const InferRequestedOutput*>& outputs) override {
      stats_->CaptureRequest(MockClientStats::ReqType::ASYNC, options, inputs, outputs);
      InferResult* result = new MockInferResult(options);

      callback(result);
      return Error::Success;
    }

    Error AsyncStreamInfer(const InferOptions& options, 
                               const std::vector<InferInput*>& inputs,
                               const std::vector<const InferRequestedOutput*>& outputs) {
      stats_->CaptureRequest(MockClientStats::ReqType::ASYNC_STREAM, options, inputs, outputs);
      InferResult* result = new MockInferResult(options);
      stream_callback_(result);
      return Error::Success;
    }

    Error StartStream(OnCompleteFn callback, bool enable_stats) {
      // FIXME don't reach into class. Call fn, use mutex
      stats_->num_start_stream_calls++;
      stream_callback_ = callback;
      return Error::Success;
    }

    Error ClientInferStat(InferStat* a) override 
    {
      return Error::Success;
    }

  private:
    std::shared_ptr<MockClientStats> stats_;
    OnCompleteFn stream_callback_;
};

/// Mock factory that always creates a MockClientBackend instead
/// of a real backend
///
class MockClientBackendFactory : public ClientBackendFactory {
  public:
    MockClientBackendFactory(std::shared_ptr<MockClientStats> stats) {
      stats_ = stats;
    }

    Error CreateClientBackend(std::unique_ptr<ClientBackend>* backend) override {
      std::unique_ptr<MockClientBackend> mock_backend(new MockClientBackend(stats_));
      *backend = std::move(mock_backend);
      return Error::Success;
    }
  private:
    std::shared_ptr<MockClientStats> stats_;
};

}}}
