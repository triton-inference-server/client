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

#include "command_line_parser.h"
#include "concurrency_manager.h"
#include "doctest.h"
#include "mock_client_backend.h"
#include "mock_model_parser.h"
#include "test_load_manager_base.h"

namespace triton { namespace perfanalyzer {

class TestConcurrencyManager : public TestLoadManagerBase,
                               public ConcurrencyManager {
 public:
  TestConcurrencyManager(
      PerfAnalyzerParameters params, bool is_sequence_model = false,
      bool use_mock_infer = false)
      : use_mock_infer_(use_mock_infer),
        TestLoadManagerBase(params, is_sequence_model),
        ConcurrencyManager(
            params.async, params.streaming, params.batch_size,
            params.max_threads, params.max_concurrency, params.sequence_length,
            params.shared_memory_type, params.output_shm_size,
            params.start_sequence_id, params.sequence_id_range,
            params.string_length, params.string_data, params.zero_input,
            params.user_data, GetParser(), GetFactory())
  {
  }

  void Infer(
      std::shared_ptr<ThreadStat> thread_stat,
      std::shared_ptr<ThreadConfig> thread_config) override
  {
    if (use_mock_infer_) {
      return MockInfer(thread_stat, thread_config);
    } else {
      return ConcurrencyManager::Infer(thread_stat, thread_config);
    }
  }

  // Mock out most of the complicated Infer code
  //
  void MockInfer(
      std::shared_ptr<ThreadStat> thread_stat,
      std::shared_ptr<ThreadConfig> thread_config)
  {
    if (!execute_) {
      thread_config->is_paused_ = true;
    }
  }

  /// Test that the correct Infer function is called in the backend
  ///
  void TestInferType()
  {
    // FIXME TMA-982: This delay is to avoid deadlock. Investigate why delay is
    // needed.
    stats_->request_delay = std::chrono::milliseconds(50);

    ChangeConcurrencyLevel(params_.max_concurrency);

    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    CheckInferType();
  }

  /// Test that the correct concurrency is maintained in the load manager
  ///
  void TestConcurrency(
      std::chrono::milliseconds request_delay,
      std::chrono::milliseconds sleep_time)
  {
    stats_->request_delay = request_delay;

    ChangeConcurrencyLevel(params_.max_concurrency);
    std::this_thread::sleep_for(sleep_time);

    CheckConcurrency();
  }

  /// Test sequence handling
  ///
  void TestSequences()
  {
    auto sleep_time = std::chrono::milliseconds(500);

    ChangeConcurrencyLevel(params_.max_concurrency);
    std::this_thread::sleep_for(sleep_time);

    CheckSequences();
  }

 private:
  void CheckConcurrency()
  {
    if (params_.max_concurrency < 4) {
      CHECK(stats_->num_active_infer_calls.load() == params_.max_concurrency);
    } else {
      CHECK(
          stats_->num_active_infer_calls.load() ==
          doctest::Approx(params_.max_concurrency).epsilon(0.25));
    }
  }

  const bool use_mock_infer_{false};
};

/// Test that the correct Infer function is called in the backend
///
TEST_CASE("concurrency_infer_type")
{
  PerfAnalyzerParameters params{};
  params.max_concurrency = 1;

  SUBCASE("async_streaming")
  {
    params.async = true;
    params.streaming = true;
  }
  SUBCASE("async_no_streaming")
  {
    params.async = true;
    params.streaming = false;
  }
  SUBCASE("no_async_streaming")
  {
    params.async = false;
    params.streaming = true;
  }
  SUBCASE("no_async_no_streaming")
  {
    params.async = false;
    params.streaming = false;
  }

  TestConcurrencyManager tcm(params);
  tcm.TestInferType();
}

/// Test that the correct concurrency is maintained in the load manager
///
TEST_CASE("concurrency_concurrency")
{
  PerfAnalyzerParameters params{};
  std::chrono::milliseconds request_delay{50};
  std::chrono::milliseconds sleep_time{225};

  SUBCASE("sync, no-streaming, 1 concurrency, 1 thread")
  {
    params.forced_sync = true;
    params.async = false;
    params.streaming = false;
    params.max_concurrency = 1;
    params.max_threads = 1;
  }

  SUBCASE("sync, no-streaming, 4 concurrency, 4 threads")
  {
    params.forced_sync = true;
    params.async = false;
    params.streaming = false;
    params.max_concurrency = 4;
    params.max_threads = 4;
  }

  SUBCASE("async, no-streaming, 1 concurrency, 1 thread")
  {
    params.forced_sync = false;
    params.async = true;
    params.streaming = false;
    params.max_concurrency = 1;
    params.max_threads = 1;
  }

  SUBCASE("async, no-streaming, 4 concurrency, 1 thread")
  {
    params.forced_sync = false;
    params.async = true;
    params.streaming = false;
    params.max_concurrency = 4;
    params.max_threads = 1;
  }

  SUBCASE("async, no-streaming, 4 concurrency, 2 threads")
  {
    params.forced_sync = false;
    params.async = true;
    params.streaming = false;
    params.max_concurrency = 4;
    params.max_threads = 2;
  }

  SUBCASE("async, no-streaming, 4 concurrency, 4 threads")
  {
    params.forced_sync = false;
    params.async = true;
    params.streaming = false;
    params.max_concurrency = 4;
    params.max_threads = 4;
  }

  SUBCASE("async, streaming, 1 concurrency, 1 thread")
  {
    params.forced_sync = false;
    params.async = true;
    params.streaming = true;
    params.max_concurrency = 1;
    params.max_threads = 1;
  }

  SUBCASE("async, streaming, 4 concurrency, 1 thread")
  {
    params.forced_sync = false;
    params.async = true;
    params.streaming = true;
    params.max_concurrency = 4;
    params.max_threads = 1;
  }

  SUBCASE("async, streaming, 4 concurrency, 2 threads")
  {
    params.forced_sync = false;
    params.async = true;
    params.streaming = true;
    params.max_concurrency = 4;
    params.max_threads = 2;
  }

  SUBCASE("async, streaming, 4 concurrency, 4 threads")
  {
    params.forced_sync = false;
    params.async = true;
    params.streaming = true;
    params.max_concurrency = 4;
    params.max_threads = 4;
  }

  TestConcurrencyManager tcm(params);
  tcm.TestConcurrency(request_delay, sleep_time);
}

/// Check that the inference requests for sequences follow all rules and
/// parameters
///
TEST_CASE("concurrency_sequence")
{
  PerfAnalyzerParameters params{};

  // Generally we want short sequences for testing
  // so we can hit the corner cases more often
  //
  params.sequence_length = 3;

  SUBCASE("Normal") {}
  SUBCASE("sequence IDs 1")
  {
    params.start_sequence_id = 1;
    params.sequence_id_range = 5;
  }
  SUBCASE("sequence IDs 2")
  {
    params.start_sequence_id = 17;
    params.sequence_id_range = 8;
  }
  SUBCASE("num_of_sequences 1") { params.num_of_sequences = 1; }
  SUBCASE("num_of_sequences 10")
  {
    params.num_of_sequences = 10;
    // Make sequences longer so we actually get 10 in flight at a time
    params.sequence_length = 8;
  }
  SUBCASE("sequence_length 1") { params.sequence_length = 1; }
  SUBCASE("sequence_length 10") { params.sequence_length = 10; }

  params.max_concurrency = params.num_of_sequences;

  const bool is_sequence_model{true};
  TestConcurrencyManager tcm(params, is_sequence_model);
  tcm.TestSequences();
}

}}  // namespace triton::perfanalyzer
