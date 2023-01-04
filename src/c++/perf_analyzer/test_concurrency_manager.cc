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
      bool is_decoupled_model = false)
      : TestLoadManagerBase(params, is_sequence_model, is_decoupled_model),
        ConcurrencyManager(
            params.async, params.streaming, params.batch_size,
            params.max_threads, params.max_concurrency, params.sequence_length,
            params.shared_memory_type, params.output_shm_size,
            params.start_sequence_id, params.sequence_id_range, GetParser(),
            GetFactory())
  {
    InitManager(
        params.string_length, params.string_data, params.zero_input,
        params.user_data);
  }

  /// Test that the correct Infer function is called in the backend
  ///
  void TestInferType()
  {
    // FIXME TMA-982: This delay is to avoid deadlock. Investigate why delay is
    // needed.
    stats_->response_delay = std::chrono::milliseconds(50);

    ChangeConcurrencyLevel(params_.max_concurrency);

    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    CheckInferType();
  }

  /// Test that the correct concurrency is maintained in the load manager
  ///
  void TestConcurrency(
      std::chrono::milliseconds response_delay,
      std::chrono::milliseconds sleep_time)
  {
    stats_->response_delay = response_delay;

    ChangeConcurrencyLevel(params_.max_concurrency);
    std::this_thread::sleep_for(sleep_time);

    CheckConcurrency();
  }

  /// Test sequence handling
  ///
  void TestSequences()
  {
    int delay_ms = 10;
    stats_->response_delay = std::chrono::milliseconds(delay_ms);

    auto stats = cb::InferStat();
    double concurrency1 = params_.max_concurrency / 2;
    double concurrency2 = params_.max_concurrency;
    int sleep_ms = 500;

    auto sleep_time = std::chrono::milliseconds(sleep_ms);
    size_t expected_count1 = sleep_ms * concurrency1 / delay_ms;
    size_t expected_count2 =
        sleep_ms * concurrency2 / delay_ms + expected_count1;

    // Run and check request rate 1
    //
    ChangeConcurrencyLevel(concurrency1);
    std::this_thread::sleep_for(sleep_time);

    stats = cb::InferStat();
    GetAccumulatedClientStat(&stats);
    CHECK(
        stats.completed_request_count ==
        doctest::Approx(expected_count1).epsilon(0.10));

    PauseSequenceWorkers();
    CheckSequences(concurrency1);

    // Make sure that the client and the manager are in agreement on the request
    // count in between rates
    //
    stats = cb::InferStat();
    GetAccumulatedClientStat(&stats);
    int client_total_requests = stats_->num_async_infer_calls +
                                stats_->num_async_stream_infer_calls +
                                stats_->num_infer_calls;
    CHECK(stats.completed_request_count == client_total_requests);

    ResetStats();

    // Run and check request rate 2
    //
    ChangeConcurrencyLevel(concurrency2);
    std::this_thread::sleep_for(sleep_time);

    stats = cb::InferStat();
    GetAccumulatedClientStat(&stats);
    CHECK(
        stats.completed_request_count ==
        doctest::Approx(expected_count2).epsilon(0.10));

    // Stop all threads and make sure everything is as expected
    //
    StopWorkerThreads();

    CheckSequences(concurrency2);
  }

 private:
  void CheckConcurrency()
  {
    if (params_.max_concurrency < 4) {
      CHECK(stats_->num_active_infer_calls == params_.max_concurrency);
    } else {
      CHECK(
          stats_->num_active_infer_calls ==
          doctest::Approx(params_.max_concurrency).epsilon(0.25));
    }
  }
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
  std::chrono::milliseconds response_delay{50};
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
  tcm.TestConcurrency(response_delay, sleep_time);
}

/// Check that the inference requests for sequences follow all rules and
/// parameters
///
TEST_CASE("concurrency_sequence")
{
  PerfAnalyzerParameters params = TestLoadManagerBase::GetSequenceTestParams();
  const bool is_sequence_model{true};
  TestConcurrencyManager tcm(params, is_sequence_model);
  tcm.TestSequences();
}

}}  // namespace triton::perfanalyzer
