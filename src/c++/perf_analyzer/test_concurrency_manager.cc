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

namespace triton { namespace perfanalyzer {

class TestConcurrencyManager {
 public:
  TestConcurrencyManager()
  {
    // Must reset this global flag every unit test.
    // It gets set to true when we deconstruct any load manager
    // (aka every unit test)
    //
    early_exit = false;

    stats_ = std::make_shared<cb::MockClientStats>();
  }

  /// Test that the correct Infer function is called in the backend
  ///
  void TestInferType(const PerfAnalyzerParameters& params)
  {
    auto sleep_time = std::chrono::milliseconds(500);

    std::unique_ptr<LoadManager> manager = CreateManager(params);
    dynamic_cast<ConcurrencyManager*>(manager.get())
        ->ChangeConcurrencyLevel(params.max_concurrency);
    std::this_thread::sleep_for(sleep_time);

    CheckInferType(params);
  }

  /// Test that the correct concurrency is maintained in the load manager
  ///
  void TestConcurrency(
      const PerfAnalyzerParameters& params,
      std::chrono::milliseconds request_delay,
      std::chrono::milliseconds sleep_time)
  {
    stats_->request_delay = request_delay;

    std::shared_ptr<LoadManager> manager = CreateManager(params);
    dynamic_cast<ConcurrencyManager*>(manager.get())
        ->ChangeConcurrencyLevel(params.max_concurrency);
    std::this_thread::sleep_for(sleep_time);

    CheckConcurrency(params);
  }

  /// Test sequence handling
  ///
  void TestSequences(const PerfAnalyzerParameters& params)
  {
    bool is_sequence_model = true;
    std::unique_ptr<LoadManager> manager =
        CreateManager(params, is_sequence_model);

    auto sleep_time = std::chrono::milliseconds(500);

    dynamic_cast<ConcurrencyManager*>(manager.get())
        ->ChangeConcurrencyLevel(params.max_concurrency);
    std::this_thread::sleep_for(sleep_time);

    CheckSequences(params);
  }

 private:
  std::shared_ptr<cb::MockClientStats> stats_;

  std::shared_ptr<cb::MockClientStats> GetStats() { return stats_; }

  void ResetStats() { stats_->Reset(); }

  void CheckInferType(const PerfAnalyzerParameters& params)
  {
    auto stats = GetStats();

    if (params.async) {
      if (params.streaming) {
        CHECK(stats->num_infer_calls == 0);
        CHECK(stats->num_async_infer_calls == 0);
        CHECK(stats->num_async_stream_infer_calls > 0);
        CHECK(stats->num_start_stream_calls > 0);
      } else {
        CHECK(stats->num_infer_calls == 0);
        CHECK(stats->num_async_infer_calls > 0);
        CHECK(stats->num_async_stream_infer_calls == 0);
        CHECK(stats->num_start_stream_calls == 0);
      }
    } else {
      if (params.streaming) {
        CHECK(stats->num_infer_calls > 0);
        CHECK(stats->num_async_infer_calls == 0);
        CHECK(stats->num_async_stream_infer_calls == 0);
        CHECK(stats->num_start_stream_calls > 0);
      } else {
        CHECK(stats->num_infer_calls > 0);
        CHECK(stats->num_async_infer_calls == 0);
        CHECK(stats->num_async_stream_infer_calls == 0);
        CHECK(stats->num_start_stream_calls == 0);
      }
    }
  }

  void CheckConcurrency(const PerfAnalyzerParameters& params)
  {
    auto stats = GetStats();

    if (params.max_concurrency < 4) {
      CHECK(stats->num_active_infer_calls.load() == params.max_concurrency);
    } else {
      CHECK(
          stats->num_active_infer_calls.load() ==
          doctest::Approx(params.max_concurrency).epsilon(0.25));
    }
  }

  void CheckSequences(const PerfAnalyzerParameters& params)
  {
    auto stats = GetStats();

    // Make sure all seq IDs are within range
    //
    for (auto seq_id : stats->sequence_status.used_seq_ids) {
      CHECK(seq_id >= params.start_sequence_id);
      CHECK(seq_id <= params.start_sequence_id + params.sequence_id_range);
    }

    // Make sure that we had the correct number of concurrently live sequences
    //
    // If the sequence length is only 1 then there is nothing to check because
    // there are never any overlapping requests -- they always immediately exit
    //
    if (params.sequence_length != 1) {
      CHECK(
          stats->sequence_status.max_live_seq_count == params.max_concurrency);
    }

    // Make sure that the length of each sequence is as expected
    // (The code explicitly has a 20% slop, so that is what we are checking)
    //
    for (auto len : stats->sequence_status.seq_lengths) {
      CHECK(len == doctest::Approx(params.sequence_length).epsilon(0.20));
    }
  }

  std::unique_ptr<LoadManager> CreateManager(
      PerfAnalyzerParameters params, bool is_sequence_model = false)
  {
    std::unique_ptr<LoadManager> manager;
    std::shared_ptr<cb::ClientBackendFactory> factory =
        std::make_shared<cb::MockClientBackendFactory>(stats_);
    std::shared_ptr<ModelParser> parser =
        std::make_shared<MockModelParser>(is_sequence_model);

    ConcurrencyManager::Create(
        params.async, params.streaming, params.batch_size, params.max_threads,
        params.max_concurrency, params.sequence_length, params.string_length,
        params.string_data, params.zero_input, params.user_data,
        params.shared_memory_type, params.output_shm_size,
        params.start_sequence_id, params.sequence_id_range, parser, factory,
        &manager);

    return manager;
  }
};

/// Test that the correct concurrency is maintained in the load manager
///
TEST_CASE("concurrency_infer_type")
{
  TestConcurrencyManager tcm{};
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

  tcm.TestInferType(params);
}

/// Check that the corre
/// are used given different param values for async and stream
///
TEST_CASE("concurrency_concurrency")
{
  TestConcurrencyManager tcm{};
  PerfAnalyzerParameters params{};
  std::chrono::milliseconds request_delay{50};
  std::chrono::milliseconds sleep_time{225};

  SUBCASE("")
  {
    params.forced_sync = true;
    params.async = false;
    params.streaming = false;
    params.max_concurrency = 1;
    params.max_threads = 1;
  }

  SUBCASE("")
  {
    params.forced_sync = true;
    params.async = false;
    params.streaming = false;
    params.max_concurrency = 4;
    params.max_threads = 4;
  }

  SUBCASE("")
  {
    params.forced_sync = false;
    params.async = true;
    params.streaming = false;
    params.max_concurrency = 1;
    params.max_threads = 1;
  }

  SUBCASE("")
  {
    params.forced_sync = false;
    params.async = true;
    params.streaming = false;
    params.max_concurrency = 4;
    params.max_threads = 1;
  }

  SUBCASE("")
  {
    params.forced_sync = false;
    params.async = true;
    params.streaming = false;
    params.max_concurrency = 4;
    params.max_threads = 2;
  }

  SUBCASE("")
  {
    params.forced_sync = false;
    params.async = true;
    params.streaming = false;
    params.max_concurrency = 4;
    params.max_threads = 4;
  }

  SUBCASE("")
  {
    params.forced_sync = false;
    params.async = true;
    params.streaming = true;
    params.max_concurrency = 1;
    params.max_threads = 1;
  }

  SUBCASE("")
  {
    params.forced_sync = false;
    params.async = true;
    params.streaming = true;
    params.max_concurrency = 4;
    params.max_threads = 1;
  }

  SUBCASE("")
  {
    params.forced_sync = false;
    params.async = true;
    params.streaming = true;
    params.max_concurrency = 4;
    params.max_threads = 2;
  }

  SUBCASE("")
  {
    params.forced_sync = false;
    params.async = true;
    params.streaming = true;
    params.max_concurrency = 4;
    params.max_threads = 4;
  }

  tcm.TestConcurrency(params, request_delay, sleep_time);
}

/// Check that the inference requests for sequences
/// follow all rules and parameters
///
TEST_CASE("concurrency_sequence")
{
  TestConcurrencyManager tcm{};
  PerfAnalyzerParameters params{};

  params.max_concurrency = 1;

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

  tcm.TestSequences(params);
}

}}  // namespace triton::perfanalyzer
