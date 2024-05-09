// Copyright 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <future>
#include <memory>

#include "command_line_parser.h"
#include "concurrency_manager.h"
#include "doctest.h"
#include "mock_client_backend.h"
#include "mock_concurrency_worker.h"
#include "mock_data_loader.h"
#include "mock_infer_data_manager.h"
#include "mock_model_parser.h"
#include "mock_sequence_manager.h"
#include "sequence_manager.h"
#include "test_load_manager_base.h"
#include "test_utils.h"

namespace triton { namespace perfanalyzer {

class TestConcurrencyManager : public TestLoadManagerBase,
                               public ConcurrencyManager {
 public:
  TestConcurrencyManager(
      PerfAnalyzerParameters params, bool is_sequence_model = false,
      bool is_decoupled_model = false, bool use_mock_infer = false)
      : use_mock_infer_(use_mock_infer),
        TestLoadManagerBase(params, is_sequence_model, is_decoupled_model),
        ConcurrencyManager(
            params.async, params.streaming, params.batch_size,
            params.max_threads, params.max_concurrency,
            params.shared_memory_type, params.output_shm_size, GetParser(),
            GetFactory(), params.request_parameters)
  {
  }

  std::shared_ptr<IWorker> MakeWorker(
      std::shared_ptr<ThreadStat> thread_stat,
      std::shared_ptr<ThreadConfig> thread_config) override
  {
    size_t id = workers_.size();

    auto worker = std::make_shared<MockConcurrencyWorker>(
        id, thread_stat, thread_config, parser_, data_loader_, factory_,
        on_sequence_model_, async_, max_concurrency_, using_json_data_,
        streaming_, batch_size_, wake_signal_, wake_mutex_, active_threads_,
        execute_, infer_data_manager_, sequence_manager_);

    if (use_mock_infer_) {
      EXPECT_CALL(*worker, Infer())
          .WillRepeatedly(testing::Invoke(
              worker.get(), &MockConcurrencyWorker::EmptyInfer));
    }
    return worker;
  }


  void TestReconfigThreads(
      const size_t concurrent_request_count, const size_t num_requests,
      std::vector<ThreadConfig>& expected_configs)
  {
    ConcurrencyManager::ReconfigThreads(concurrent_request_count, num_requests);

    auto expected_size = expected_configs.size();

    // Check that the correct number of threads are created
    //
    CHECK(threads_.size() == expected_size);

    // Check that threads_config has correct concurrency and seq stat index
    // offset
    for (auto i = 0; i < expected_configs.size(); i++) {
      CHECK(
          threads_config_[i]->concurrency_ == expected_configs[i].concurrency_);
      CHECK(
          threads_config_[i]->seq_stat_index_offset_ ==
          expected_configs[i].seq_stat_index_offset_);
      CHECK(
          threads_config_[i]->num_requests_ ==
          expected_configs[i].num_requests_);
    }
  }

  void StopWorkerThreads() { LoadManager::StopWorkerThreads(); }

  /// Test that the correct Infer function is called in the backend
  ///
  void TestInferType()
  {
    // FIXME TMA-982: This delay is to avoid deadlock. Investigate why delay is
    // needed.
    stats_->SetDelays({50});

    ChangeConcurrencyLevel(params_.max_concurrency);

    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    CheckInferType();
  }

  /// Test that the correct concurrency is maintained in the load manager
  ///
  void TestConcurrency(
      size_t response_delay, std::chrono::milliseconds sleep_time)
  {
    stats_->SetDelays({response_delay});

    ChangeConcurrencyLevel(params_.max_concurrency);
    std::this_thread::sleep_for(sleep_time);

    CheckConcurrency();
  }

  /// Test sequence handling
  ///
  void TestSequences()
  {
    size_t delay_ms = 10;
    stats_->SetDelays({delay_ms});

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

  /// Test that tries to find deadlocks and livelocks
  ///
  void TestTimeouts()
  {
    TestWatchDog watchdog(1000);
    ChangeConcurrencyLevel(params_.max_concurrency);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    StopWorkerThreads();
    watchdog.stop();
  }

  /// Test that idle time is tracked correctly
  void TestOverhead()
  {
    stats_->SetDelays({1});
    ChangeConcurrencyLevel(params_.max_concurrency);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    // During a run of 100 ms (100,000,000 ns), make sure that the idle time is
    // at least 95% of that
    //
    auto idle_time_ns = GetIdleTime();
    CHECK(idle_time_ns > 95000000);
    StopWorkerThreads();
  }

  std::shared_ptr<ModelParser>& parser_{LoadManager::parser_};
  std::shared_ptr<DataLoader>& data_loader_{LoadManager::data_loader_};
  std::shared_ptr<SequenceManager>& sequence_manager_{
      LoadManager::sequence_manager_};
  bool& using_json_data_{LoadManager::using_json_data_};
  bool& execute_{ConcurrencyManager::execute_};
  size_t& batch_size_{LoadManager::batch_size_};
  size_t& max_threads_{LoadManager::max_threads_};
  std::shared_ptr<cb::ClientBackendFactory> factory_{
      TestLoadManagerBase::factory_};
  std::shared_ptr<IInferDataManager>& infer_data_manager_{
      LoadManager::infer_data_manager_};

 private:
  bool use_mock_infer_{false};

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


  std::shared_ptr<SequenceManager> MakeSequenceManager(
      const uint64_t start_sequence_id, const uint64_t sequence_id_range,
      const size_t sequence_length, const bool sequence_length_specified,
      const double sequence_length_variation, const bool using_json_data,
      std::shared_ptr<DataLoader> data_loader) override
  {
    return std::make_shared<MockSequenceManager>(
        start_sequence_id, sequence_id_range, sequence_length,
        sequence_length_specified, sequence_length_variation, using_json_data,
        data_loader);
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

  tcm.InitManager(
      params.string_length, params.string_data, params.zero_input,
      params.user_data, params.start_sequence_id, params.sequence_id_range,
      params.sequence_length, params.sequence_length_specified,
      params.sequence_length_variation);

  tcm.TestInferType();
}

/// Test that the correct concurrency is maintained in the load manager
///
TEST_CASE("concurrency_concurrency")
{
  PerfAnalyzerParameters params{};
  size_t response_delay{50};
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

  tcm.InitManager(
      params.string_length, params.string_data, params.zero_input,
      params.user_data, params.start_sequence_id, params.sequence_id_range,
      params.sequence_length, params.sequence_length_specified,
      params.sequence_length_variation);

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

  tcm.InitManager(
      params.string_length, params.string_data, params.zero_input,
      params.user_data, params.start_sequence_id, params.sequence_id_range,
      params.sequence_length, params.sequence_length_specified,
      params.sequence_length_variation);
  tcm.TestSequences();
}

/// Create the case where the sequences do NOT go round robin due to
/// the first request taking longer than the rest.
///
/// This exposed a bug where we were constantly resetting ctx IDs
/// and issuing over and over again to the first sequence even though
/// it was the only sequence that should NOT be issued because it was
/// still outstanding
///
TEST_CASE("concurrency_free_ctx_ids")
{
  PerfAnalyzerParameters params{};
  params.async = true;
  params.streaming = true;
  params.max_concurrency = 6;

  bool is_sequence_model{true};


  TestConcurrencyManager tcm(params, is_sequence_model);


  tcm.InitManager(
      params.string_length, params.string_data, params.zero_input,
      params.user_data, params.start_sequence_id, params.sequence_id_range,
      params.sequence_length, params.sequence_length_specified,
      params.sequence_length_variation);

  // Have the first request (sequence ID 1) take very long, and all the other
  // requests are fast
  //
  tcm.stats_->SetDelays({50, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5});

  std::shared_ptr<ThreadStat> thread_stat{std::make_shared<ThreadStat>()};
  std::shared_ptr<ThreadConfig> thread_config{
      std::make_shared<ThreadConfig>(0)};
  thread_config->concurrency_ = 4;

  std::shared_ptr<IWorker> worker{tcm.MakeWorker(thread_stat, thread_config)};

  std::future<void> infer_future{std::async(&IWorker::Infer, worker)};

  std::this_thread::sleep_for(std::chrono::milliseconds(15));

  early_exit = true;
  infer_future.get();

  // The first sequence should only be called two times, once at the very start,
  // and once during shutdown
  //
  CHECK(tcm.stats_->sequence_status.seq_ids_to_count.at(1) == 2);
}

TEST_CASE("Concurrency - shared memory infer input calls")
{
  PerfAnalyzerParameters params{};
  params.max_concurrency = 4;
  bool is_sequence_model{false};

  const auto& ParameterizeAsyncAndStreaming{[&]() {
    SUBCASE("sync non-streaming")
    {
      params.async = false;
      params.streaming = false;
    }
    SUBCASE("async non-streaming")
    {
      params.async = true;
      params.streaming = false;
    }
    SUBCASE("async streaming")
    {
      params.async = true;
      params.streaming = true;
    }
  }};

  const auto& ParameterizeSequence{[&]() {
    SUBCASE("non-sequence")
    {
      is_sequence_model = false;
      ParameterizeAsyncAndStreaming();
    }
    SUBCASE("sequence")
    {
      is_sequence_model = true;
      params.num_of_sequences = 1;
      ParameterizeAsyncAndStreaming();
    }
  }};

  const auto& ParameterizeMemory{[&]() {
    SUBCASE("No shared memory")
    {
      params.shared_memory_type = NO_SHARED_MEMORY;
      ParameterizeSequence();
    }
    SUBCASE("system shared memory")
    {
      params.shared_memory_type = SYSTEM_SHARED_MEMORY;
      ParameterizeSequence();
    }
    SUBCASE("cuda shared memory")
    {
      params.shared_memory_type = CUDA_SHARED_MEMORY;
      ParameterizeSequence();
    }
  }};

  ParameterizeMemory();


  const std::string json_str{R"(
  {
    "data": [
      {
        "INPUT0": [2000000000]
      },
      {
        "INPUT0": [2000000001]
      }
    ]
  }
      )"};

  MockInputPipeline mip =
      TestLoadManagerBase::ProcessCustomJsonData(json_str, is_sequence_model);


  TestConcurrencyManager tcm(params, is_sequence_model);

  tcm.infer_data_manager_ =
      MockInferDataManagerFactory::CreateMockInferDataManager(
          params.max_threads, params.batch_size, params.shared_memory_type,
          params.output_shm_size, params.request_parameters,
          mip.mock_model_parser_, tcm.factory_, mip.mock_data_loader_);

  std::shared_ptr<ThreadStat> thread_stat{std::make_shared<ThreadStat>()};
  std::shared_ptr<ThreadConfig> thread_config{
      std::make_shared<ThreadConfig>(0)};
  thread_config->concurrency_ = 1;

  tcm.parser_ = mip.mock_model_parser_;
  tcm.data_loader_ = mip.mock_data_loader_;
  tcm.using_json_data_ = true;
  tcm.execute_ = true;
  tcm.batch_size_ = 1;
  tcm.max_threads_ = 1;

  tcm.InitManager(
      params.string_length, params.string_data, params.zero_input,
      params.user_data, params.start_sequence_id, params.sequence_id_range,
      params.sequence_length, params.sequence_length_specified,
      params.sequence_length_variation);

  std::shared_ptr<IWorker> worker{tcm.MakeWorker(thread_stat, thread_config)};
  std::future<void> infer_future{std::async(&IWorker::Infer, worker)};

  std::this_thread::sleep_for(std::chrono::milliseconds(18));

  early_exit = true;
  infer_future.get();

  const auto& actual_append_raw_calls{tcm.stats_->num_append_raw_calls};
  const auto& actual_set_shared_memory_calls{
      tcm.stats_->num_set_shared_memory_calls};

  if (params.shared_memory_type == NO_SHARED_MEMORY) {
    CHECK(actual_append_raw_calls > 0);
    CHECK(actual_set_shared_memory_calls == 0);
  } else {
    CHECK(actual_append_raw_calls == 0);
    CHECK(actual_set_shared_memory_calls > 0);
  }
}

/// Verify Shared Memory api calls
///
TEST_CASE("Concurrency - Shared memory methods")
{
  PerfAnalyzerParameters params;
  bool is_sequence = false;
  bool is_decoupled = false;
  bool use_mock_infer = true;

  const std::string json_str{R"(
  {
    "data": [
      {
        "INPUT0": [2123456789]
      }
    ]
  }
      )"};

  MockInputPipeline mip = TestLoadManagerBase::ProcessCustomJsonData(json_str);

  cb::MockClientStats::SharedMemoryStats expected_stats;

  SUBCASE("System shared memory usage")
  {
    params.shared_memory_type = SYSTEM_SHARED_MEMORY;
    TestConcurrencyManager tcm(
        params, is_sequence, is_decoupled, use_mock_infer);

    tcm.infer_data_manager_ =
        MockInferDataManagerFactory::CreateMockInferDataManager(
            params.max_threads, params.batch_size, params.shared_memory_type,
            params.output_shm_size, params.request_parameters,
            mip.mock_model_parser_, tcm.factory_, mip.mock_data_loader_);

    tcm.InitManager(
        params.string_length, params.string_data, params.zero_input,
        params.user_data, params.start_sequence_id, params.sequence_id_range,
        params.sequence_length, params.sequence_length_specified,
        params.sequence_length_variation);

    expected_stats.num_unregister_all_shared_memory_calls = 1;
    expected_stats.num_register_system_shared_memory_calls = 1;
    expected_stats.num_create_shared_memory_region_calls = 1;
    expected_stats.num_map_shared_memory_calls = 1;
    tcm.CheckSharedMemory(expected_stats);
  }

  SUBCASE("Cuda shared memory usage")
  {
    params.shared_memory_type = CUDA_SHARED_MEMORY;
    TestConcurrencyManager tcm(
        params, is_sequence, is_decoupled, use_mock_infer);

    tcm.infer_data_manager_ =
        MockInferDataManagerFactory::CreateMockInferDataManager(
            params.max_threads, params.batch_size, params.shared_memory_type,
            params.output_shm_size, params.request_parameters,
            mip.mock_model_parser_, tcm.factory_, mip.mock_data_loader_);

    tcm.InitManager(
        params.string_length, params.string_data, params.zero_input,
        params.user_data, params.start_sequence_id, params.sequence_id_range,
        params.sequence_length, params.sequence_length_specified,
        params.sequence_length_variation);

    expected_stats.num_unregister_all_shared_memory_calls = 1;
    expected_stats.num_register_cuda_shared_memory_calls = 1;
    tcm.CheckSharedMemory(expected_stats);
  }

  SUBCASE("No shared memory usage")
  {
    params.shared_memory_type = NO_SHARED_MEMORY;
    TestConcurrencyManager tcm(
        params, is_sequence, is_decoupled, use_mock_infer);
    tcm.infer_data_manager_ =
        MockInferDataManagerFactory::CreateMockInferDataManager(
            params.max_threads, params.batch_size, params.shared_memory_type,
            params.output_shm_size, params.request_parameters,
            mip.mock_model_parser_, tcm.factory_, mip.mock_data_loader_);
    tcm.InitManager(
        params.string_length, params.string_data, params.zero_input,
        params.user_data, params.start_sequence_id, params.sequence_id_range,
        params.sequence_length, params.sequence_length_specified,
        params.sequence_length_variation);

    tcm.CheckSharedMemory(expected_stats);
  }
}

TEST_CASE("concurrency_deadlock")
{
  PerfAnalyzerParameters params{};
  params.max_concurrency = 6;
  bool is_sequence_model{true};
  bool some_infer_failures{false};

  const auto& ParameterizeSyncStreaming{[&]() {
    SUBCASE("sync")
    {
      params.async = false;
      params.streaming = false;
    }
    SUBCASE("aync no streaming")
    {
      params.async = true;
      params.streaming = false;
    }
    SUBCASE("async streaming")
    {
      params.async = true;
      params.streaming = true;
    }
  }};

  const auto& ParameterizeConcurrency{[&]() {
    SUBCASE("10 concurrency, 10 thread")
    {
      ParameterizeSyncStreaming();
      params.max_concurrency = 10;
      params.max_threads = 10;
    }
    SUBCASE("10 concurrency, 4 thread")
    {
      ParameterizeSyncStreaming();
      params.max_concurrency = 10;
      params.max_threads = 4;
    }
  }};

  const auto& ParameterizeSequence{[&]() {
    SUBCASE("non-sequence")
    {
      ParameterizeConcurrency();
      is_sequence_model = false;
    }
    SUBCASE("sequence")
    {
      ParameterizeConcurrency();
      is_sequence_model = true;
    }
  }};

  const auto& ParameterizeFailures{[&]() {
    SUBCASE("yes_failures")
    {
      some_infer_failures = true;
      ParameterizeSequence();
    }
    SUBCASE("no_failures")
    {
      some_infer_failures = false;
      ParameterizeSequence();
    }
  }};

  std::vector<uint64_t> delays;

  const auto& ParameterizeDelays{[&]() {
    SUBCASE("no_delay")
    {
      delays = {0};
      ParameterizeFailures();
    }
    SUBCASE("random_delay")
    {
      delays = {1, 5, 20, 4, 3};
      ParameterizeFailures();
    }
  }};


  ParameterizeDelays();


  TestConcurrencyManager tcm(params, is_sequence_model);

  tcm.InitManager(
      params.string_length, params.string_data, params.zero_input,
      params.user_data, params.start_sequence_id, params.sequence_id_range,
      params.sequence_length, params.sequence_length_specified,
      params.sequence_length_variation);

  tcm.stats_->SetDelays(delays);

  // Sometimes have a request fail
  if (some_infer_failures) {
    tcm.stats_->SetReturnStatuses({true, true, true, false});
  }

  tcm.TestTimeouts();
}

TEST_CASE("concurrency_overhead")
{
  PerfAnalyzerParameters params{};
  SUBCASE("sync, conc 1")
  {
    params.async = false;
    params.max_concurrency = 1;
  }
  SUBCASE("sync, conc 4")
  {
    params.async = false;
    params.max_concurrency = 4;
  }
  SUBCASE("async, conc 1")
  {
    params.async = true;
    params.max_concurrency = 1;
  }
  SUBCASE("async, conc 1")
  {
    params.async = true;
    params.max_concurrency = 4;
  }
  TestConcurrencyManager tcm(params, false);
  tcm.InitManager(
      params.string_length, params.string_data, params.zero_input,
      params.user_data, params.start_sequence_id, params.sequence_id_range,
      params.sequence_length, params.sequence_length_specified,
      params.sequence_length_variation);

  tcm.TestOverhead();
}

TEST_CASE(
    "send_request_rate_concurrency_manager: testing logic around detecting "
    "send request count")
{
  PerfAnalyzerParameters params{};

  SUBCASE("sync")
  {
    params.async = false;
  }
  SUBCASE("async")
  {
    params.async = true;
  }

  TestConcurrencyManager tcm(params);

  tcm.stats_->SetDelays({10});

  tcm.InitManager(
      params.string_length, params.string_data, params.zero_input,
      params.user_data, params.start_sequence_id, params.sequence_id_range,
      params.sequence_length, params.sequence_length_specified,
      params.sequence_length_variation);

  tcm.ChangeConcurrencyLevel(4);
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  tcm.StopWorkerThreads();

  const size_t num_sent_requests{tcm.GetAndResetNumSentRequests()};

  CHECK(num_sent_requests == doctest::Approx(40).epsilon(0.1));
}

TEST_CASE(
    "reconfigure_threads" *
    doctest::description(
        "This test confirms the side-effects of ReconfigThreads(). Namely, "
        "that the correct number of threads are created and that they are "
        "configured properly"))
{
  PerfAnalyzerParameters params{};
  std::vector<ThreadConfig> expected_config_values;
  std::vector<size_t> expected_concurrencies;
  std::vector<size_t> expected_seq_stat_index_offsets;
  std::vector<size_t> expected_num_requests;

  size_t target_concurrency = 0;
  size_t target_num_requests = 0;

  SUBCASE("normal")
  {
    params.max_threads = 10;
    target_concurrency = 5;
    target_num_requests = 15;

    expected_concurrencies = {1, 1, 1, 1, 1};
    expected_seq_stat_index_offsets = {0, 1, 2, 3, 4};
    expected_num_requests = {3, 3, 3, 3, 3};
  }
  SUBCASE("thread_limited")
  {
    params.max_threads = 5;
    target_concurrency = 10;
    target_num_requests = 20;

    expected_concurrencies = {2, 2, 2, 2, 2};
    expected_seq_stat_index_offsets = {0, 2, 4, 6, 8};
    expected_num_requests = {4, 4, 4, 4, 4};
  }
  SUBCASE("unbalanced")
  {
    params.max_threads = 6;
    target_concurrency = 14;
    target_num_requests = 15;

    expected_concurrencies = {3, 3, 2, 2, 2, 2};
    expected_seq_stat_index_offsets = {0, 3, 6, 8, 10, 12};
    expected_num_requests = {3, 3, 3, 2, 2, 2};
  }
  SUBCASE("no requests specified")
  {
    params.max_threads = 2;
    target_concurrency = 14;
    target_num_requests = 0;

    expected_concurrencies = {7, 7};
    expected_seq_stat_index_offsets = {0, 7};
    expected_num_requests = {0, 0};
  }

  for (auto i = 0; i < expected_concurrencies.size(); i++) {
    ThreadConfig tc(i);
    tc.concurrency_ = expected_concurrencies[i];
    tc.seq_stat_index_offset_ = expected_seq_stat_index_offsets[i];
    tc.num_requests_ = expected_num_requests[i];
    expected_config_values.push_back(tc);
  }

  TestConcurrencyManager tcm(params);
  tcm.TestReconfigThreads(
      target_concurrency, target_num_requests, expected_config_values);
}


}}  // namespace triton::perfanalyzer
