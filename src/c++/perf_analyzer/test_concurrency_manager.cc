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

#include <future>
#include "command_line_parser.h"
#include "concurrency_manager.h"
#include "doctest.h"
#include "mock_client_backend.h"
#include "mock_data_loader.h"
#include "mock_model_parser.h"
#include "test_load_manager_base.h"

namespace triton { namespace perfanalyzer {

class TestConcurrencyWorker : public ConcurrencyWorker {
 public:
  TestConcurrencyWorker(
      std::shared_ptr<ThreadStat> thread_stat,
      std::shared_ptr<ThreadConfig> thread_config,
      const std::shared_ptr<ModelParser> parser,
      std::shared_ptr<DataLoader> data_loader, cb::BackendKind backend_kind,
      const std::shared_ptr<cb::ClientBackendFactory> factory,
      const size_t sequence_length, const uint64_t start_sequence_id,
      const uint64_t sequence_id_range, const bool on_sequence_model,
      const bool async, const size_t max_concurrency,
      const bool using_json_data, const bool streaming,
      const SharedMemoryType shared_memory_type, const int32_t batch_size,
      std::vector<std::shared_ptr<ThreadConfig>>& threads_config,
      std::vector<std::shared_ptr<SequenceStat>>& sequence_stat,
      std::unordered_map<std::string, SharedMemoryData>& shared_memory_regions,
      std::condition_variable& wake_signal, std::mutex& wake_mutex,
      size_t& active_threads, bool& execute, std::atomic<uint64_t>& curr_seq_id,
      std::uniform_int_distribution<uint64_t>& distribution)
      : ConcurrencyWorker(
            thread_stat, thread_config, parser, data_loader, backend_kind,
            factory, sequence_length, start_sequence_id, sequence_id_range,
            on_sequence_model, async, max_concurrency, using_json_data,
            streaming, shared_memory_type, batch_size, threads_config,
            sequence_stat, shared_memory_regions, wake_signal, wake_mutex,
            active_threads, execute, curr_seq_id, distribution)
  {
  }

  /// Mock out the InferInput object
  ///
  cb::Error CreateInferInput(
      cb::InferInput** infer_input, const cb::BackendKind kind,
      const std::string& name, const std::vector<int64_t>& dims,
      const std::string& datatype) override
  {
    *infer_input = new cb::MockInferInput(kind, name, dims, datatype);
    return cb::Error::Success;
  }
};

class MockConcurrencyWorker : public IWorker {
 public:
  MockConcurrencyWorker(
      std::shared_ptr<ConcurrencyWorker::ThreadConfig> thread_config)
      : thread_config_(thread_config)
  {
  }

  void Infer() override { thread_config_->is_paused_ = true; }

 private:
  std::shared_ptr<ConcurrencyWorker::ThreadConfig> thread_config_;
};

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
            params.max_threads, params.max_concurrency, params.sequence_length,
            params.shared_memory_type, params.output_shm_size,
            params.start_sequence_id, params.sequence_id_range, GetParser(),
            GetFactory())
  {
    InitManager(
        params.string_length, params.string_data, params.zero_input,
        params.user_data);
  }

  /// Constructor that adds an arg to pass in the model parser and does NOT call
  /// the InitManager code. This enables InitManager to be overloaded and mocked
  /// out.
  ///
  TestConcurrencyManager(
      PerfAnalyzerParameters params, const std::shared_ptr<ModelParser>& parser,
      bool is_sequence_model = false, bool is_decoupled_model = false,
      bool use_mock_infer = false)
      : use_mock_infer_(use_mock_infer),
        TestLoadManagerBase(params, is_sequence_model, is_decoupled_model),
        ConcurrencyManager(
            params.async, params.streaming, params.batch_size,
            params.max_threads, params.max_concurrency, params.sequence_length,
            params.shared_memory_type, params.output_shm_size,
            params.start_sequence_id, params.sequence_id_range, parser,
            GetFactory())
  {
  }

  // Mocked version of the CopySharedMemory method in loadmanager.
  // This is strictly for testing to mock out the memcpy calls
  //
  cb::Error CopySharedMemory(
      uint8_t* input_shm_ptr, std::vector<const uint8_t*>& data_ptrs,
      std::vector<size_t>& byte_size, bool is_shape_tensor,
      std::string& region_name) override
  {
    return cb::Error::Success;
  }

  std::shared_ptr<IWorker> MakeWorker(
      std::shared_ptr<ThreadStat> thread_stat,
      std::shared_ptr<ConcurrencyWorker::ThreadConfig> thread_config) override
  {
    if (use_mock_infer_) {
      return std::make_shared<MockConcurrencyWorker>(thread_config);
    } else {
      return std::make_shared<TestConcurrencyWorker>(
          thread_stat, thread_config, parser_, data_loader_, backend_->Kind(),
          ConcurrencyManager::factory_, sequence_length_, start_sequence_id_,
          sequence_id_range_, on_sequence_model_, async_, max_concurrency_,
          using_json_data_, streaming_, shared_memory_type_, batch_size_,
          threads_config_, sequence_stat_, shared_memory_regions_, wake_signal_,
          wake_mutex_, active_threads_, execute_, curr_seq_id_, distribution_);
    }
  }

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

  std::shared_ptr<ModelParser>& parser_{LoadManager::parser_};
  std::shared_ptr<DataLoader>& data_loader_{LoadManager::data_loader_};
  bool& using_json_data_{LoadManager::using_json_data_};
  bool& execute_{ConcurrencyManager::execute_};
  size_t& batch_size_{LoadManager::batch_size_};
  size_t& max_threads_{LoadManager::max_threads_};
  bool& using_shared_memory_{LoadManager::using_shared_memory_};
  std::uniform_int_distribution<uint64_t>& distribution_{
      LoadManager::distribution_};

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

/// Create the case where the sequences do NOT go round robin due to
/// the first request taking longer than the rest.
///
/// This exposed a bug where we were constantly resetting free_ctx_ids
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

  // Have the first request (sequence ID 1) take very long, and all the other
  // requests are fast
  //
  tcm.stats_->SetDelays({50, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5});

  std::shared_ptr<ThreadStat> thread_stat{std::make_shared<ThreadStat>()};
  std::shared_ptr<ConcurrencyWorker::ThreadConfig> thread_config{
      std::make_shared<ConcurrencyWorker::ThreadConfig>(0)};
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

  TestConcurrencyManager tcm(params, is_sequence_model);
  std::shared_ptr<MockModelParser> mmp{
      std::make_shared<MockModelParser>(false, false)};
  ModelTensor model_tensor{};
  model_tensor.datatype_ = "INT32";
  model_tensor.is_optional_ = false;
  model_tensor.is_shape_tensor_ = false;
  model_tensor.name_ = "INPUT0";
  model_tensor.shape_ = {1};
  mmp->inputs_ = std::make_shared<ModelTensorMap>();
  (*mmp->inputs_)[model_tensor.name_] = model_tensor;

  std::shared_ptr<MockDataLoader> mdl{std::make_shared<MockDataLoader>()};
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
  mdl->ReadDataFromStr(json_str, mmp->Inputs(), mmp->Outputs());

  std::shared_ptr<ThreadStat> thread_stat{std::make_shared<ThreadStat>()};
  std::shared_ptr<ConcurrencyWorker::ThreadConfig> thread_config{
      std::make_shared<ConcurrencyWorker::ThreadConfig>(0)};
  thread_config->concurrency_ = 1;

  tcm.parser_ = mmp;
  tcm.data_loader_ = mdl;
  tcm.using_json_data_ = true;
  tcm.execute_ = true;
  tcm.batch_size_ = 1;
  tcm.max_threads_ = 1;
  tcm.distribution_ = std::uniform_int_distribution<uint64_t>(
      0, mdl->GetDataStreamsCount() - 1);

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

/// Check that the using_shared_memory_ is being set correctly
///
TEST_CASE("Concurrency - Check setting of InitSharedMemory")
{
  PerfAnalyzerParameters params;
  bool is_sequence = false;
  bool is_decoupled = false;
  bool use_mock_infer = true;

  SUBCASE("No shared memory")
  {
    params.shared_memory_type = NO_SHARED_MEMORY;
    TestConcurrencyManager tcm(
        params, is_sequence, is_decoupled, use_mock_infer);
    CHECK(false == tcm.using_shared_memory_);
  }

  SUBCASE("System shared memory")
  {
    params.shared_memory_type = SYSTEM_SHARED_MEMORY;
    TestConcurrencyManager tcm(
        params, is_sequence, is_decoupled, use_mock_infer);
    CHECK(true == tcm.using_shared_memory_);
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

  std::shared_ptr<MockModelParser> mmp{
      std::make_shared<MockModelParser>(false, false)};
  ModelTensor model_tensor{};
  model_tensor.datatype_ = "INT32";
  model_tensor.is_optional_ = false;
  model_tensor.is_shape_tensor_ = false;
  model_tensor.name_ = "INPUT0";
  model_tensor.shape_ = {1};
  mmp->inputs_ = std::make_shared<ModelTensorMap>();
  (*mmp->inputs_)[model_tensor.name_] = model_tensor;

  std::shared_ptr<MockDataLoader> mdl{std::make_shared<MockDataLoader>()};
  const std::string json_str{R"(
  {
    "data": [
      {
        "INPUT0": [2123456789]
      }
    ]
  }
      )"};

  mdl->ReadDataFromStr(json_str, mmp->Inputs(), mmp->Outputs());

  cb::MockClientStats::SharedMemoryStats expected_stats;

  SUBCASE("System shared memory usage")
  {
    params.shared_memory_type = SYSTEM_SHARED_MEMORY;
    TestConcurrencyManager tcm(
        params, mmp, is_sequence, is_decoupled, use_mock_infer);
    tcm.InitManager(
        params.string_length, params.string_data, params.zero_input,
        params.user_data);

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
        params, mmp, is_sequence, is_decoupled, use_mock_infer);
    tcm.InitManager(
        params.string_length, params.string_data, params.zero_input,
        params.user_data);

    expected_stats.num_unregister_all_shared_memory_calls = 1;
    expected_stats.num_register_cuda_shared_memory_calls = 1;
    tcm.CheckSharedMemory(expected_stats);
  }

  SUBCASE("No shared memory usage")
  {
    params.shared_memory_type = NO_SHARED_MEMORY;
    TestConcurrencyManager tcm(
        params, mmp, is_sequence, is_decoupled, use_mock_infer);
    tcm.InitManager(
        params.string_length, params.string_data, params.zero_input,
        params.user_data);

    tcm.CheckSharedMemory(expected_stats);
  }
}

}}  // namespace triton::perfanalyzer
