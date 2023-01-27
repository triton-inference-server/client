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

#include <algorithm>
#include "command_line_parser.h"
#include "doctest.h"
#include "mock_client_backend.h"
#include "mock_data_loader.h"
#include "mock_model_parser.h"

namespace cb = triton::perfanalyzer::clientbackend;

namespace triton { namespace perfanalyzer {

class InferContextMockedInferInput : public InferContext {
 public:
  InferContextMockedInferInput(
      const uint32_t id, const bool async, const bool streaming,
      const bool on_sequence_model, const bool using_json_data,
      const int32_t batch_size, const cb::BackendKind backend_kind,
      const SharedMemoryType shared_memory_type,
      std::unordered_map<std::string, SharedMemoryData>& shared_memory_regions,
      const uint64_t start_sequence_id, const uint64_t sequence_id_range,
      const size_t sequence_length, std::atomic<uint64_t>& curr_seq_id,
      std::uniform_int_distribution<uint64_t>& distribution,
      std::shared_ptr<ThreadStat> thread_stat,
      std::vector<std::shared_ptr<SequenceStat>>& sequence_stat,
      std::shared_ptr<DataLoader> data_loader,
      std::shared_ptr<ModelParser> parser,
      std::shared_ptr<cb::ClientBackendFactory> factory)
      : InferContext(
            id, async, streaming, on_sequence_model, using_json_data,
            batch_size, backend_kind, shared_memory_type, shared_memory_regions,
            start_sequence_id, sequence_id_range, sequence_length, curr_seq_id,
            distribution, thread_stat, sequence_stat, data_loader, parser,
            factory)
  {
  }

 protected:
  /// Creates inference input object
  /// \param infer_input Output parameter storing newly created inference input
  /// \param kind Backend kind
  /// \param name Name of inference input
  /// \param dims Shape of inference input
  /// \param datatype Data type of inference input
  /// \return cb::Error object indicating success or failure.
  cb::Error CreateInferInput(
      cb::InferInput** infer_input, const cb::BackendKind kind,
      const std::string& name, const std::vector<int64_t>& dims,
      const std::string& datatype) override
  {
    *infer_input = new cb::MockInferInput(kind, name, dims, datatype);
    return cb::Error::Success;
  }
};

// Struct to hold the mock pieces to ingest custom json data
struct MockInputPipeline {
  MockInputPipeline(
      std::shared_ptr<MockModelParser> mmp, std::shared_ptr<MockDataLoader> mdl)
      : mock_model_parser_(mmp), mock_data_loader_(mdl)
  {
  }
  std::shared_ptr<MockModelParser> mock_model_parser_;
  std::shared_ptr<MockDataLoader> mock_data_loader_;
};

/// Helper base class to be inherited when testing any Load Manager class
///
class TestLoadManagerBase {
 public:
  TestLoadManagerBase(
      PerfAnalyzerParameters params, bool is_sequence_model,
      bool is_decoupled_model)
      : params_(params)
  {
    // Must reset this global flag every unit test.
    // It gets set to true when we deconstruct any load manager
    // (aka every unit test)
    //
    early_exit = false;

    stats_ = std::make_shared<cb::MockClientStats>();
    factory_ = std::make_shared<cb::MockClientBackendFactory>(stats_);
    parser_ = std::make_shared<MockModelParser>(
        is_sequence_model, is_decoupled_model);
  }
  // Helper function to process custom json data in testing
  // Creates a model tensor to pass to a mock parser which is consumed by the
  // mock data loader
  static MockInputPipeline ProcessCustomJsonData(const std::string& json_str)
  {
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
    mdl->ReadDataFromStr(json_str, mmp->Inputs(), mmp->Outputs());
    return MockInputPipeline{mmp, mdl};
  }

  // Set up all combinations of parameters for sequence testing
  //
  static PerfAnalyzerParameters GetSequenceTestParams()
  {
    PerfAnalyzerParameters params;
    bool is_async;

    SUBCASE("Async sequence")
    {
      is_async = true;
      params = GetSequenceTestParamsHelper(is_async);
    }
    SUBCASE("Sync sequence")
    {
      is_async = false;
      params = GetSequenceTestParamsHelper(is_async);
    }
    return params;
  }

  void CheckInferType()
  {
    auto stats = GetStats();

    if (params_.async) {
      if (params_.streaming) {
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
      if (params_.streaming) {
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


  void CheckSharedMemory(
      const cb::MockClientStats::SharedMemoryStats& expected_stats)
  {
    auto actual_stats = GetStats();
    CHECK(expected_stats == actual_stats->memory_stats);
  }

  void CheckSequences(uint64_t expected_num_seq)
  {
    auto stats = GetStats();

    // Make sure no live sequences remain
    CHECK(stats->sequence_status.live_seq_ids_to_length.size() == 0);

    // Make sure all seq IDs are within range
    //
    for (auto seq_id : stats->sequence_status.used_seq_ids) {
      CHECK(seq_id >= params_.start_sequence_id);
      CHECK(seq_id <= params_.start_sequence_id + params_.sequence_id_range);
    }

    // Make sure that we had the correct number of concurrently live sequences
    //
    // If the sequence length is only 1 then there is nothing to check because
    // there are never any overlapping requests -- they always immediately exit
    //
    if (params_.sequence_length != 1) {
      expected_num_seq = std::min(expected_num_seq, params_.sequence_id_range);
      CHECK(expected_num_seq == stats->sequence_status.max_live_seq_count);
    }

    // Make sure that the length of each sequence is as expected
    //
    // All but X of them should be within 20% (The code explicitly has a 20%
    // slop) of the requested sequence length, where X is the number of
    // sequences (This is due to the shutdown of sequences at the end that will
    // create shorter than expected sequences)
    //
    auto num_values = stats->sequence_status.seq_lengths.size();
    auto max_len = params_.sequence_length * 1.2;
    auto min_len = params_.sequence_length * 0.8;
    auto num_allowed_to_be_below_min_len = expected_num_seq;
    auto num_below_min_len = 0;

    for (size_t i = 0; i < num_values; i++) {
      auto len = stats->sequence_status.seq_lengths[i];

      CHECK(len <= max_len);
      if (len < min_len) {
        num_below_min_len++;
      }
    }
    CHECK(num_below_min_len <= num_allowed_to_be_below_min_len);
  }

  std::shared_ptr<cb::MockClientStats> stats_;

 protected:
  PerfAnalyzerParameters params_;
  std::shared_ptr<cb::ClientBackendFactory> factory_;
  std::shared_ptr<ModelParser> parser_;

  const std::shared_ptr<ModelParser>& GetParser() { return parser_; }
  const std::shared_ptr<cb::ClientBackendFactory>& GetFactory()
  {
    return factory_;
  }
  std::shared_ptr<cb::MockClientStats> GetStats() { return stats_; }
  void ResetStats() { stats_->Reset(); }

  static PerfAnalyzerParameters GetSequenceTestParamsHelper(bool is_async)
  {
    PerfAnalyzerParameters params;

    params.async = is_async;

    // Generally we want short sequences for testing
    // so we can hit the corner cases more often
    //
    params.sequence_length = 4;
    params.max_concurrency = 8;
    params.max_threads = 8;

    SUBCASE("Normal") {}
    SUBCASE("sequence IDs test 1")
    {
      params.start_sequence_id = 1;
      params.sequence_id_range = 3;
    }
    SUBCASE("sequence IDs test 2")
    {
      params.start_sequence_id = 17;
      params.sequence_id_range = 8;
    }
    SUBCASE("num_of_sequences 1") { params.num_of_sequences = 1; }
    SUBCASE("num_of_sequences 8")
    {
      params.num_of_sequences = 8;
      // Make sequences long so we actually get 8 in flight at a time
      params.sequence_length = 20;
    }
    SUBCASE("sequence_length 1") { params.sequence_length = 1; }
    SUBCASE("sequence_length 10") { params.sequence_length = 10; }
    return params;
  }
};
}}  // namespace triton::perfanalyzer
