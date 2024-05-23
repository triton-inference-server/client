// Copyright 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <memory>

#include "command_line_parser.h"
#include "doctest.h"
#include "mock_client_backend.h"
#include "mock_data_loader.h"
#include "mock_model_parser.h"
#include "sequence_manager.h"

namespace cb = triton::perfanalyzer::clientbackend;

namespace triton { namespace perfanalyzer {

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
  TestLoadManagerBase() = default;
  TestLoadManagerBase(
      PerfAnalyzerParameters params, bool is_sequence_model,
      bool is_decoupled_model)
      : params_(params)
  {
    stats_ = std::make_shared<cb::MockClientStats>();
    factory_ = std::make_shared<cb::MockClientBackendFactory>(stats_);
    parser_ = std::make_shared<MockModelParser>(
        is_sequence_model, is_decoupled_model);
  }

  ~TestLoadManagerBase()
  {
    // FIXME TKG

    // Reset early_exit in case any test sets it to true during execution.
    early_exit = false;
  }

  // Helper function to process custom json data in testing
  // Creates a model tensor to pass to a mock parser which is consumed by the
  // mock data loader
  static MockInputPipeline ProcessCustomJsonData(
      const std::string& json_str, const bool is_sequence_model = false)
  {
    std::shared_ptr<MockModelParser> mmp{
        std::make_shared<MockModelParser>(is_sequence_model, false)};
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

  // Verifies that the number of inferences for each sequence is n or n+1.
  //
  void CheckSequenceBalance()
  {
    auto first_value = -1;
    auto second_value = -1;

    for (auto seq : stats_->sequence_status.seq_ids_to_count) {
      auto count = seq.second;
      // set first possible value for seqs
      if (first_value == -1) {
        first_value = count;
        continue;
      }
      // set second possible value for seqs count
      if (second_value == -1) {
        if (count == first_value + 1 || count == first_value - 1) {
          second_value = count;
          continue;
        } else if (first_value == count) {
          continue;
        }
      }

      if (count != first_value || count != second_value) {
        std::stringstream os;
        os << "Sequence request counts were not balanced: ";
        for (auto x : stats_->sequence_status.seq_ids_to_count) {
          os << x.second << ",";
        }
        CHECK_MESSAGE(
            (count == first_value || count == second_value), os.str());
        break;
      }
    }
  }

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
    SUBCASE("num_of_sequences 1")
    {
      params.num_of_sequences = 1;
    }
    SUBCASE("less threads than seq")
    {
      params.num_of_sequences = 12;
    }
    SUBCASE("num_of_sequences 8")
    {
      params.num_of_sequences = 8;
      // Make sequences long so we actually get 8 in flight at a time
      params.sequence_length = 20;
    }
    SUBCASE("sequence_length 1")
    {
      params.sequence_length = 1;
    }
    SUBCASE("sequence_length 10")
    {
      params.sequence_length = 10;
    }
    return params;
  }
};
}}  // namespace triton::perfanalyzer
