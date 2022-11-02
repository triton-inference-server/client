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

#include "command_line_parser.h"
#include "doctest.h"
#include "mock_client_backend.h"
#include "mock_model_parser.h"

namespace cb = triton::perfanalyzer::clientbackend;

namespace triton { namespace perfanalyzer {

/// Helper base class to be inherited when testing any Load Manager class
///
class TestLoadManagerBase {
 public:
  TestLoadManagerBase(PerfAnalyzerParameters params, bool is_sequence_model)
      : params_(params)
  {
    // Must reset this global flag every unit test.
    // It gets set to true when we deconstruct any load manager
    // (aka every unit test)
    //
    early_exit = false;

    stats_ = std::make_shared<cb::MockClientStats>();
    factory_ = std::make_shared<cb::MockClientBackendFactory>(stats_);
    parser_ = std::make_shared<MockModelParser>(is_sequence_model);
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

  void CheckSequences()
  {
    auto stats = GetStats();

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
      CHECK(
          params_.num_of_sequences ==
          stats->sequence_status.max_live_seq_count);
    }

    // Make sure that the length of each sequence is as expected
    // (The code explicitly has a 20% slop, so that is what we are checking)
    //
    auto num_sequences = params_.num_of_sequences;
    auto num_values = stats->sequence_status.seq_lengths.size();
    for (size_t i = 0; i < num_values; i++) {
      auto len = stats->sequence_status.seq_lengths[i];

      if (i + num_sequences < num_values) {
        CHECK(len == doctest::Approx(params_.sequence_length).epsilon(0.20));
      }
      // The last instance of each sequence might be shorter than expected, as
      // they may be terminated part way through
      //
      else {
        CHECK(len <= doctest::Approx(params_.sequence_length).epsilon(0.20));
      }
    }
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
};

}}  // namespace triton::perfanalyzer
