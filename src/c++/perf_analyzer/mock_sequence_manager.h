// Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "gmock/gmock.h"
#include "sequence_manager.h"

namespace triton { namespace perfanalyzer {

class MockSequenceManager : public SequenceManager {
 public:
  MockSequenceManager() { SetupMocks(); }

  MockSequenceManager(
      const uint64_t start_sequence_id, const uint64_t sequence_id_range,
      const size_t sequence_length, const bool sequence_length_specified,
      const double sequence_length_variation, const bool using_json_data,
      std::shared_ptr<DataLoader> data_loader)
      : SequenceManager(
            start_sequence_id, sequence_id_range, sequence_length,
            sequence_length_specified, sequence_length_variation,
            using_json_data, data_loader)
  {
    SetupMocks();
  }

  void SetupMocks()
  {
    ON_CALL(*this, SetInferSequenceOptions(testing::_, testing::_))
        .WillByDefault([this](
                           const uint32_t seq_stat_index,
                           std::unique_ptr<cb::InferOptions>& options) {
          this->SequenceManager::SetInferSequenceOptions(
              seq_stat_index, options);
        });
    ON_CALL(*this, InitNewSequence(testing::_))
        .WillByDefault([this](int seq_stat_index) {
          this->SequenceManager::InitNewSequence(seq_stat_index);
        });
    ON_CALL(*this, GetNextSeqId(testing::_))
        .WillByDefault([this](int seq_stat_index) -> uint64_t {
          return this->SequenceManager::GetNextSeqId(seq_stat_index);
        });
    ON_CALL(*this, GetRandomSequenceLength(testing::_))
        .WillByDefault([this](double offset_ratio) -> size_t {
          return this->SequenceManager::GetRandomSequenceLength(offset_ratio);
        });
  }

  MOCK_METHOD(
      void, SetInferSequenceOptions,
      (const uint32_t, std::unique_ptr<cb::InferOptions>&), (override));
  MOCK_METHOD(void, InitNewSequence, (int), (override));
  MOCK_METHOD(uint64_t, GetNextSeqId, (int), (override));
  MOCK_METHOD(size_t, GetRandomSequenceLength, (double), (override));

  std::vector<std::shared_ptr<SequenceStatus>>& sequence_statuses_{
      SequenceManager::sequence_statuses_};
  std::atomic<uint64_t>& curr_seq_id_{SequenceManager::curr_seq_id_};
};

}}  // namespace triton::perfanalyzer
