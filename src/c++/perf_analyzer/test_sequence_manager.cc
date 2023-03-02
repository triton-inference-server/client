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

#include "doctest.h"
#include "mock_data_loader.h"
#include "sequence_manager.h"

namespace triton { namespace perfanalyzer {

class TestSequenceManager : public SequenceManager {
 public:
  TestSequenceManager(
      const uint64_t start_sequence_id, const uint64_t sequence_id_range,
      const size_t sequence_length, const bool sequence_length_specified,
      const double sequence_length_variation, const bool using_json_data,
      std::shared_ptr<DataLoader> data_loader)
      : SequenceManager(
            start_sequence_id, sequence_id_range, sequence_length,
            sequence_length_specified, sequence_length_variation,
            using_json_data, data_loader)
  {
  }

  void InitNewSequence(int& seq_stat_index)
  {
    SequenceManager::InitNewSequence(seq_stat_index);
  }

  uint64_t GetNextSeqId(int& seq_stat_index)
  {
    return SequenceManager::GetNextSeqId(seq_stat_index);
  }

  size_t GetRandomSequenceLength(double& offset_ratio)
  {
    return SequenceManager::GetRandomSequenceLength(offset_ratio);
  }

  std::vector<std::shared_ptr<SequenceStatus>>& sequence_statuses_{
      SequenceManager::sequence_statuses_};
  std::atomic<uint64_t>& curr_seq_id_{SequenceManager::curr_seq_id_};
};

TEST_CASE(
    "test_set_infer_sequence_options: testing the SetInferSequenceOptions "
    "function")
{
  const uint64_t seq_id{5};
  std::vector<std::shared_ptr<SequenceStatus>> sequence_statuses{
      std::make_shared<SequenceStatus>(seq_id)};
  std::uniform_int_distribution<uint64_t> distribution(0, 0);
  const uint64_t start_sequence_id{1};
  const uint64_t sequence_id_range{UINT32_MAX};
  const size_t sequence_length{20};
  const bool sequence_length_specified{false};
  const double sequence_length_variation{0.2};
  bool using_json_data{false};
  std::shared_ptr<MockDataLoader> data_loader{
      std::make_shared<MockDataLoader>()};
  const uint32_t seq_stat_index{0};
  const std::string model_name{"model"};
  std::unique_ptr<cb::InferOptions> options{
      std::make_unique<cb::InferOptions>(model_name)};

  SUBCASE("start false, end false")
  {
    sequence_statuses[seq_stat_index]->remaining_queries_ = 2;

    TestSequenceManager tsm(
        start_sequence_id, sequence_id_range, sequence_length,
        sequence_length_specified, sequence_length_variation, using_json_data,
        data_loader);
    tsm.sequence_statuses_ = sequence_statuses;
    tsm.curr_seq_id_ = 5;

    tsm.SetInferSequenceOptions(seq_stat_index, options);

    CHECK(options->sequence_start_ == false);
    CHECK(options->sequence_id_ == 5);
    CHECK(options->sequence_end_ == false);
  }
  SUBCASE("start true, end false")
  {
    sequence_statuses[seq_stat_index]->remaining_queries_ = 0;

    TestSequenceManager tsm(
        start_sequence_id, sequence_id_range, sequence_length,
        sequence_length_specified, sequence_length_variation, using_json_data,
        data_loader);
    tsm.sequence_statuses_ = sequence_statuses;
    tsm.curr_seq_id_ = 5;

    tsm.SetInferSequenceOptions(seq_stat_index, options);

    CHECK(options->sequence_start_ == true);
    CHECK(options->sequence_id_ == 6);
    CHECK(options->sequence_end_ == false);
  }
  SUBCASE("start false, end true")
  {
    sequence_statuses[seq_stat_index]->remaining_queries_ = 1;

    TestSequenceManager tsm(
        start_sequence_id, sequence_id_range, sequence_length,
        sequence_length_specified, sequence_length_variation, using_json_data,
        data_loader);
    tsm.sequence_statuses_ = sequence_statuses;
    tsm.curr_seq_id_ = 5;

    tsm.SetInferSequenceOptions(seq_stat_index, options);

    CHECK(options->sequence_start_ == false);
    CHECK(options->sequence_id_ == 5);
    CHECK(options->sequence_end_ == true);
  }
  SUBCASE("start true, end true")
  {
    sequence_statuses[seq_stat_index]->remaining_queries_ = 0;
    using_json_data = true;
    data_loader->step_num_.push_back(1);
    data_loader->data_stream_cnt_ = 1;

    TestSequenceManager tsm(
        start_sequence_id, sequence_id_range, sequence_length,
        sequence_length_specified, sequence_length_variation, using_json_data,
        data_loader);
    tsm.sequence_statuses_ = sequence_statuses;
    tsm.curr_seq_id_ = 5;

    tsm.SetInferSequenceOptions(seq_stat_index, options);

    CHECK(options->sequence_start_ == true);
    CHECK(options->sequence_id_ == 6);
    CHECK(options->sequence_end_ == true);
  }
}

TEST_CASE("init_new_sequence: testing the InitNewSequence function")
{
  const uint64_t seq_id{5};
  std::vector<std::shared_ptr<SequenceStatus>> sequence_statuses{
      std::make_shared<SequenceStatus>(seq_id)};
  std::uniform_int_distribution<uint64_t> distribution(0, 0);
  const uint64_t start_sequence_id{1};
  const uint64_t sequence_id_range{UINT32_MAX};
  size_t sequence_length{20};
  bool sequence_length_specified{false};
  const double sequence_length_variation{0.2};
  bool using_json_data{false};
  std::shared_ptr<MockDataLoader> data_loader{
      std::make_shared<MockDataLoader>()};
  int seq_stat_index{0};
  size_t expected_sequence_length{0};

  SUBCASE("not using json data")
  {
    TestSequenceManager tsm(
        start_sequence_id, sequence_id_range, sequence_length,
        sequence_length_specified, sequence_length_variation, using_json_data,
        data_loader);
    tsm.sequence_statuses_ = sequence_statuses;
    tsm.curr_seq_id_ = 5;

    tsm.InitNewSequence(seq_stat_index);

    CHECK(tsm.sequence_statuses_[seq_stat_index]->seq_id_ == 6);
    CHECK(tsm.sequence_statuses_[seq_stat_index]->remaining_queries_ > 0);
  }

  SUBCASE("using json data")
  {
    using_json_data = true;
    data_loader->step_num_.push_back(5);
    data_loader->data_stream_cnt_ = 1;

    SUBCASE("sequence length not specified")
    {
      sequence_length_specified = false;
      expected_sequence_length = 5;
    }

    SUBCASE("sequence length specified, smaller than input data")
    {
      sequence_length_specified = true;
      sequence_length = 4;
      expected_sequence_length = 4;
    }

    SUBCASE("sequence length specified, larger than input data")
    {
      sequence_length_specified = true;
      sequence_length = 6;
      expected_sequence_length = 6;
    }

    TestSequenceManager tsm(
        start_sequence_id, sequence_id_range, sequence_length,
        sequence_length_specified, sequence_length_variation, using_json_data,
        data_loader);
    tsm.sequence_statuses_ = sequence_statuses;
    tsm.curr_seq_id_ = 5;

    tsm.InitNewSequence(seq_stat_index);

    CHECK(tsm.sequence_statuses_[seq_stat_index]->seq_id_ == 6);
    CHECK(
        tsm.sequence_statuses_[seq_stat_index]->remaining_queries_ ==
        expected_sequence_length);
    CHECK(
        tsm.sequence_statuses_[seq_stat_index]->sequence_length_ ==
        expected_sequence_length);
  }
}

TEST_CASE("get_next_seq_id: testing the GetNextSeqId function")
{
  std::vector<std::shared_ptr<SequenceStatus>> sequence_statuses{};
  std::uniform_int_distribution<uint64_t> distribution(0, 0);
  uint64_t start_sequence_id{0};
  uint64_t sequence_id_range{0};
  const size_t sequence_length{20};
  const bool sequence_length_specified{false};
  const double sequence_length_variation{0.2};
  const bool using_json_data{false};
  std::shared_ptr<MockDataLoader> data_loader{
      std::make_shared<MockDataLoader>()};
  int seq_stat_index{0};

  SUBCASE("next sequence id not in use")
  {
    sequence_statuses.push_back(std::make_shared<SequenceStatus>(1));
    start_sequence_id = 1;
    sequence_id_range = 2;

    TestSequenceManager tsm(
        start_sequence_id, sequence_id_range, sequence_length,
        sequence_length_specified, sequence_length_variation, using_json_data,
        data_loader);
    tsm.sequence_statuses_ = sequence_statuses;
    tsm.curr_seq_id_ = 3;

    uint64_t result{tsm.GetNextSeqId(seq_stat_index)};

    CHECK(result == 2);
  }

  SUBCASE("next sequence id in use")
  {
    sequence_statuses.push_back(std::make_shared<SequenceStatus>(1));
    sequence_statuses.push_back(std::make_shared<SequenceStatus>(2));
    start_sequence_id = 1;
    sequence_id_range = 2;

    TestSequenceManager tsm(
        start_sequence_id, sequence_id_range, sequence_length,
        sequence_length_specified, sequence_length_variation, using_json_data,
        data_loader);
    tsm.sequence_statuses_ = sequence_statuses;
    tsm.curr_seq_id_ = 3;

    uint64_t result{tsm.GetNextSeqId(seq_stat_index)};

    CHECK(result == 1);
  }
}

TEST_CASE(
    "get_random_sequence_length: testing the GetRandomSequenceLength function")
{
  std::vector<std::shared_ptr<SequenceStatus>> sequence_statuses{};
  std::uniform_int_distribution<uint64_t> distribution(0, 0);
  const uint64_t start_sequence_id{0};
  const uint64_t sequence_id_range{0};
  size_t sequence_length{20};
  const bool sequence_length_specified{false};
  const double sequence_length_variation{0.2};
  const bool using_json_data{false};
  std::shared_ptr<MockDataLoader> data_loader{
      std::make_shared<MockDataLoader>()};
  int seq_stat_index{0};
  double offset_ratio{0.2};

  TestSequenceManager tsm(
      start_sequence_id, sequence_id_range, sequence_length,
      sequence_length_specified, sequence_length_variation, using_json_data,
      data_loader);
  tsm.sequence_statuses_ = sequence_statuses;
  tsm.curr_seq_id_ = 3;

  uint64_t result{tsm.GetRandomSequenceLength(offset_ratio)};

  CHECK(result >= 16);
  CHECK(result <= 24);
}

}}  // namespace triton::perfanalyzer
