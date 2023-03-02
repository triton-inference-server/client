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
#include "infer_context.h"
#include "mock_data_loader.h"
#include "mock_infer_data_manager.h"
#include "mock_sequence_manager.h"

namespace triton { namespace perfanalyzer {

class TestInferContext : public InferContext {
 public:
  void UpdateSeqJsonData(size_t seq_stat_index)
  {
    InferContext::UpdateSeqJsonData(seq_stat_index);
  }

  std::shared_ptr<SequenceManager>& sequence_manager_{
      InferContext::sequence_manager_};
  std::shared_ptr<DataLoader>& data_loader_{InferContext::data_loader_};
  std::shared_ptr<IInferDataManager>& infer_data_manager_{
      InferContext::infer_data_manager_};
  std::shared_ptr<ThreadStat>& thread_stat_{InferContext::thread_stat_};
};

TEST_CASE("update_seq_json_data: testing the UpdateSeqJsonData function")
{
  std::shared_ptr<MockSequenceManager> mock_sequence_manager{
      std::make_shared<MockSequenceManager>()};
  mock_sequence_manager->sequence_length_ = 5;
  mock_sequence_manager->remaining_queries_ = 1;

  std::shared_ptr<MockDataLoader> mock_data_loader{
      std::make_shared<MockDataLoader>(true)};
  mock_data_loader->total_steps_ = 4;

  std::shared_ptr<MockInferDataManager> mock_infer_data_manager{
      std::make_shared<MockInferDataManager>(true)};

  TestInferContext tic{};
  tic.sequence_manager_ = mock_sequence_manager;
  tic.data_loader_ = mock_data_loader;
  tic.infer_data_manager_ = mock_infer_data_manager;
  tic.thread_stat_ = std::make_shared<ThreadStat>();

  tic.UpdateSeqJsonData(0);

  REQUIRE(
      mock_infer_data_manager->update_infer_data_step_index_values_.size() ==
      1);
  CHECK(mock_infer_data_manager->update_infer_data_step_index_values_[0] == 0);
}

}}  // namespace triton::perfanalyzer
