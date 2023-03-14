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

#include "sequence_manager.h"

namespace triton { namespace perfanalyzer {

SequenceManager::SequenceManager(
    const uint64_t start_sequence_id, const uint64_t sequence_id_range,
    const size_t sequence_length, const bool sequence_length_specified,
    const double sequence_length_variation, const bool using_json_data,
    std::shared_ptr<DataLoader> data_loader)
    : start_sequence_id_(start_sequence_id),
      sequence_id_range_(sequence_id_range), sequence_length_(sequence_length),
      sequence_length_specified_(sequence_length_specified),
      sequence_length_variation_(sequence_length_variation),
      using_json_data_(using_json_data), data_loader_(data_loader)
{
  distribution_ = std::uniform_int_distribution<uint64_t>(
      0, data_loader_->GetDataStreamsCount() - 1);
}

void
SequenceManager::InitSequenceStatuses(size_t num_sequence_statuses)
{
  sequence_statuses_.clear();
  for (size_t sequence_status_index{0};
       sequence_status_index < num_sequence_statuses; sequence_status_index++) {
    sequence_statuses_.push_back(std::make_shared<SequenceStatus>());
  }
}

std::mutex&
SequenceManager::GetMutex(size_t sequence_status_index)
{
  return sequence_statuses_.at(sequence_status_index)->mtx_;
}

const uint64_t
SequenceManager::GetDataStreamID(size_t sequence_status_index) const
{
  return sequence_statuses_.at(sequence_status_index)->data_stream_id_;
}

const size_t
SequenceManager::GetRemainingQueries(size_t sequence_status_index) const
{
  return sequence_statuses_.at(sequence_status_index)->remaining_queries_;
}

void
SequenceManager::SetRemainingQueries(
    size_t sequence_status_index, size_t remaining_queries)
{
  sequence_statuses_.at(sequence_status_index)->remaining_queries_ =
      remaining_queries;
}

void
SequenceManager::DecrementRemainingQueries(size_t sequence_status_index)
{
  sequence_statuses_.at(sequence_status_index)->remaining_queries_--;
}

const size_t
SequenceManager::GetNumSequenceStatuses() const
{
  return sequence_statuses_.size();
}

void
SequenceManager::SetInferSequenceOptions(
    const uint32_t seq_stat_index, std::unique_ptr<cb::InferOptions>& options)
{
  options->sequence_start_ =
      (sequence_statuses_[seq_stat_index]->remaining_queries_ == 0);

  // New sequence must be intialized before setting the id.
  if (options->sequence_start_) {
    InitNewSequence(seq_stat_index);
  }
  options->sequence_id_ = sequence_statuses_[seq_stat_index]->seq_id_;
  options->sequence_end_ =
      (sequence_statuses_[seq_stat_index]->remaining_queries_ == 1);
}

const size_t
SequenceManager::GetSequenceLength(size_t sequence_status_index) const
{
  return sequence_statuses_.at(sequence_status_index)->sequence_length_;
}

void
SequenceManager::InitNewSequence(int seq_stat_index)
{
  sequence_statuses_[seq_stat_index]->seq_id_ = GetNextSeqId(seq_stat_index);
  if (!using_json_data_) {
    size_t new_length = GetRandomSequenceLength(sequence_length_variation_);
    sequence_statuses_[seq_stat_index]->remaining_queries_ =
        new_length == 0 ? 1 : new_length;
  } else {
    // Selecting next available data stream based on uniform distribution.
    sequence_statuses_[seq_stat_index]->data_stream_id_ = GetNewDataStreamId();
    const uint64_t data_stream_id{
        sequence_statuses_[seq_stat_index]->data_stream_id_};
    const size_t total_steps{data_loader_->GetTotalSteps(data_stream_id)};
    if (sequence_length_specified_) {
      const size_t varied_sequence_length{
          GetRandomSequenceLength(sequence_length_variation_)};
      sequence_statuses_[seq_stat_index]->sequence_length_ =
          varied_sequence_length;
    } else {
      sequence_statuses_[seq_stat_index]->sequence_length_ = total_steps;
    }
    sequence_statuses_[seq_stat_index]->remaining_queries_ =
        sequence_statuses_[seq_stat_index]->sequence_length_;
  }
}

uint64_t
SequenceManager::GetNextSeqId(int seq_stat_index)
{
  uint64_t old_seq_id = sequence_statuses_[seq_stat_index]->seq_id_;
  uint64_t next_seq_id =
      curr_seq_id_++ % sequence_id_range_ + start_sequence_id_;

  // If the next sequence ID is still in use, reuse the same sequence ID
  // that this sequence_status used last time
  //
  for (uint i = 0; i < sequence_statuses_.size(); i++) {
    if (next_seq_id == sequence_statuses_[i]->seq_id_) {
      next_seq_id = old_seq_id;
      break;
    }
  }
  return next_seq_id;
}

size_t
SequenceManager::GetRandomSequenceLength(double offset_ratio)
{
  int random_offset = ((2.0 * rand() / double(RAND_MAX)) - 1.0) * offset_ratio /
                      100.0 * sequence_length_;
  if (int(sequence_length_) + random_offset <= 0) {
    return 1;
  }
  return sequence_length_ + random_offset;
}

}}  // namespace triton::perfanalyzer
