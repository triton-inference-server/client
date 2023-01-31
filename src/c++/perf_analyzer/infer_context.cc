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

#include "infer_context.h"

namespace triton { namespace perfanalyzer {

void
InferContext::Init()
{
  if (shared_memory_type_ == SharedMemoryType::NO_SHARED_MEMORY) {
    thread_stat_->status_ = memory_manager_->PrepareInfer(&infer_data_);
  } else {
    thread_stat_->status_ =
        memory_manager_->PrepareSharedMemoryInfer(&infer_data_);
  }
  if (!thread_stat_->status_.IsOk()) {
    return;
  }

  if (streaming_) {
    // Decoupled models should not collect client side statistics
    thread_stat_->status_ = infer_backend_->StartStream(
        async_callback_func_, (!parser_->IsDecoupled()));
    if (!thread_stat_->status_.IsOk()) {
      return;
    }
  }
}

void
InferContext::SendInferRequest(bool delayed)
{
  // Update the inputs if required
  if (using_json_data_) {
    UpdateJsonData(infer_data_);
  }
  SendRequest(request_id_++, delayed);
}

void
InferContext::SendSequenceInferRequest(uint32_t seq_stat_index, bool delayed)
{
  // Need lock to protect the order of dispatch across worker threads.
  // This also helps in reporting the realistic latencies.
  std::lock_guard<std::mutex> guard(sequence_stat_[seq_stat_index]->mtx_);
  if (!early_exit && !sequence_stat_[seq_stat_index]->paused_) {
    SetInferSequenceOptions(seq_stat_index, infer_data_.options_);

    // Update the inputs if required
    if (using_json_data_) {
      UpdateSeqJsonData(infer_data_, sequence_stat_[seq_stat_index]);
    }

    sequence_stat_[seq_stat_index]->remaining_queries_--;

    SendRequest(request_id_++, delayed);
  }
}

void
InferContext::CompleteOngoingSequence(uint32_t seq_stat_index)
{
  std::lock_guard<std::mutex> guard(sequence_stat_[seq_stat_index]->mtx_);
  sequence_stat_[seq_stat_index]->paused_ = true;

  if (sequence_stat_[seq_stat_index]->remaining_queries_ != 0) {
    sequence_stat_[seq_stat_index]->remaining_queries_ = 1;
    SetInferSequenceOptions(seq_stat_index, infer_data_.options_);

    if (using_json_data_) {
      UpdateSeqJsonData(infer_data_, sequence_stat_[seq_stat_index]);
    }
    sequence_stat_[seq_stat_index]->remaining_queries_--;

    bool is_delayed = false;
    SendRequest(request_id_++, is_delayed);
  }
}

void
InferContext::SendRequest(const uint64_t request_id, const bool delayed)
{
  if (!thread_stat_->status_.IsOk()) {
    return;
  }

  if (async_) {
    infer_data_.options_->request_id_ = std::to_string(request_id);
    {
      std::lock_guard<std::mutex> lock(thread_stat_->mu_);
      auto it =
          async_req_map_
              .emplace(
                  infer_data_.options_->request_id_, AsyncRequestProperties())
              .first;
      it->second.start_time_ = std::chrono::system_clock::now();
      it->second.sequence_end_ = infer_data_.options_->sequence_end_;
      it->second.delayed_ = delayed;
    }
    if (streaming_) {
      thread_stat_->status_ = infer_backend_->AsyncStreamInfer(
          *(infer_data_.options_), infer_data_.valid_inputs_,
          infer_data_.outputs_);
    } else {
      thread_stat_->status_ = infer_backend_->AsyncInfer(
          async_callback_func_, *(infer_data_.options_),
          infer_data_.valid_inputs_, infer_data_.outputs_);
    }
    total_ongoing_requests_++;
  } else {
    std::chrono::time_point<std::chrono::system_clock> start_time_sync,
        end_time_sync;
    start_time_sync = std::chrono::system_clock::now();
    cb::InferResult* results = nullptr;
    thread_stat_->status_ = infer_backend_->Infer(
        &results, *(infer_data_.options_), infer_data_.valid_inputs_,
        infer_data_.outputs_);
    if (results != nullptr) {
      if (thread_stat_->status_.IsOk()) {
        thread_stat_->status_ = ValidateOutputs(results);
      }
      delete results;
    }
    if (!thread_stat_->status_.IsOk()) {
      return;
    }
    end_time_sync = std::chrono::system_clock::now();
    {
      // Add the request timestamp to thread Timestamp vector with proper
      // locking
      std::lock_guard<std::mutex> lock(thread_stat_->mu_);
      thread_stat_->request_timestamps_.emplace_back(std::make_tuple(
          start_time_sync, end_time_sync, infer_data_.options_->sequence_end_,
          delayed));
      thread_stat_->status_ =
          infer_backend_->ClientInferStat(&(thread_stat_->contexts_stat_[id_]));
      if (!thread_stat_->status_.IsOk()) {
        return;
      }
    }
  }
}

cb::Error
InferContext::CreateInferInput(
    cb::InferInput** infer_input, const cb::BackendKind kind,
    const std::string& name, const std::vector<int64_t>& dims,
    const std::string& datatype)
{
  return cb::InferInput::Create(infer_input, kind, name, dims, datatype);
}

void
InferContext::UpdateJsonData(InferData& infer_data)
{
  int step_id =
      (data_step_id_ % data_loader_->GetTotalStepsNonSequence()) * batch_size_;
  data_step_id_ += GetNumActiveThreads();
  thread_stat_->status_ =
      UpdateInputs(infer_data.inputs_, infer_data.valid_inputs_, 0, step_id);
  if (thread_stat_->status_.IsOk()) {
    thread_stat_->status_ = memory_manager_->UpdateValidationOutputs(
        infer_data.outputs_, 0, step_id, infer_data.expected_outputs_);
  }
}

void
InferContext::UpdateSeqJsonData(
    InferData& infer_data, std::shared_ptr<SequenceStat> seq_stat)
{
  int step_id = data_loader_->GetTotalSteps(seq_stat->data_stream_id_) -
                seq_stat->remaining_queries_;
  thread_stat_->status_ = UpdateInputs(
      infer_data.inputs_, infer_data.valid_inputs_, seq_stat->data_stream_id_,
      step_id);
  if (thread_stat_->status_.IsOk()) {
    thread_stat_->status_ = memory_manager_->UpdateValidationOutputs(
        infer_data.outputs_, seq_stat->data_stream_id_, step_id,
        infer_data.expected_outputs_);
  }
}


void
InferContext::SetInferSequenceOptions(
    const uint32_t seq_stat_index, std::unique_ptr<cb::InferOptions>& options)
{
  options->sequence_start_ =
      (sequence_stat_[seq_stat_index]->remaining_queries_ == 0);

  // New sequence must be intialized before setting the id.
  if (options->sequence_start_) {
    InitNewSequence(seq_stat_index);
  }
  options->sequence_id_ = sequence_stat_[seq_stat_index]->seq_id_;
  options->sequence_end_ =
      (sequence_stat_[seq_stat_index]->remaining_queries_ == 1);
}

void
InferContext::InitNewSequence(int seq_stat_index)
{
  sequence_stat_[seq_stat_index]->seq_id_ = GetNextSeqId(seq_stat_index);
  if (!using_json_data_) {
    size_t new_length = GetRandomSequenceLength(0.2);
    sequence_stat_[seq_stat_index]->remaining_queries_ =
        new_length == 0 ? 1 : new_length;
  } else {
    // Selecting next available data stream based on uniform distribution.
    sequence_stat_[seq_stat_index]->data_stream_id_ =
        distribution_(rng_generator_);
    sequence_stat_[seq_stat_index]->remaining_queries_ =
        data_loader_->GetTotalSteps(
            sequence_stat_[seq_stat_index]->data_stream_id_);
  }
}

cb::Error
InferContext::ValidateOutputs(const cb::InferResult* result_ptr)
{
  // Validate output if set
  if (!infer_data_.expected_outputs_.empty()) {
    for (size_t i = 0; i < infer_data_.outputs_.size(); ++i) {
      const uint8_t* buf = nullptr;
      size_t byte_size = 0;
      result_ptr->RawData(infer_data_.outputs_[i]->Name(), &buf, &byte_size);
      for (const auto& expected : infer_data_.expected_outputs_[i]) {
        if (byte_size < expected.second) {
          return cb::Error(
              "Output size doesn't match expected size", pa::GENERIC_ERROR);
        } else if (memcmp(buf, expected.first, expected.second) != 0) {
          return cb::Error(
              "Output doesn't match expected output", pa::GENERIC_ERROR);
        } else {
          buf += expected.second;
          byte_size -= expected.second;
        }
      }
      if (byte_size != 0) {
        return cb::Error(
            "Output size doesn't match expected size", pa::GENERIC_ERROR);
      }
    }
  }
  return cb::Error::Success;
}


cb::Error
InferContext::UpdateInputs(
    const std::vector<cb::InferInput*>& inputs,
    std::vector<cb::InferInput*>& valid_inputs, int stream_index,
    int step_index)
{
  // Validate update parameters here
  size_t data_stream_count = data_loader_->GetDataStreamsCount();
  if (stream_index < 0 || stream_index >= (int)data_stream_count) {
    return cb::Error(
        "stream_index for retrieving the data should be less than " +
            std::to_string(data_stream_count) + ", got " +
            std::to_string(stream_index),
        pa::GENERIC_ERROR);
  }
  size_t step_count = data_loader_->GetTotalSteps(stream_index);
  if (step_index < 0 || step_index >= (int)step_count) {
    return cb::Error(
        "step_id for retrieving the data should be less than " +
            std::to_string(step_count) + ", got " + std::to_string(step_index),
        pa::GENERIC_ERROR);
  }

  if (shared_memory_type_ == SharedMemoryType::NO_SHARED_MEMORY) {
    RETURN_IF_ERROR(memory_manager_->SetInputs(
        inputs, valid_inputs, stream_index, step_index));
  } else {
    RETURN_IF_ERROR(memory_manager_->SetInputsSharedMemory(
        inputs, stream_index, step_index));
  }

  return cb::Error::Success;
}

uint64_t
InferContext::GetNextSeqId(int seq_stat_index)
{
  uint64_t old_seq_id = sequence_stat_[seq_stat_index]->seq_id_;
  uint64_t next_seq_id =
      curr_seq_id_++ % sequence_id_range_ + start_sequence_id_;

  // If the next sequence ID is still in use, reuse the same sequence ID
  // that this sequence_stat used last time
  //
  for (uint i = 0; i < sequence_stat_.size(); i++) {
    if (next_seq_id == sequence_stat_[i]->seq_id_) {
      next_seq_id = old_seq_id;
      break;
    }
  }
  return next_seq_id;
}

size_t
InferContext::GetRandomSequenceLength(double offset_ratio)
{
  int random_offset = ((2.0 * rand() / double(RAND_MAX)) - 1.0) * offset_ratio *
                      sequence_length_;
  if (int(sequence_length_) + random_offset <= 0) {
    return 1;
  }
  return sequence_length_ + random_offset;
}

void
InferContext::AsyncCallbackFuncImpl(cb::InferResult* result)
{
  std::shared_ptr<cb::InferResult> result_ptr(result);
  if (thread_stat_->cb_status_.IsOk()) {
    // Add the request timestamp to thread Timestamp vector with
    // proper locking
    std::lock_guard<std::mutex> lock(thread_stat_->mu_);
    thread_stat_->cb_status_ = result_ptr->RequestStatus();
    if (thread_stat_->cb_status_.IsOk()) {
      std::chrono::time_point<std::chrono::system_clock> end_time_async;
      end_time_async = std::chrono::system_clock::now();
      std::string request_id;
      thread_stat_->cb_status_ = result_ptr->Id(&request_id);
      const auto& it = async_req_map_.find(request_id);
      if (it != async_req_map_.end()) {
        thread_stat_->request_timestamps_.emplace_back(std::make_tuple(
            it->second.start_time_, end_time_async, it->second.sequence_end_,
            it->second.delayed_));
        infer_backend_->ClientInferStat(&(thread_stat_->contexts_stat_[id_]));
        thread_stat_->cb_status_ = ValidateOutputs(result);
        async_req_map_.erase(request_id);
      }
    }
  }

  total_ongoing_requests_--;

  if (async_callback_finalize_func_ != nullptr) {
    async_callback_finalize_func_(id_);
  }
}

}}  // namespace triton::perfanalyzer