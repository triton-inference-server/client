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
  thread_stat_->status_ = infer_data_manager_->PrepareInfer(infer_data_);
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
    UpdateJsonData();
  }
  SendRequest(request_id_++, delayed);
}

void
InferContext::SendSequenceInferRequest(uint32_t seq_stat_index, bool delayed)
{
  // Need lock to protect the order of dispatch across worker threads.
  // This also helps in reporting the realistic latencies.
  std::lock_guard<std::mutex> guard(
      sequence_manager_->GetMutex(seq_stat_index));
  if (!early_exit && execute_) {
    sequence_manager_->SetInferSequenceOptions(
        seq_stat_index, infer_data_.options_);

    // Update the inputs if required
    if (using_json_data_) {
      UpdateSeqJsonData(seq_stat_index);
    }

    sequence_manager_->DecrementRemainingQueries(seq_stat_index);

    SendRequest(request_id_++, delayed);
  }
}

void
InferContext::CompleteOngoingSequence(uint32_t seq_stat_index)
{
  std::lock_guard<std::mutex> guard(
      sequence_manager_->GetMutex(seq_stat_index));

  if (sequence_manager_->GetRemainingQueries(seq_stat_index) != 0) {
    sequence_manager_->SetRemainingQueries(seq_stat_index, 1);
    sequence_manager_->SetInferSequenceOptions(
        seq_stat_index, infer_data_.options_);

    if (using_json_data_) {
      UpdateSeqJsonData(seq_stat_index);
    }
    sequence_manager_->DecrementRemainingQueries(seq_stat_index);

    bool is_delayed = false;
    SendRequest(request_id_++, is_delayed);
  }
}

std::chrono::steady_clock::time_point origin{std::chrono::steady_clock::now()};

void
InferContext::SendRequest(const uint64_t request_id, const bool delayed)
{
  std::cout << (std::chrono::steady_clock::now() - origin).count() / 1000000.0
            << "ms" << std::endl;

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
      thread_stat_->request_send_times_.push_back(
          std::chrono::steady_clock::now());
    }

    thread_stat_->idle_timer.Start();
    if (streaming_) {
      thread_stat_->status_ = infer_backend_->AsyncStreamInfer(
          *(infer_data_.options_), infer_data_.valid_inputs_,
          infer_data_.outputs_);
    } else {
      thread_stat_->status_ = infer_backend_->AsyncInfer(
          async_callback_func_, *(infer_data_.options_),
          infer_data_.valid_inputs_, infer_data_.outputs_);
    }
    thread_stat_->idle_timer.Stop();

    total_ongoing_requests_++;
  } else {
    std::chrono::time_point<std::chrono::system_clock> start_time_sync,
        end_time_sync;
    thread_stat_->idle_timer.Start();
    start_time_sync = std::chrono::system_clock::now();
    cb::InferResult* results = nullptr;
    {
      std::lock_guard<std::mutex> lock(thread_stat_->mu_);
      thread_stat_->request_send_times_.push_back(
          std::chrono::steady_clock::now());
    }
    thread_stat_->status_ = infer_backend_->Infer(
        &results, *(infer_data_.options_), infer_data_.valid_inputs_,
        infer_data_.outputs_);
    thread_stat_->idle_timer.Stop();
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
      auto total = end_time_sync - start_time_sync;
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


void
InferContext::UpdateJsonData()
{
  int step_id =
      (data_step_id_ % data_loader_->GetTotalStepsNonSequence()) * batch_size_;
  data_step_id_ += GetNumActiveThreads();
  thread_stat_->status_ =
      UpdateInputs(infer_data_.inputs_, infer_data_.valid_inputs_, 0, step_id);
  if (thread_stat_->status_.IsOk()) {
    thread_stat_->status_ = infer_data_manager_->UpdateValidationOutputs(
        infer_data_.outputs_, 0, step_id, infer_data_.expected_outputs_);
  }
}

void
InferContext::UpdateSeqJsonData(size_t seq_stat_index)
{
  const uint64_t data_stream_id{
      sequence_manager_->GetDataStreamID(seq_stat_index)};
  const size_t remaining_queries{
      sequence_manager_->GetRemainingQueries(seq_stat_index)};
  int step_id = data_loader_->GetTotalSteps(data_stream_id) - remaining_queries;
  thread_stat_->status_ = UpdateInputs(
      infer_data_.inputs_, infer_data_.valid_inputs_, data_stream_id, step_id);
  if (thread_stat_->status_.IsOk()) {
    thread_stat_->status_ = infer_data_manager_->UpdateValidationOutputs(
        infer_data_.outputs_, data_stream_id, step_id,
        infer_data_.expected_outputs_);
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

  RETURN_IF_ERROR(infer_data_manager_->SetInputs(
      inputs, valid_inputs, stream_index, step_index));

  return cb::Error::Success;
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