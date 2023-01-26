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

#include <algorithm>

#include "client_backend/client_backend.h"
#include "load_worker.h"
#include "perf_utils.h"

namespace triton { namespace perfanalyzer {

cb::Error
LoadWorker::UpdateInputs(
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

cb::Error
LoadWorker::ValidateOutputs(
    const InferContext& ctx, const cb::InferResult* result_ptr)
{
  // Validate output if set
  if (!ctx.expected_outputs_.empty()) {
    for (size_t i = 0; i < ctx.outputs_.size(); ++i) {
      const uint8_t* buf = nullptr;
      size_t byte_size = 0;
      result_ptr->RawData(ctx.outputs_[i]->Name(), &buf, &byte_size);
      for (const auto& expected : ctx.expected_outputs_[i]) {
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

void
LoadWorker::SetInferSequenceOptions(
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
LoadWorker::InitNewSequence(int seq_stat_index)
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

uint64_t
LoadWorker::GetNextSeqId(int seq_stat_index)
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
LoadWorker::GetRandomSequenceLength(double offset_ratio)
{
  int random_offset = ((2.0 * rand() / double(RAND_MAX)) - 1.0) * offset_ratio *
                      sequence_length_;
  if (int(sequence_length_) + random_offset <= 0) {
    return 1;
  }
  return sequence_length_ + random_offset;
}


void
LoadWorker::SendRequest(
    std::shared_ptr<InferContext> context, const uint32_t ctx_id,
    const uint64_t request_id, const bool delayed,
    cb::OnCompleteFn callback_func,
    std::map<std::string, AsyncRequestProperties>& async_req_map,
    std::shared_ptr<ThreadStat> thread_stat)
{
  if (!thread_stat->status_.IsOk()) {
    return;
  }

  if (async_) {
    context->options_->request_id_ = std::to_string(request_id);
    {
      std::lock_guard<std::mutex> lock(thread_stat->mu_);
      auto it =
          async_req_map
              .emplace(context->options_->request_id_, AsyncRequestProperties())
              .first;
      it->second.start_time_ = std::chrono::system_clock::now();
      it->second.ctx_id_ = ctx_id;
      it->second.sequence_end_ = context->options_->sequence_end_;
      it->second.delayed_ = delayed;
    }
    if (streaming_) {
      thread_stat->status_ = context->infer_backend_->AsyncStreamInfer(
          *(context->options_), context->valid_inputs_, context->outputs_);
    } else {
      thread_stat->status_ = context->infer_backend_->AsyncInfer(
          callback_func, *(context->options_), context->valid_inputs_,
          context->outputs_);
    }
    context->inflight_request_cnt_++;
  } else {
    std::chrono::time_point<std::chrono::system_clock> start_time_sync,
        end_time_sync;
    start_time_sync = std::chrono::system_clock::now();
    cb::InferResult* results = nullptr;
    thread_stat->status_ = context->infer_backend_->Infer(
        &results, *(context->options_), context->valid_inputs_,
        context->outputs_);
    if (results != nullptr) {
      if (thread_stat->status_.IsOk()) {
        thread_stat->status_ = ValidateOutputs(*context, results);
      }
      delete results;
    }
    if (!thread_stat->status_.IsOk()) {
      return;
    }
    end_time_sync = std::chrono::system_clock::now();
    {
      // Add the request timestamp to thread Timestamp vector with proper
      // locking
      std::lock_guard<std::mutex> lock(thread_stat->mu_);
      thread_stat->request_timestamps_.emplace_back(std::make_tuple(
          start_time_sync, end_time_sync, context->options_->sequence_end_,
          delayed));
      thread_stat->status_ = context->infer_backend_->ClientInferStat(
          &(thread_stat->contexts_stat_[ctx_id]));
      if (!thread_stat->status_.IsOk()) {
        return;
      }
    }
  }
}

void
LoadWorker::AsyncCallbackFuncImpl(cb::InferResult* result)
{
  uint32_t ctx_id = 0;
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
        ctx_id = it->second.ctx_id_;
        ctxs_[ctx_id]->infer_backend_->ClientInferStat(
            &(thread_stat_->contexts_stat_[ctx_id]));
        // FIXME not used by conc_worker
        ctxs_[ctx_id]->inflight_request_cnt_--;
        thread_stat_->cb_status_ = ValidateOutputs(*ctxs_[ctx_id], result);
        async_req_map_.erase(request_id);
      }
    }
  }
  AsyncCallbackFinalize(ctx_id);
}

void
LoadWorker::UpdateJsonData(
    std::shared_ptr<DataStepIdTracker> step_id_tracker, const uint32_t ctx_id,
    const size_t num_threads)
{
  size_t curr_step_id = step_id_tracker->GetDataStepId();
  int step_id =
      (curr_step_id % data_loader_->GetTotalStepsNonSequence()) * batch_size_;
  step_id_tracker->SetDataStepId(curr_step_id + num_threads);
  thread_stat_->status_ = UpdateInputs(
      ctxs_[ctx_id]->inputs_, ctxs_[ctx_id]->valid_inputs_, 0, step_id);
  if (thread_stat_->status_.IsOk()) {
    thread_stat_->status_ = memory_manager_->UpdateValidationOutputs(
        ctxs_[ctx_id]->outputs_, 0, step_id, ctxs_[ctx_id]->expected_outputs_);
  }
}

void
LoadWorker::UpdateSeqJsonData(
    const uint32_t ctx_id, std::shared_ptr<SequenceStat> seq_stat)
{
  int step_id = data_loader_->GetTotalSteps(seq_stat->data_stream_id_) -
                seq_stat->remaining_queries_;
  thread_stat_->status_ = UpdateInputs(
      ctxs_[ctx_id]->inputs_, ctxs_[ctx_id]->valid_inputs_,
      seq_stat->data_stream_id_, step_id);
  if (thread_stat_->status_.IsOk()) {
    thread_stat_->status_ = memory_manager_->UpdateValidationOutputs(
        ctxs_[ctx_id]->outputs_, seq_stat->data_stream_id_, step_id,
        ctxs_[ctx_id]->expected_outputs_);
  }
}

void
LoadWorker::CreateContext()
{
  ctxs_.push_back(std::make_shared<InferContext>());

  thread_stat_->status_ =
      factory_->CreateClientBackend(&(ctxs_.back()->infer_backend_));
  ctxs_.back()->options_.reset(new cb::InferOptions(parser_->ModelName()));
  ctxs_.back()->options_->model_version_ = parser_->ModelVersion();
  ctxs_.back()->options_->model_signature_name_ = parser_->ModelSignatureName();

  thread_stat_->contexts_stat_.emplace_back();

  if (shared_memory_type_ == SharedMemoryType::NO_SHARED_MEMORY) {
    thread_stat_->status_ = memory_manager_->PrepareInfer(ctxs_.back().get());
  } else {
    thread_stat_->status_ =
        memory_manager_->PrepareSharedMemoryInfer(ctxs_.back().get());
  }
  if (!thread_stat_->status_.IsOk()) {
    return;
  }

  if (streaming_) {
    // Decoupled models should not collect client side statistics
    thread_stat_->status_ = ctxs_.back()->infer_backend_->StartStream(
        async_callback_func_, (!parser_->IsDecoupled()));
    if (!thread_stat_->status_.IsOk()) {
      return;
    }
  }
}
}}  // namespace triton::perfanalyzer
