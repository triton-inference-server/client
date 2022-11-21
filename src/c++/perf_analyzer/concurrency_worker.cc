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

#include <algorithm>

#include "client_backend/client_backend.h"
#include "concurrency_worker.h"
#include "data_loader.h"
#include "perf_utils.h"

namespace triton { namespace perfanalyzer {

// FIXME this was in load_manager.cc in ifdefs and namespaces and in barecode
std::string
TensorToRegionName(std::string name)
{
  // Remove slashes from the name, if any.
  name.erase(
      std::remove_if(
          name.begin(), name.end(),
          [](const char& c) { return ((c == '/') || (c == '\\')); }),
      name.end());
  return name;
}


ConcurrencyWorker::~ConcurrencyWorker()
{
  // FIXME
}

// Function for worker threads.
// If the model is non-sequence model, each worker uses only one context
// to maintain concurrency assigned to worker.
// If the model is sequence model, each worker has to use multiples contexts
// to maintain (sequence) concurrency assigned to worker.
void
ConcurrencyWorker::Infer(
    std::shared_ptr<ConcurrencyManager::ThreadStat> thread_stat,
    std::shared_ptr<ConcurrencyManager::ThreadConfig> thread_config)
{
  // FIXME this struct likely moves into this class
  std::vector<std::unique_ptr<LoadManager::InferContext>> ctxs;
  uint32_t seq_stat_index = 0, ctx_id = 0;
  std::queue<int> free_ctx_ids;

  // Reserve the vectors in case of sequence models. In non-sequence or
  // synchronous mode only one context will be opened hence no need of
  // reserving.
  if (on_sequence_model_ && async_) {
    thread_stat->contexts_stat_.reserve(max_concurrency_);
    ctxs.reserve(max_concurrency_);
  }

  // Variable used to signal request completion
  bool notified = false;
  std::mutex cb_mtx;
  std::condition_variable cb_cv;

  std::atomic<int> total_ongoing_requests(0);
  uint64_t request_id = 0;

  // request_id to start timestamp map
  // FIXME asyncrequestproperties likely moves into this class
  std::map<std::string, LoadManager::AsyncRequestProperties> async_req_map;

  // Callback function for handling asynchronous requests
  const auto callback_func = [&](cb::InferResult* result) {
    uint32_t ctx_id = 0;
    std::shared_ptr<cb::InferResult> result_ptr(result);
    if (thread_stat->cb_status_.IsOk()) {
      // Add the request timestamp to thread Timestamp vector with
      // proper locking
      std::lock_guard<std::mutex> lock(thread_stat->mu_);
      thread_stat->cb_status_ = result_ptr->RequestStatus();
      if (thread_stat->cb_status_.IsOk()) {
        std::chrono::time_point<std::chrono::system_clock> end_time_async;
        end_time_async = std::chrono::system_clock::now();
        std::string request_id;
        thread_stat->cb_status_ = result_ptr->Id(&request_id);
        const auto& it = async_req_map.find(request_id);
        if (it != async_req_map.end()) {
          thread_stat->request_timestamps_.emplace_back(std::make_tuple(
              it->second.start_time_, end_time_async, it->second.sequence_end_,
              false /* delayed */));
          ctx_id = it->second.ctx_id_;
          ctxs[ctx_id]->infer_backend_->ClientInferStat(
              &(thread_stat->contexts_stat_[ctx_id]));
          thread_stat->cb_status_ = ValidateOutputs(*ctxs[ctx_id], result);
          async_req_map.erase(request_id);
        }
      }
    }
    // avoid competition over 'cb_mtx'
    {
      std::lock_guard<std::mutex> lk(cb_mtx);
      free_ctx_ids.push(ctx_id);
      notified = true;
    }

    total_ongoing_requests--;

    cb_cv.notify_all();
  };

  // Specify the function as lambda here to work around the possible callback
  // lifecycle issue when making this a class member function.
  // Note that 'free_ctx_ids' must be reconstruct after the call because
  // this function doesn't utilize 'free_ctx_ids' in the same way as in main
  // loop
  const auto complete_ongoing_sequence_func = [&]() {
    if (!on_sequence_model_) {
      return cb::Error::Success;
    }
    size_t offset = 0;
    for (size_t i = 0; i < thread_config->thread_id_; i++) {
      offset += threads_config_[i]->concurrency_;
    }

    for (size_t ctx_id = 0; ctx_id < ctxs.size(); ++ctx_id) {
      size_t seq_stat_index = offset + ctx_id;

      std::lock_guard<std::mutex> guard(sequence_stat_[seq_stat_index]->mtx_);
      // Complete the sequence if there are remaining queries
      while (sequence_stat_[seq_stat_index]->remaining_queries_ != 0) {
        sequence_stat_[seq_stat_index]->remaining_queries_ = 1;
        SetInferSequenceOptions(seq_stat_index, ctxs[ctx_id]->options_);

        // Update the inputs if required
        if (using_json_data_) {
          int step_id = data_loader_->GetTotalSteps(
                            sequence_stat_[seq_stat_index]->data_stream_id_) -
                        sequence_stat_[seq_stat_index]->remaining_queries_;

          RETURN_IF_ERROR(UpdateInputs(
              ctxs[ctx_id]->inputs_, ctxs[ctx_id]->valid_inputs_,
              sequence_stat_[seq_stat_index]->data_stream_id_, step_id));
          RETURN_IF_ERROR(UpdateValidationOutputs(
              ctxs[ctx_id]->outputs_,
              sequence_stat_[seq_stat_index]->data_stream_id_, step_id,
              ctxs[ctx_id]->expected_outputs_));
        }
        sequence_stat_[seq_stat_index]->remaining_queries_--;

        if (async_) {
          ctxs[ctx_id]->options_->request_id_ = "0";
          if (streaming_) {
            RETURN_IF_ERROR(ctxs[ctx_id]->infer_backend_->AsyncStreamInfer(
                *(ctxs[ctx_id]->options_), ctxs[ctx_id]->inputs_,
                ctxs[ctx_id]->outputs_));
          } else {
            RETURN_IF_ERROR(ctxs[ctx_id]->infer_backend_->AsyncInfer(
                callback_func, *(ctxs[ctx_id]->options_), ctxs[ctx_id]->inputs_,
                ctxs[ctx_id]->outputs_));
          }
          total_ongoing_requests++;
        } else {
          cb::InferResult* results = nullptr;
          auto err = ctxs[ctx_id]->infer_backend_->Infer(
              &results, *(ctxs[ctx_id]->options_), ctxs[ctx_id]->inputs_,
              ctxs[ctx_id]->outputs_);
          if (results != nullptr) {
            delete results;
          }
          RETURN_IF_ERROR(err);
        }
      }
    }
    return cb::Error::Success;
  };

  // run inferencing until receiving exit signal to maintain server load.
  do {
    if (on_sequence_model_) {
      if (!execute_) {
        // Ensures the clean exit of the sequences
        auto status = complete_ongoing_sequence_func();
        if (thread_stat->status_.IsOk()) {
          thread_stat->status_ = status;
        }
        while (total_ongoing_requests != 0) {
          std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
        // Make sure all threads are in sync with the client's stats
        //
        for (size_t i = 0; i < ctxs.size(); ++i) {
          ctxs[i]->infer_backend_->ClientInferStat(
              &(thread_stat->contexts_stat_[i]));
        }
        // Reconstruct 'free_ctx_ids' because complete_ongoing_sequence_func()
        // has destructive side affects
        free_ctx_ids = std::queue<int>();
        for (size_t i = 0; i < ctxs.size(); ++i) {
          free_ctx_ids.push(i);
        }
        // Wait if no request should be sent and it is not exiting
        thread_config->is_paused_ = true;
        std::unique_lock<std::mutex> lock(wake_mutex_);
        wake_signal_.wait(lock, [this]() { return early_exit || execute_; });
      }
    }

    thread_config->is_paused_ = false;

    // Only interact with synchronous mechanism if the worker should wait
    if (thread_config->concurrency_ == 0) {
      // Wait if no request should be sent and it is not exiting
      std::unique_lock<std::mutex> lock(wake_mutex_);
      wake_signal_.wait(lock, [&thread_config]() {
        return early_exit || (thread_config->concurrency_ > 0);
      });
      // Stop executing if concurrency is 0 and early exit is requested
      if (early_exit && thread_config->concurrency_ == 0) {
        break;
      }
    }

    size_t num_reqs = thread_config->concurrency_;

    // If the model is non-sequence model, use one LoadManager::InferContext to
    // maintain concurrency for this thread.
    size_t active_ctx_cnt = on_sequence_model_ ? num_reqs : 1;

    while (active_ctx_cnt > ctxs.size()) {
      {
        std::lock_guard<std::mutex> lock(cb_mtx);
        free_ctx_ids.push(ctxs.size());
      }
      ctxs.emplace_back(new LoadManager::InferContext());
      thread_stat->status_ =
          factory_->CreateClientBackend(&(ctxs.back()->infer_backend_));
      ctxs.back()->options_.reset(new cb::InferOptions(parser_->ModelName()));
      ctxs.back()->options_->model_version_ = parser_->ModelVersion();
      ctxs.back()->options_->model_signature_name_ =
          parser_->ModelSignatureName();
      thread_stat->contexts_stat_.emplace_back();
      if (shared_memory_type_ == SharedMemoryType::NO_SHARED_MEMORY) {
        thread_stat->status_ = PrepareInfer(ctxs.back().get());
      } else {
        thread_stat->status_ = PrepareSharedMemoryInfer(ctxs.back().get());
      }
      if (!thread_stat->status_.IsOk()) {
        return;
      }
      if (streaming_) {
        // Decoupled models should not collect client side statistics
        thread_stat->status_ = ctxs.back()->infer_backend_->StartStream(
            callback_func, (!parser_->IsDecoupled()));
        if (!thread_stat->status_.IsOk()) {
          return;
        }
      }
    }

    // Create async requests such that the number of ongoing requests
    // matches the concurrency level
    // Non-sequence model is 'num_reqs' * 1 ctx
    // Sequence model is 1 request of 1 sequence * 'active_ctx_cnt' ctxs
    while (total_ongoing_requests < (int)num_reqs && early_exit == false) {
      // Update the inputs if required for non-sequence
      if (using_json_data_ && (!on_sequence_model_)) {
        int step_id = (thread_config->non_sequence_data_step_id_ %
                       data_loader_->GetTotalStepsNonSequence()) *
                      batch_size_;
        thread_config->non_sequence_data_step_id_ += active_threads_;
        // There will be only one ctx in non-sequence case
        thread_stat->status_ = UpdateInputs(
            ctxs[ctx_id]->inputs_, ctxs[ctx_id]->valid_inputs_, 0, step_id);
        if (thread_stat->status_.IsOk()) {
          thread_stat->status_ = UpdateValidationOutputs(
              ctxs[ctx_id]->outputs_, 0, step_id,
              ctxs[ctx_id]->expected_outputs_);
        }
        if (!thread_stat->status_.IsOk()) {
          return;
        }
      }

      if (on_sequence_model_) {
        size_t offset = 0;
        for (size_t i = 0; i < thread_config->thread_id_; i++) {
          offset += threads_config_[i]->concurrency_;
        }

        // Find the next available context id to use for this request
        {
          std::lock_guard<std::mutex> lk(cb_mtx);
          ctx_id = free_ctx_ids.front();
          free_ctx_ids.pop();
        }
        seq_stat_index = offset + ctx_id;

        {
          std::lock_guard<std::mutex> guard(
              sequence_stat_[seq_stat_index]->mtx_);
          SetInferSequenceOptions(seq_stat_index, ctxs[ctx_id]->options_);

          // Update the inputs if required
          if (using_json_data_) {
            int step_id = data_loader_->GetTotalSteps(
                              sequence_stat_[seq_stat_index]->data_stream_id_) -
                          sequence_stat_[seq_stat_index]->remaining_queries_;

            thread_stat->status_ = UpdateInputs(
                ctxs[ctx_id]->inputs_, ctxs[ctx_id]->valid_inputs_,
                sequence_stat_[seq_stat_index]->data_stream_id_, step_id);
            if (thread_stat->status_.IsOk()) {
              thread_stat->status_ = UpdateValidationOutputs(
                  ctxs[ctx_id]->outputs_,
                  sequence_stat_[seq_stat_index]->data_stream_id_, step_id,
                  ctxs[ctx_id]->expected_outputs_);
            }
            if (!thread_stat->status_.IsOk()) {
              return;
            }
          }
          sequence_stat_[seq_stat_index]->remaining_queries_--;
        }
      }
      if (async_) {
        ctxs[ctx_id]->options_->request_id_ = std::to_string(request_id++);
        {
          std::lock_guard<std::mutex> lock(thread_stat->mu_);
          auto it = async_req_map
                        .emplace(
                            ctxs[ctx_id]->options_->request_id_,
                            LoadManager::AsyncRequestProperties())
                        .first;
          it->second.start_time_ = std::chrono::system_clock::now();
          it->second.ctx_id_ = ctx_id;
          it->second.sequence_end_ = ctxs[ctx_id]->options_->sequence_end_;
        }
        if (streaming_) {
          thread_stat->status_ = ctxs[ctx_id]->infer_backend_->AsyncStreamInfer(
              *(ctxs[ctx_id]->options_), ctxs[ctx_id]->valid_inputs_,
              ctxs[ctx_id]->outputs_);
        } else {
          thread_stat->status_ = ctxs[ctx_id]->infer_backend_->AsyncInfer(
              callback_func, *(ctxs[ctx_id]->options_),
              ctxs[ctx_id]->valid_inputs_, ctxs[ctx_id]->outputs_);
        }
        if (!thread_stat->status_.IsOk()) {
          return;
        }
      } else {
        std::chrono::time_point<std::chrono::system_clock> start_time_sync,
            end_time_sync;
        start_time_sync = std::chrono::system_clock::now();
        cb::InferResult* results = nullptr;
        thread_stat->status_ = ctxs[ctx_id]->infer_backend_->Infer(
            &results, *(ctxs[ctx_id]->options_), ctxs[ctx_id]->valid_inputs_,
            ctxs[ctx_id]->outputs_);
        if (results != nullptr) {
          if (thread_stat->status_.IsOk()) {
            thread_stat->status_ = ValidateOutputs(*ctxs[ctx_id], results);
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
              start_time_sync, end_time_sync,
              ctxs[ctx_id]->options_->sequence_end_, false /* delayed */));
          thread_stat->status_ = ctxs[ctx_id]->infer_backend_->ClientInferStat(
              &(thread_stat->contexts_stat_[ctx_id]));
          if (!thread_stat->status_.IsOk()) {
            return;
          }
        }
        {
          std::lock_guard<std::mutex> lock(cb_mtx);
          free_ctx_ids.push(ctx_id);
        }
      }
      total_ongoing_requests++;
    }

    if (async_) {
      {
        // If async, then wait for signal from callback.
        std::unique_lock<std::mutex> lk(cb_mtx);
        cb_cv.wait(lk, [&notified] {
          if (notified) {
            notified = false;
            return true;
          }
          return false;
        });
      }
    } else {
      // If synchronous, then all the requests have already been completed.
      total_ongoing_requests = 0;
    }

    if (early_exit || (!thread_stat->cb_status_.IsOk())) {
      // Wait for all callbacks to complete.
      // Loop to ensure all the inflight requests have been completed.
      auto status = complete_ongoing_sequence_func();
      if (thread_stat->status_.IsOk()) {
        thread_stat->status_ = status;
      }
      while (total_ongoing_requests != 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
      }
      // end loop
      break;
    }
  } while (true);
}


cb::Error
ConcurrencyWorker::PrepareInfer(LoadManager::InferContext* ctx)
{
  // Initialize inputs
  for (const auto& input : *(parser_->Inputs())) {
    const uint8_t* data_ptr{nullptr};
    size_t batch1_bytesize;
    // Set input shape before getting the input data
    std::vector<int64_t> shape;
    RETURN_IF_ERROR(data_loader_->GetInputShape(input.second, 0, 0, &shape));
    if (shape.empty() && (backend_kind_ == cb::BackendKind::TRITON)) {
      return cb::Error("unable to set shape for the input", pa::GENERIC_ERROR);
    }

    if ((parser_->MaxBatchSize() != 0) && (!input.second.is_shape_tensor_)) {
      shape.insert(shape.begin(), (int64_t)batch_size_);
    }

    cb::InferInput* infer_input;
    RETURN_IF_ERROR(cb::InferInput::Create(
        &infer_input, backend_kind_, input.first, shape,
        input.second.datatype_));
    ctx->inputs_.push_back(infer_input);

    data_ptr = nullptr;
    RETURN_IF_ERROR(data_loader_->GetInputData(
        input.second, 0, 0, &data_ptr, &batch1_bytesize));

    // Add optional input to request if data was found
    if (data_ptr != nullptr) {
      ctx->valid_inputs_.push_back(infer_input);
    }

    if (!shape.empty()) {
      size_t max_count = (parser_->MaxBatchSize() == 0) ? 1 : batch_size_;
      for (size_t i = 0; i < max_count; ++i) {
        RETURN_IF_ERROR(infer_input->AppendRaw(data_ptr, batch1_bytesize));
      }
    }
  }

  for (const auto& output : *(parser_->Outputs())) {
    std::string region_name(TensorToRegionName(output.first));

    cb::InferRequestedOutput* requested_output;
    RETURN_IF_ERROR(cb::InferRequestedOutput::Create(
        &requested_output, backend_kind_, output.first));
    ctx->outputs_.push_back(requested_output);
  }
  RETURN_IF_ERROR(
      UpdateValidationOutputs(ctx->outputs_, 0, 0, ctx->expected_outputs_));

  return cb::Error::Success;
}

cb::Error
ConcurrencyWorker::PrepareSharedMemoryInfer(LoadManager::InferContext* ctx)
{
  for (const auto& input : *(parser_->Inputs())) {
    std::string region_name(
        TensorToRegionName(input.first) + "_" + std::to_string(0) + "_" +
        std::to_string(0));

    std::vector<int64_t> shape;
    RETURN_IF_ERROR(data_loader_->GetInputShape(input.second, 0, 0, &shape));
    if (!shape.empty()) {
      if ((parser_->MaxBatchSize() != 0) && (!input.second.is_shape_tensor_)) {
        shape.insert(shape.begin(), (int64_t)batch_size_);
      }
    } else {
      return cb::Error("unable to set shape for the input", pa::GENERIC_ERROR);
    }

    cb::InferInput* infer_input;
    RETURN_IF_ERROR(cb::InferInput::Create(
        &infer_input, backend_kind_, input.first, shape,
        input.second.datatype_));
    ctx->inputs_.push_back(infer_input);

    // FIXME: TMA-765 - Shared memory mode does not support optional inputs,
    // currently, and will be implemented in the associated story.
    ctx->valid_inputs_.push_back(infer_input);

    RETURN_IF_ERROR(infer_input->SetSharedMemory(
        region_name, shared_memory_regions_[region_name].byte_size_));
  }

  for (const auto& output : *(parser_->Outputs())) {
    std::string region_name(TensorToRegionName(output.first));

    cb::InferRequestedOutput* requested_output;
    RETURN_IF_ERROR(cb::InferRequestedOutput::Create(
        &requested_output, backend_kind_, output.first));
    ctx->outputs_.push_back(requested_output);

    RETURN_IF_ERROR(requested_output->SetSharedMemory(
        region_name, shared_memory_regions_[region_name].byte_size_));
  }

  return cb::Error::Success;
}

cb::Error
ConcurrencyWorker::UpdateInputs(
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
    RETURN_IF_ERROR(SetInputs(inputs, valid_inputs, stream_index, step_index));
  } else {
    RETURN_IF_ERROR(SetInputsSharedMemory(inputs, stream_index, step_index));
  }

  return cb::Error::Success;
}

cb::Error
ConcurrencyWorker::UpdateValidationOutputs(
    const std::vector<const cb::InferRequestedOutput*>& outputs,
    int stream_index, int step_index,
    std::vector<std::vector<std::pair<const uint8_t*, size_t>>>& data)
{
  data.clear();
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

  for (const auto& output : outputs) {
    const auto& model_output = (*(parser_->Outputs()))[output->Name()];
    const uint8_t* data_ptr;
    size_t batch1_bytesize;
    const int* set_shape_values = nullptr;
    int set_shape_value_cnt = 0;

    std::vector<std::pair<const uint8_t*, size_t>> output_data;
    for (size_t i = 0; i < batch_size_; ++i) {
      RETURN_IF_ERROR(data_loader_->GetOutputData(
          output->Name(), stream_index,
          (step_index + i) % data_loader_->GetTotalSteps(0), &data_ptr,
          &batch1_bytesize));
      if (data_ptr == nullptr) {
        break;
      }
      output_data.emplace_back(data_ptr, batch1_bytesize);
      // Shape tensor only need the first batch element
      if (model_output.is_shape_tensor_) {
        break;
      }
    }
    if (!output_data.empty()) {
      data.emplace_back(std::move(output_data));
    }
  }
  return cb::Error::Success;
}

cb::Error
ConcurrencyWorker::ValidateOutputs(
    const LoadManager::InferContext& ctx, const cb::InferResult* result_ptr)
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


cb::Error
ConcurrencyWorker::SetInputs(
    const std::vector<cb::InferInput*>& inputs,
    std::vector<cb::InferInput*>& valid_inputs, const int stream_index,
    const int step_index)
{
  // Reset inputs for this inference request
  valid_inputs.clear();

  for (const auto& input : inputs) {
    RETURN_IF_ERROR(input->Reset());

    const auto& model_input = (*(parser_->Inputs()))[input->Name()];

    const uint8_t* data_ptr{nullptr};
    size_t batch1_bytesize;
    const int* set_shape_values = nullptr;
    int set_shape_value_cnt = 0;

    // Number of missing pieces of data for optional inputs
    int missing_data_cnt = 0;

    for (size_t i = 0; i < batch_size_; ++i) {
      std::vector<int64_t> shape;
      RETURN_IF_ERROR(data_loader_->GetInputShape(
          model_input, stream_index,
          (step_index + i) % data_loader_->GetTotalSteps(stream_index),
          &shape));
      if ((parser_->MaxBatchSize() != 0) && (!model_input.is_shape_tensor_)) {
        shape.insert(shape.begin(), (int64_t)batch_size_);
      }
      if (!shape.empty()) {
        if (i == 0) {
          input->SetShape(shape);
        } else {
          if (!std::equal(shape.begin(), shape.end(), input->Shape().begin())) {
            return cb::Error(
                "can not batch tensors with different shapes together "
                "(input '" +
                    input->Name() + "' expected shape " +
                    ShapeVecToString(input->Shape(), true /* skip_first */) +
                    " and received " +
                    ShapeVecToString(shape, true /* skip_first */),
                pa::GENERIC_ERROR);
          }
        }
      }
      data_ptr = nullptr;
      RETURN_IF_ERROR(data_loader_->GetInputData(
          model_input, stream_index,
          (step_index + i) % data_loader_->GetTotalSteps(0), &data_ptr,
          &batch1_bytesize));

      // Update number of missing pieces of data for optional inputs to
      // potentially detect error
      if (data_ptr == nullptr) {
        missing_data_cnt++;
        continue;
      }

      if (!model_input.is_shape_tensor_) {
        RETURN_IF_ERROR(input->AppendRaw(data_ptr, batch1_bytesize));
      } else {
        if (i == 0) {
          // Set data only once for shape tensors
          RETURN_IF_ERROR(input->AppendRaw(data_ptr, batch1_bytesize));
          set_shape_values = (const int*)data_ptr;
          set_shape_value_cnt = batch1_bytesize / sizeof(int);
        } else {
          // Validate if the shape values are identical in the batch
          bool is_identical = true;
          if ((size_t)set_shape_value_cnt != (batch1_bytesize / sizeof(int))) {
            is_identical = false;
          } else {
            for (int i = 0; i < set_shape_value_cnt; i++) {
              if (*(set_shape_values + i) != *((const int*)data_ptr + i)) {
                is_identical = false;
                break;
              }
            }
          }
          if (!is_identical) {
            return cb::Error(
                "can not batch shape tensors with different values together "
                "(input '" +
                    input->Name() + "' expected shape values" +
                    ShapeTensorValuesToString(
                        set_shape_values, set_shape_value_cnt) +
                    " and received " +
                    ShapeTensorValuesToString(
                        (int*)data_ptr, (batch1_bytesize / sizeof(int))),
                pa::GENERIC_ERROR);
          }
        }
      }
    }

    // If all optional inputs had data provided, this is a valid input. But if
    // some inferences in the batch provided data for an optional input and
    // some inferences did not, this is an invalid case and an error is
    // thrown.
    if (missing_data_cnt == 0) {
      valid_inputs.push_back(input);
    } else if (missing_data_cnt > 0 && missing_data_cnt < batch_size_) {
      return cb::Error(
          "For batch sizes larger than 1, the same set of inputs must be "
          "specified for each batch. You cannot use different set of "
          "optional "
          "inputs for each individual batch.");
    }
  }
  return cb::Error::Success;
}

cb::Error
ConcurrencyWorker::SetInputsSharedMemory(
    const std::vector<cb::InferInput*>& inputs, const int stream_index,
    const int step_index)
{
  for (const auto& input : inputs) {
    RETURN_IF_ERROR(input->Reset());
    const auto& model_input = (*(parser_->Inputs()))[input->Name()];

    std::string region_name(
        TensorToRegionName(input->Name()) + '_' + std::to_string(stream_index) +
        "_" + std::to_string(step_index));

    std::vector<int64_t> shape;
    RETURN_IF_ERROR(data_loader_->GetInputShape(
        model_input, stream_index, step_index, &shape));
    if (!shape.empty()) {
      if ((parser_->MaxBatchSize() != 0) && (!model_input.is_shape_tensor_)) {
        shape.insert(shape.begin(), (int64_t)batch_size_);
      }
      input->SetShape(shape);
    }
    RETURN_IF_ERROR(input->SetSharedMemory(
        region_name, shared_memory_regions_[region_name].byte_size_));
  }
  return cb::Error::Success;
}

void
ConcurrencyWorker::SetInferSequenceOptions(
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
ConcurrencyWorker::InitNewSequence(int seq_stat_index)
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
ConcurrencyWorker::GetNextSeqId(int seq_stat_index)
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
ConcurrencyWorker::GetRandomSequenceLength(double offset_ratio)
{
  int random_offset = ((2.0 * rand() / double(RAND_MAX)) - 1.0) * offset_ratio *
                      sequence_length_;
  if (int(sequence_length_) + random_offset <= 0) {
    return 1;
  }
  return sequence_length_ + random_offset;
}

}}  // namespace triton::perfanalyzer
