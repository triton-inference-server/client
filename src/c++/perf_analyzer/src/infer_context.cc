// Copyright 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
  thread_stat_->status_ = infer_data_manager_->InitInferData(infer_data_);
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

    SendRequest(
        request_id_++, delayed,
        sequence_manager_->GetSequenceID(seq_stat_index));
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
    SendRequest(
        request_id_++, is_delayed,
        sequence_manager_->GetSequenceID(seq_stat_index));
  }
}

void
InferContext::SendRequest(
    const uint64_t request_id, const bool delayed, const uint64_t sequence_id)
{
  if (!thread_stat_->status_.IsOk()) {
    return;
  }

  thread_stat_->num_sent_requests_++;

  // Parse the request inputs to save in the profile export file
  RequestRecord::RequestInput request_inputs{GetInputs()};

  if (async_) {
    uint64_t unique_request_id{(thread_id_ << 48) | ((request_id << 16) >> 16)};
    infer_data_.options_->request_id_ = std::to_string(unique_request_id);
    {
      std::lock_guard<std::mutex> lock(thread_stat_->mu_);
      auto it = async_req_map_
                    .emplace(infer_data_.options_->request_id_, RequestRecord())
                    .first;
      it->second.request_inputs_ = {request_inputs};
      it->second.start_time_ = std::chrono::system_clock::now();
      it->second.sequence_end_ = infer_data_.options_->sequence_end_;
      it->second.delayed_ = delayed;
      it->second.sequence_id_ = sequence_id;
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
    thread_stat_->status_ = infer_backend_->Infer(
        &results, *(infer_data_.options_), infer_data_.valid_inputs_,
        infer_data_.outputs_);
    thread_stat_->idle_timer.Stop();
    RequestRecord::ResponseOutput response_outputs{};
    if (results != nullptr) {
      if (thread_stat_->status_.IsOk()) {
        response_outputs = GetOutputs(*results);
        thread_stat_->status_ = ValidateOutputs(results);
      }
      delete results;
    }
    if (!thread_stat_->status_.IsOk()) {
      return;
    }
    end_time_sync = std::chrono::system_clock::now();
    std::vector<std::chrono::time_point<std::chrono::system_clock>>
        end_time_syncs{end_time_sync};
    {
      // Add the request record to thread request records vector with proper
      // locking
      std::lock_guard<std::mutex> lock(thread_stat_->mu_);
      auto total = end_time_sync - start_time_sync;
      thread_stat_->request_records_.emplace_back(RequestRecord(
          start_time_sync, std::move(end_time_syncs), {request_inputs},
          {response_outputs}, infer_data_.options_->sequence_end_, delayed,
          sequence_id, false));
      thread_stat_->status_ =
          infer_backend_->ClientInferStat(&(thread_stat_->contexts_stat_[id_]));
      if (!thread_stat_->status_.IsOk()) {
        return;
      }
    }
  }
}

const RequestRecord::RequestInput
InferContext::GetInputs()
{
  RequestRecord::RequestInput input{};
  for (const auto& request_input : infer_data_.valid_inputs_) {
    std::string data_type{request_input->Datatype()};
    const uint8_t* buf{nullptr};
    size_t byte_size{0};
    request_input->RawData(&buf, &byte_size);

    // The first 4 bytes of BYTES data is a 32-bit integer to indicate the size
    // of the rest of the data (which we already know based on byte_size). It
    // should be ignored here, as it isn't part of the actual request
    if (data_type == "BYTES" && byte_size >= 4) {
      buf += 4;
      byte_size -= 4;
    }
    input.emplace(request_input->Name(), RecordData(buf, byte_size, data_type));
  }
  return input;
}

const RequestRecord::ResponseOutput
InferContext::GetOutputs(const cb::InferResult& infer_result)
{
  RequestRecord::ResponseOutput output{};
  for (const auto& requested_output : infer_data_.outputs_) {
    std::string data_type{requested_output->Datatype()};
    const uint8_t* buf{nullptr};
    size_t byte_size{0};
    infer_result.RawData(requested_output->Name(), &buf, &byte_size);

    // The first 4 bytes of BYTES data is a 32-bit integer to indicate the size
    // of the rest of the data (which we already know based on byte_size). It
    // should be ignored here, as it isn't part of the actual response
    if (data_type == "BYTES" && byte_size >= 4) {
      buf += 4;
      byte_size -= 4;
    }
    output.emplace(
        requested_output->Name(), RecordData(buf, byte_size, data_type));
  }
  return output;
}

void
InferContext::UpdateJsonData()
{
  int step_id = (data_step_id_ * batch_size_) % data_loader_->GetTotalSteps(0);
  data_step_id_ += GetNumActiveThreads();
  thread_stat_->status_ =
      infer_data_manager_->UpdateInferData(thread_id_, 0, step_id, infer_data_);
}

void
InferContext::UpdateSeqJsonData(size_t seq_stat_index)
{
  const size_t sequence_length{
      sequence_manager_->GetSequenceLength(seq_stat_index)};
  const size_t remaining_queries{
      sequence_manager_->GetRemainingQueries(seq_stat_index)};
  const uint64_t data_stream_id{
      sequence_manager_->GetDataStreamID(seq_stat_index)};
  const size_t total_steps{data_loader_->GetTotalSteps(data_stream_id)};
  int step_id = (sequence_length - remaining_queries) % total_steps;
  thread_stat_->status_ = infer_data_manager_->UpdateInferData(
      thread_id_, data_stream_id, step_id, infer_data_);
}

cb::Error
InferContext::ValidateOutputs(const cb::InferResult* result_ptr)
{
  // Validate output if set
  if (!infer_data_.expected_outputs_.empty()) {
    for (size_t i = 0; i < infer_data_.expected_outputs_.size(); ++i) {
      const uint8_t* buf = nullptr;
      size_t byte_size = 0;
      for (const auto& expected : infer_data_.expected_outputs_[i]) {
        // Request output by validation output's name explicitly, rather than
        // relying on the array indices being sorted equally in both arrays.
        result_ptr->RawData(expected.name, &buf, &byte_size);
        if (!expected.is_valid) {
          return cb::Error(
              "Expected output can't be invalid", pa::GENERIC_ERROR);
        }
        if (byte_size < expected.batch1_size) {
          return cb::Error(
              "Output size doesn't match expected size", pa::GENERIC_ERROR);
        } else if (memcmp(buf, expected.data_ptr, expected.batch1_size) != 0) {
          return cb::Error(
              "Output doesn't match expected output", pa::GENERIC_ERROR);
        } else {
          buf += expected.batch1_size;
          byte_size -= expected.batch1_size;
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
InferContext::AsyncCallbackFuncImpl(cb::InferResult* result)
{
  std::shared_ptr<cb::InferResult> result_ptr(result);
  bool is_final_response{true};
  if (thread_stat_->cb_status_.IsOk()) {
    // Add the request record to thread request records vector with
    // proper locking
    std::lock_guard<std::mutex> lock(thread_stat_->mu_);
    thread_stat_->cb_status_ = result_ptr->RequestStatus();
    if (thread_stat_->cb_status_.IsOk()) {
      std::string request_id;
      thread_stat_->cb_status_ = result_ptr->Id(&request_id);
      const auto& it = async_req_map_.find(request_id);
      if (it != async_req_map_.end()) {
        bool is_null_response{false};
        thread_stat_->cb_status_ =
            result_ptr->IsNullResponse(&is_null_response);
        if (thread_stat_->cb_status_.IsOk() == false) {
          return;
        }
        it->second.response_timestamps_.push_back(
            std::chrono::system_clock::now());
        it->second.response_outputs_.push_back(GetOutputs(*result));
        num_responses_++;
        if (is_null_response == true) {
          it->second.has_null_last_response_ = true;
        }
        thread_stat_->cb_status_ =
            result_ptr->IsFinalResponse(&is_final_response);
        if (thread_stat_->cb_status_.IsOk() == false) {
          return;
        }
        if (is_final_response) {
          has_received_final_response_ = is_final_response;
          thread_stat_->request_records_.emplace_back(
              it->second.start_time_, it->second.response_timestamps_,
              it->second.request_inputs_, it->second.response_outputs_,
              it->second.sequence_end_, it->second.delayed_,
              it->second.sequence_id_, it->second.has_null_last_response_);
          infer_backend_->ClientInferStat(&(thread_stat_->contexts_stat_[id_]));
          thread_stat_->cb_status_ = ValidateOutputs(result);
          async_req_map_.erase(request_id);
        }
      }
    }
  }

  if (worker_callback_) {
    worker_callback_(id_);
  }

  if (is_final_response) {
    total_ongoing_requests_--;
    num_responses_ = 0;

    if (async_callback_finalize_func_ != nullptr) {
      async_callback_finalize_func_(id_);
    }
  }
}

}}  // namespace triton::perfanalyzer
