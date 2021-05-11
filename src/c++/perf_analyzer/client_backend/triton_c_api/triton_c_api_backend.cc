// Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
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

#include "triton_c_api_backend.h"

#include "json_utils.h"
#include "triton_loader.h"

namespace perfanalyzer { namespace clientbackend {

//==============================================================================

Error
TritonLocalClientBackend::Create(
    const std::string& url, const ProtocolType protocol,
    const grpc_compression_algorithm compression_algorithm,
    std::shared_ptr<Headers> http_headers, const std::string& library_directory,
    const std::string& model_repository, const std::string& memory_type,
    const bool verbose, std::unique_ptr<ClientBackend>* client_backend)
{
  std::cout << "backend creating" << std::endl;
  std::unique_ptr<TritonLocalClientBackend> triton_client_backend(
      new TritonLocalClientBackend(
          protocol, compression_algorithm, http_headers));
  TritonLoader::Create(
      library_directory, model_repository, memory_type, verbose);
  *client_backend = std::move(triton_client_backend);
  return Error::Success;
}

Error
TritonLocalClientBackend::ServerExtensions(std::set<std::string>* extensions)
{
  rapidjson::Document server_metadata_json;
  RETURN_IF_ERROR(TritonLoader::ServerMetaData(&server_metadata_json));
  for (const auto& extension : server_metadata_json["extensions"].GetArray()) {
    extensions->insert(
        std::string(extension.GetString(), extension.GetStringLength()));
  }
  return Error::Success;
}

Error
TritonLocalClientBackend::ModelMetadata(
    rapidjson::Document* model_metadata, const std::string& model_name,
    const std::string& model_version)
{
  if (!TritonLoader::ModelIsLoaded()) {
    TritonLoader::LoadModel(model_name, model_version);
  }
  RETURN_IF_ERROR(TritonLoader::ModelMetadata(model_metadata));
  return Error::Success;
}

Error
TritonLocalClientBackend::ModelConfig(
    rapidjson::Document* model_config, const std::string& model_name,
    const std::string& model_version)
{
  if (!TritonLoader::ModelIsLoaded()) {
    TritonLoader::LoadModel(model_name, model_version);
  }
  RETURN_IF_ERROR(TritonLoader::ModelConfig(model_config));
  return Error::Success;
}

Error
TritonLocalClientBackend::Infer(
    InferResult** result, const InferOptions& options,
    const std::vector<InferInput*>& inputs,
    const std::vector<const InferRequestedOutput*>& outputs)
{
  std::vector<nic::InferInput*> triton_inputs;
  ParseInferInputToTriton(inputs, &triton_inputs);

  std::vector<const nic::InferRequestedOutput*> triton_outputs;
  ParseInferRequestedOutputToTriton(outputs, &triton_outputs);

  nic::InferOptions triton_options(options.model_name_);
  ParseInferOptionsToTriton(options, &triton_options);

  nic::InferResult* triton_result;
  RETURN_IF_ERROR(TritonLoader::Infer(
      triton_options, triton_inputs, triton_outputs, &triton_result));

  *result = new TritonLocalInferResult(triton_result);
  return Error::Success;
}

Error
TritonLocalClientBackend::AsyncInfer(
    OnCompleteFn callback, const InferOptions& options,
    const std::vector<InferInput*>& inputs,
    const std::vector<const InferRequestedOutput*>& outputs)
{
  return Error("Async inference not supported with C API");
}

Error
TritonLocalClientBackend::StartStream(OnCompleteFn callback, bool enable_stats)
{
  return Error("Streaming inferences not supported with C API");
}

Error
TritonLocalClientBackend::AsyncStreamInfer(
    const InferOptions& options, const std::vector<InferInput*>& inputs,
    const std::vector<const InferRequestedOutput*>& outputs)
{
  return Error("Async streaming inferences not supported with C API");
}

Error
TritonLocalClientBackend::ClientInferStat(InferStat* infer_stat)
{
  nic::InferStat triton_infer_stat;

  TritonLoader::ClientInferStat(&triton_infer_stat);
  ParseInferStat(triton_infer_stat, infer_stat);
  return Error::Success;
}

Error
TritonLocalClientBackend::ModelInferenceStatistics(
    std::map<ModelIdentifier, ModelStatistics>* model_stats,
    const std::string& model_name, const std::string& model_version)
{
  rapidjson::Document infer_stat_json;
  TritonLoader::ModelInferenceStatistics(
      model_name, model_version, &infer_stat_json);
  ParseStatistics(infer_stat_json, model_stats);

  return Error::Success;
}

void
TritonLocalClientBackend::ParseInferInputToTriton(
    const std::vector<InferInput*>& inputs,
    std::vector<nic::InferInput*>* triton_inputs)
{
  for (const auto input : inputs) {
    triton_inputs->push_back(
        (dynamic_cast<TritonLocalInferInput*>(input))->Get());
  }
}

void
TritonLocalClientBackend::ParseInferRequestedOutputToTriton(
    const std::vector<const InferRequestedOutput*>& outputs,
    std::vector<const nic::InferRequestedOutput*>* triton_outputs)
{
  for (const auto output : outputs) {
    triton_outputs->push_back(
        (dynamic_cast<const TritonLocalInferRequestedOutput*>(output))->Get());
  }
}

void
TritonLocalClientBackend::ParseInferOptionsToTriton(
    const InferOptions& options, nic::InferOptions* triton_options)
{
  triton_options->model_version_ = options.model_version_;
  triton_options->request_id_ = options.request_id_;
  triton_options->sequence_id_ = options.sequence_id_;
  triton_options->sequence_start_ = options.sequence_start_;
  triton_options->sequence_end_ = options.sequence_end_;
}

void
TritonLocalClientBackend::ParseStatistics(
    const rapidjson::Document& infer_stat,
    std::map<ModelIdentifier, ModelStatistics>* model_stats)
{
  model_stats->clear();
  for (const auto& this_stat : infer_stat["model_stats"].GetArray()) {
    auto it = model_stats
                  ->emplace(
                      std::make_pair(
                          this_stat["name"].GetString(),
                          this_stat["version"].GetString()),
                      ModelStatistics())
                  .first;
    it->second.inference_count_ = this_stat["inference_count"].GetUint64();
    it->second.execution_count_ = this_stat["execution_count"].GetUint64();
    it->second.success_count_ =
        this_stat["inference_stats"]["success"]["count"].GetUint64();
    it->second.cumm_time_ns_ =
        this_stat["inference_stats"]["success"]["ns"].GetUint64();
    it->second.queue_time_ns_ =
        this_stat["inference_stats"]["queue"]["ns"].GetUint64();
    it->second.compute_input_time_ns_ =
        this_stat["inference_stats"]["compute_input"]["ns"].GetUint64();
    it->second.compute_infer_time_ns_ =
        this_stat["inference_stats"]["compute_infer"]["ns"].GetUint64();
    it->second.compute_output_time_ns_ =
        this_stat["inference_stats"]["compute_output"]["ns"].GetUint64();
  }
}

void
TritonLocalClientBackend::ParseInferStat(
    const nic::InferStat& triton_infer_stat, InferStat* infer_stat)
{
  infer_stat->completed_request_count =
      triton_infer_stat.completed_request_count;
  infer_stat->cumulative_total_request_time_ns =
      triton_infer_stat.cumulative_total_request_time_ns;
  infer_stat->cumulative_send_time_ns =
      triton_infer_stat.cumulative_send_time_ns;
  infer_stat->cumulative_receive_time_ns =
      triton_infer_stat.cumulative_receive_time_ns;
}

//==============================================================================

Error
TritonLocalInferInput::Create(
    InferInput** infer_input, const std::string& name,
    const std::vector<int64_t>& dims, const std::string& datatype)
{
  TritonLocalInferInput* local_infer_input =
      new TritonLocalInferInput(name, datatype);

  nic::InferInput* triton_infer_input;
  RETURN_IF_TRITON_ERROR(
      nic::InferInput::Create(&triton_infer_input, name, dims, datatype));
  local_infer_input->input_.reset(triton_infer_input);

  *infer_input = local_infer_input;
  return Error::Success;
}

const std::vector<int64_t>&
TritonLocalInferInput::Shape() const
{
  return input_->Shape();
}

Error
TritonLocalInferInput::SetShape(const std::vector<int64_t>& shape)
{
  RETURN_IF_TRITON_ERROR(input_->SetShape(shape));
  return Error::Success;
}

Error
TritonLocalInferInput::Reset()
{
  RETURN_IF_TRITON_ERROR(input_->Reset());
  return Error::Success;
}

Error
TritonLocalInferInput::AppendRaw(const uint8_t* input, size_t input_byte_size)
{
  RETURN_IF_TRITON_ERROR(input_->AppendRaw(input, input_byte_size));
  return Error::Success;
}

Error
TritonLocalInferInput::SetSharedMemory(
    const std::string& name, size_t byte_size, size_t offset)
{
  return Error("Shared memory not supported with C API");
}

TritonLocalInferInput::TritonLocalInferInput(
    const std::string& name, const std::string& datatype)
    : InferInput(BackendKind::TRITON_LOCAL, name, datatype)
{
}


//==============================================================================

Error
TritonLocalInferRequestedOutput::Create(
    InferRequestedOutput** infer_output, const std::string name,
    const size_t class_count)
{
  TritonLocalInferRequestedOutput* local_infer_output =
      new TritonLocalInferRequestedOutput();

  nic::InferRequestedOutput* triton_infer_output;
  RETURN_IF_TRITON_ERROR(nic::InferRequestedOutput::Create(
      &triton_infer_output, name, class_count));
  local_infer_output->output_.reset(triton_infer_output);

  *infer_output = local_infer_output;

  return Error::Success;
}

Error
TritonLocalInferRequestedOutput::SetSharedMemory(
    const std::string& region_name, const size_t byte_size, const size_t offset)
{
  RETURN_IF_TRITON_ERROR(
      output_->SetSharedMemory(region_name, byte_size, offset));
  return Error::Success;
}


TritonLocalInferRequestedOutput::TritonLocalInferRequestedOutput()
    : InferRequestedOutput(BackendKind::TRITON_LOCAL)
{
}

//==============================================================================

TritonLocalInferResult::TritonLocalInferResult(nic::InferResult* result)
{
  result_.reset(result);
}

Error
TritonLocalInferResult::Id(std::string* id) const
{
  RETURN_IF_TRITON_ERROR(result_->Id(id));
  return Error::Success;
}

Error
TritonLocalInferResult::RequestStatus() const
{
  RETURN_IF_TRITON_ERROR(result_->RequestStatus());
  return Error::Success;
}

//==============================================================================

}}  // namespace perfanalyzer::clientbackend
