// Copyright 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "client_backend.h"

#include "triton/triton_client_backend.h"

#ifdef TRITON_ENABLE_PERF_ANALYZER_C_API
#include "triton_c_api/triton_c_api_backend.h"
#endif  // TRITON_ENABLE_PERF_ANALYZER_C_API

#ifdef TRITON_ENABLE_PERF_ANALYZER_TFS
#include "tensorflow_serving/tfserve_client_backend.h"
#endif  // TRITON_ENABLE_PERF_ANALYZER_TFS

#ifdef TRITON_ENABLE_PERF_ANALYZER_TS
#include "torchserve/torchserve_client_backend.h"
#endif  // TRITON_ENABLE_PERF_ANALYZER_TS

namespace triton { namespace perfanalyzer { namespace clientbackend {

//================================================

const Error Error::Success;
const Error Error::Failure("");

Error::Error()
{
  has_error_ = false;
}

Error::Error(const std::string& msg) : msg_(msg)
{
  has_error_ = true;
}

std::ostream&
operator<<(std::ostream& out, const Error& err)
{
  if (!err.msg_.empty()) {
    out << err.msg_ << std::endl;
  }
  return out;
}

//================================================

std::string
BackendKindToString(const BackendKind kind)
{
  switch (kind) {
    case TRITON:
      return std::string("TRITON");
      break;
    case TENSORFLOW_SERVING:
      return std::string("TENSORFLOW_SERVING");
      break;
    case TORCHSERVE:
      return std::string("TORCHSERVE");
      break;
    case TRITON_C_API:
      return std::string("TRITON_C_API");
      break;
    default:
      return std::string("UNKNOWN");
      break;
  }
}

grpc_compression_algorithm
BackendToGrpcType(const GrpcCompressionAlgorithm compression_algorithm)
{
  if (compression_algorithm == COMPRESS_STREAM_GZIP) {
    return grpc_compression_algorithm::GRPC_COMPRESS_STREAM_GZIP;
  } else if (compression_algorithm == COMPRESS_DEFLATE) {
    return grpc_compression_algorithm::GRPC_COMPRESS_DEFLATE;
  } else if (compression_algorithm == COMPRESS_GZIP) {
    return grpc_compression_algorithm::GRPC_COMPRESS_GZIP;
  } else {
    return grpc_compression_algorithm::GRPC_COMPRESS_NONE;
  }
}

//================================================

//
// ClientBackendFactory
//
Error
ClientBackendFactory::Create(
    const BackendKind kind, const std::string& url, const ProtocolType protocol,
    const SslOptionsBase& ssl_options,
    const GrpcCompressionAlgorithm compression_algorithm,
    std::shared_ptr<Headers> http_headers,
    const std::string& triton_server_path,
    const std::string& model_repository_path, const std::string& memory_type,
    const bool verbose, std::shared_ptr<ClientBackendFactory>* factory)
{
  factory->reset(new ClientBackendFactory(
      kind, url, protocol, ssl_options, compression_algorithm, http_headers,
      triton_server_path, model_repository_path, memory_type, verbose));
  return Error::Success;
}

Error
ClientBackendFactory::CreateClientBackend(
    std::unique_ptr<ClientBackend>* client_backend)
{
  RETURN_IF_CB_ERROR(ClientBackend::Create(
      kind_, url_, protocol_, ssl_options_, compression_algorithm_,
      http_headers_, verbose_, triton_server_path, model_repository_path_,
      memory_type_, client_backend));
  return Error::Success;
}

//
// ClientBackend
//
Error
ClientBackend::Create(
    const BackendKind kind, const std::string& url, const ProtocolType protocol,
    const SslOptionsBase& ssl_options,
    const GrpcCompressionAlgorithm compression_algorithm,
    std::shared_ptr<Headers> http_headers, const bool verbose,
    const std::string& triton_server_path,
    const std::string& model_repository_path, const std::string& memory_type,
    std::unique_ptr<ClientBackend>* client_backend)
{
  std::unique_ptr<ClientBackend> local_backend;
  if (kind == TRITON) {
    RETURN_IF_CB_ERROR(tritonremote::TritonClientBackend::Create(
        url, protocol, ssl_options, BackendToGrpcType(compression_algorithm),
        http_headers, verbose, &local_backend));
  }
#ifdef TRITON_ENABLE_PERF_ANALYZER_TFS
  else if (kind == TENSORFLOW_SERVING) {
    RETURN_IF_CB_ERROR(tfserving::TFServeClientBackend::Create(
        url, protocol, BackendToGrpcType(compression_algorithm), http_headers,
        verbose, &local_backend));
  }
#endif  // TRITON_ENABLE_PERF_ANALYZER_TFS
#ifdef TRITON_ENABLE_PERF_ANALYZER_TS
  else if (kind == TORCHSERVE) {
    RETURN_IF_CB_ERROR(torchserve::TorchServeClientBackend::Create(
        url, protocol, http_headers, verbose, &local_backend));
  }
#endif  // TRITON_ENABLE_PERF_ANALYZER_TS
#ifdef TRITON_ENABLE_PERF_ANALYZER_C_API
  else if (kind == TRITON_C_API) {
    RETURN_IF_CB_ERROR(tritoncapi::TritonCApiClientBackend::Create(
        triton_server_path, model_repository_path, memory_type, verbose,
        &local_backend));
  }
#endif  // TRITON_ENABLE_PERF_ANALYZER_C_API
  else {
    return Error("unsupported client backend requested");
  }

  *client_backend = std::move(local_backend);

  return Error::Success;
}

Error
ClientBackend::ServerExtensions(std::set<std::string>* server_extensions)
{
  return Error(
      "client backend of kind " + BackendKindToString(kind_) +
      " does not support ServerExtensions API");
}

Error
ClientBackend::ModelMetadata(
    rapidjson::Document* model_metadata, const std::string& model_name,
    const std::string& model_version)
{
  return Error(
      "client backend of kind " + BackendKindToString(kind_) +
      " does not support ModelMetadata API");
}

Error
ClientBackend::ModelConfig(
    rapidjson::Document* model_config, const std::string& model_name,
    const std::string& model_version)
{
  return Error(
      "client backend of kind " + BackendKindToString(kind_) +
      " does not support ModelConfig API");
}

Error
ClientBackend::Infer(
    InferResult** result, const InferOptions& options,
    const std::vector<InferInput*>& inputs,
    const std::vector<const InferRequestedOutput*>& outputs)
{
  return Error(
      "client backend of kind " + BackendKindToString(kind_) +
      " does not support Infer API");
}

Error
ClientBackend::AsyncInfer(
    OnCompleteFn callback, const InferOptions& options,
    const std::vector<InferInput*>& inputs,
    const std::vector<const InferRequestedOutput*>& outputs)
{
  return Error(
      "client backend of kind " + BackendKindToString(kind_) +
      " does not support AsyncInfer API");
}

Error
ClientBackend::StartStream(OnCompleteFn callback, bool enable_stats)
{
  return Error(
      "client backend of kind " + BackendKindToString(kind_) +
      " does not support StartStream API");
}

Error
ClientBackend::AsyncStreamInfer(
    const InferOptions& options, const std::vector<InferInput*>& inputs,
    const std::vector<const InferRequestedOutput*>& outputs)
{
  return Error(
      "client backend of kind " + BackendKindToString(kind_) +
      " does not support AsyncStreamInfer API");
}

Error
ClientBackend::ClientInferStat(InferStat* infer_stat)
{
  return Error(
      "client backend of kind " + BackendKindToString(kind_) +
      " does not support ClientInferStat API");
}

Error
ClientBackend::ModelInferenceStatistics(
    std::map<ModelIdentifier, ModelStatistics>* model_stats,
    const std::string& model_name, const std::string& model_version)
{
  return Error(
      "client backend of kind " + BackendKindToString(kind_) +
      " does not support ModelInferenceStatistics API");
}

Error
ClientBackend::UnregisterAllSharedMemory()
{
  return Error(
      "client backend of kind " + BackendKindToString(kind_) +
      " does not support UnregisterAllSharedMemory API");
}

Error
ClientBackend::RegisterSystemSharedMemory(
    const std::string& name, const std::string& key, const size_t byte_size)
{
  return Error(
      "client backend of kind " + BackendKindToString(kind_) +
      " does not support RegisterSystemSharedMemory API");
}

Error
ClientBackend::RegisterCudaSharedMemory(
    const std::string& name, const cudaIpcMemHandle_t& handle,
    const size_t byte_size)
{
  return Error(
      "client backend of kind " + BackendKindToString(kind_) +
      " does not support RegisterCudaSharedMemory API");
}

//
// Shared Memory Utilities
//
Error
ClientBackend::CreateSharedMemoryRegion(
    std::string shm_key, size_t byte_size, int* shm_fd)
{
  return Error(
      "client backend of kind " + BackendKindToString(kind_) +
      " does not support CreateSharedMemoryRegion()");
}


Error
ClientBackend::MapSharedMemory(
    int shm_fd, size_t offset, size_t byte_size, void** shm_addr)
{
  return Error(
      "client backend of kind " + BackendKindToString(kind_) +
      " does not support MapSharedMemory()");
}


Error
ClientBackend::CloseSharedMemory(int shm_fd)
{
  return Error(
      "client backend of kind " + BackendKindToString(kind_) +
      " does not support CloseSharedMemory()");
}

Error
ClientBackend::UnlinkSharedMemoryRegion(std::string shm_key)
{
  return Error(
      "client backend of kind " + BackendKindToString(kind_) +
      " does not support UnlinkSharedMemoryRegion()");
}

Error
ClientBackend::UnmapSharedMemory(void* shm_addr, size_t byte_size)
{
  return Error(
      "client backend of kind " + BackendKindToString(kind_) +
      " does not support UnmapSharedMemory()");
}


ClientBackend::ClientBackend(const BackendKind kind) : kind_(kind) {}

//
// InferInput
//
Error
InferInput::Create(
    InferInput** infer_input, const BackendKind kind, const std::string& name,
    const std::vector<int64_t>& dims, const std::string& datatype)
{
  if (kind == TRITON) {
    RETURN_IF_CB_ERROR(tritonremote::TritonInferInput::Create(
        infer_input, name, dims, datatype));
  }
#ifdef TRITON_ENABLE_PERF_ANALYZER_TFS
  else if (kind == TENSORFLOW_SERVING) {
    RETURN_IF_CB_ERROR(tfserving::TFServeInferInput::Create(
        infer_input, name, dims, datatype));
  }
#endif  // TRITON_ENABLE_PERF_ANALYZER_TFS
#ifdef TRITON_ENABLE_PERF_ANALYZER_TS
  else if (kind == TORCHSERVE) {
    RETURN_IF_CB_ERROR(torchserve::TorchServeInferInput::Create(
        infer_input, name, dims, datatype));
  }
#endif  // TRITON_ENABLE_PERF_ANALYZER_TS
#ifdef TRITON_ENABLE_PERF_ANALYZER_C_API
  else if (kind == TRITON_C_API) {
    RETURN_IF_CB_ERROR(tritoncapi::TritonCApiInferInput::Create(
        infer_input, name, dims, datatype));
  }
#endif  // TRITON_ENABLE_PERF_ANALYZER_C_API
  else {
    return Error(
        "unsupported client backend provided to create InferInput object");
  }

  return Error::Success;
}

Error
InferInput::SetShape(const std::vector<int64_t>& shape)
{
  return Error(
      "client backend of kind " + BackendKindToString(kind_) +
      " does not support SetShape() for InferInput");
}

Error
InferInput::Reset()
{
  return Error(
      "client backend of kind " + BackendKindToString(kind_) +
      " does not support Reset() for InferInput");
}

Error
InferInput::AppendRaw(const uint8_t* input, size_t input_byte_size)
{
  return Error(
      "client backend of kind " + BackendKindToString(kind_) +
      " does not support AppendRaw() for InferInput");
}

Error
InferInput::SetSharedMemory(
    const std::string& name, size_t byte_size, size_t offset)
{
  return Error(
      "client backend of kind " + BackendKindToString(kind_) +
      " does not support SetSharedMemory() for InferInput");
}


InferInput::InferInput(
    const BackendKind kind, const std::string& name,
    const std::string& datatype)
    : kind_(kind), name_(name), datatype_(datatype)
{
}

//
// InferRequestedOutput
//
Error
InferRequestedOutput::Create(
    InferRequestedOutput** infer_output, const BackendKind kind,
    const std::string& name, const size_t class_count)
{
  if (kind == TRITON) {
    RETURN_IF_CB_ERROR(tritonremote::TritonInferRequestedOutput::Create(
        infer_output, name, class_count));
  }
#ifdef TRITON_ENABLE_PERF_ANALYZER_TFS
  else if (kind == TENSORFLOW_SERVING) {
    RETURN_IF_CB_ERROR(
        tfserving::TFServeInferRequestedOutput::Create(infer_output, name));
  }
#endif  // TRITON_ENABLE_PERF_ANALYZER_TFS
#ifdef TRITON_ENABLE_PERF_ANALYZER_C_API
  else if (kind == TRITON_C_API) {
    RETURN_IF_CB_ERROR(tritoncapi::TritonCApiInferRequestedOutput::Create(
        infer_output, name, class_count));
  }
#endif  // TRITON_ENABLE_PERF_ANALYZER_C_API
  else {
    return Error(
        "unsupported client backend provided to create InferRequestedOutput "
        "object");
  }

  return Error::Success;
}

Error
InferRequestedOutput::SetSharedMemory(
    const std::string& region_name, size_t byte_size, size_t offset)
{
  return Error(
      "client backend of kind " + BackendKindToString(kind_) +
      " does not support SetSharedMemory() for InferRequestedOutput");
}

InferRequestedOutput::InferRequestedOutput(
    const BackendKind kind, const std::string& name)
    : kind_(kind), name_(name)
{
}

}}}  // namespace triton::perfanalyzer::clientbackend
