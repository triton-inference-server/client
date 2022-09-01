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
#pragma once

#include <cstdint>
#include <map>
#include <regex>
#include <string>
#include <type_traits>
#include "../../constants.h"
#include "../../metrics.h"
#include "../../perf_utils.h"
#include "../client_backend.h"
#include "grpc_client.h"
#include "http_client.h"
#include "shm_utils.h"

#define RETURN_IF_TRITON_ERROR(S)                          \
  do {                                                     \
    const tc::Error& status__ = (S);                       \
    if (!status__.IsOk()) {                                \
      return Error(status__.Message(), pa::GENERIC_ERROR); \
    }                                                      \
  } while (false)

#define FAIL_IF_TRITON_ERR(X, MSG)                                 \
  {                                                                \
    const tc::Error err = (X);                                     \
    if (!err.IsOk()) {                                             \
      std::cerr << "error: " << (MSG) << ": " << err << std::endl; \
      exit(pa::GENERIC_ERROR);                                     \
    }                                                              \
  }

namespace tc = triton::client;

namespace triton { namespace perfanalyzer { namespace clientbackend {
namespace tritonremote {

#ifndef DOCTEST_CONFIG_DISABLE
class TestTritonClientBackend;
#endif

//==============================================================================
/// TritonClientBackend uses triton client C++ library to communicate with
/// triton inference service.
///
class TritonClientBackend : public ClientBackend {
 public:
  /// Create a triton client backend which can be used to interact with the
  /// server.
  /// \param url The inference server url and port.
  /// \param protocol The protocol type used.
  /// \param ssl_options The SSL options used with client backend.
  /// \param http_headers Map of HTTP headers. The map key/value indicates
  /// the header name/value.
  /// \param verbose Enables the verbose mode.
  /// \param metrics_url The inference server metrics url and port.
  /// \param client_backend Returns a new TritonClientBackend object.
  /// \return Error object indicating success or failure.
  static Error Create(
      const std::string& url, const ProtocolType protocol,
      const SslOptionsBase& ssl_options,
      const std::map<std::string, std::vector<std::string>> trace_options,
      const grpc_compression_algorithm compression_algorithm,
      std::shared_ptr<tc::Headers> http_headers, const bool verbose,
      const std::string& metrics_url,
      std::unique_ptr<ClientBackend>* client_backend);

  /// See ClientBackend::ServerExtensions()
  Error ServerExtensions(std::set<std::string>* server_extensions) override;

  /// See ClientBackend::ModelMetadata()
  Error ModelMetadata(
      rapidjson::Document* model_metadata, const std::string& model_name,
      const std::string& model_version) override;

  /// See ClientBackend::ModelConfig()
  Error ModelConfig(
      rapidjson::Document* model_config, const std::string& model_name,
      const std::string& model_version) override;

  /// See ClientBackend::Infer()
  Error Infer(
      InferResult** result, const InferOptions& options,
      const std::vector<InferInput*>& inputs,
      const std::vector<const InferRequestedOutput*>& outputs) override;

  /// See ClientBackend::AsyncInfer()
  Error AsyncInfer(
      OnCompleteFn callback, const InferOptions& options,
      const std::vector<InferInput*>& inputs,
      const std::vector<const InferRequestedOutput*>& outputs) override;

  /// See ClientBackend::StartStream()
  Error StartStream(OnCompleteFn callback, bool enable_stats) override;

  /// See ClientBackend::AsyncStreamInfer()
  Error AsyncStreamInfer(
      const InferOptions& options, const std::vector<InferInput*>& inputs,
      const std::vector<const InferRequestedOutput*>& outputs) override;

  /// See ClientBackend::ClientInferStat()
  Error ClientInferStat(InferStat* infer_stat) override;

  /// See ClientBackend::ModelInferenceStatistics()
  Error ModelInferenceStatistics(
      std::map<ModelIdentifier, ModelStatistics>* model_stats,
      const std::string& model_name = "",
      const std::string& model_version = "") override;

  /// See ClientBackend::Metrics()
  Error Metrics(triton::perfanalyzer::Metrics& metrics) override;

  /// See ClientBackend::UnregisterAllSharedMemory()
  Error UnregisterAllSharedMemory() override;

  /// See ClientBackend::RegisterSystemSharedMemory()
  Error RegisterSystemSharedMemory(
      const std::string& name, const std::string& key,
      const size_t byte_size) override;

  /// See ClientBackend::RegisterCudaSharedMemory()
  Error RegisterCudaSharedMemory(
      const std::string& name, const cudaIpcMemHandle_t& handle,
      const size_t byte_size) override;

  /// See ClientBackend::CreateSharedMemoryRegion()
  Error CreateSharedMemoryRegion(
      std::string shm_key, size_t byte_size, int* shm_fd) override;

  /// See ClientBackend::MapSharedMemory()
  Error MapSharedMemory(
      int shm_fd, size_t offset, size_t byte_size, void** shm_addr) override;

  /// See ClientBackend::CloseSharedMemory()
  Error CloseSharedMemory(int shm_fd) override;

  /// See ClientBackend::UnlinkSharedMemoryRegion()
  Error UnlinkSharedMemoryRegion(std::string shm_key) override;

  /// See ClientBackend::UnmapSharedMemory()
  Error UnmapSharedMemory(void* shm_addr, size_t byte_size) override;

 private:
  TritonClientBackend(
      const ProtocolType protocol,
      const grpc_compression_algorithm compression_algorithm,
      std::shared_ptr<tc::Headers> http_headers, const std::string& metrics_url)
      : ClientBackend(BackendKind::TRITON), protocol_(protocol),
        compression_algorithm_(compression_algorithm),
        http_headers_(http_headers), metrics_url_(metrics_url)
  {
  }

  void ParseInferInputToTriton(
      const std::vector<InferInput*>& inputs,
      std::vector<tc::InferInput*>* triton_inputs);
  void ParseInferRequestedOutputToTriton(
      const std::vector<const InferRequestedOutput*>& outputs,
      std::vector<const tc::InferRequestedOutput*>* triton_outputs);
  void ParseInferOptionsToTriton(
      const InferOptions& options, tc::InferOptions* triton_options);
  void ParseStatistics(
      const inference::ModelStatisticsResponse& infer_stat,
      std::map<ModelIdentifier, ModelStatistics>* model_stats);
  void ParseStatistics(
      const rapidjson::Document& infer_stat,
      std::map<ModelIdentifier, ModelStatistics>* model_stats);
  void ParseInferStat(
      const tc::InferStat& triton_infer_stat, InferStat* infer_stat);
  void AccessMetricsEndpoint(std::string& metrics_endpoint_text);
  void ParseAndStoreMetrics(
      const std::string& metrics_endpoint_text,
      triton::perfanalyzer::Metrics& metrics);

  template <typename T>
  void ParseAndStoreMetric(
      const std::string& metrics_endpoint_text, const std::string metric_id,
      std::map<std::string, T>& metric_per_gpu)
  {
    std::regex metric_regex(
        R"(\n)" + metric_id + R"(\{gpu_uuid\=\"([^"]+)\"\} (\d+\.?\d*))");
    std::sregex_iterator metric_regex_match_begin{std::sregex_iterator(
        metrics_endpoint_text.begin(), metrics_endpoint_text.end(),
        metric_regex)};

    for (std::sregex_iterator i{metric_regex_match_begin};
         i != std::sregex_iterator(); i++) {
      const std::smatch& match{*i};
      const std::string& gpu_uuid{match[1].str()};
      T metric{};
      if (std::is_same<T, double>::value) {
        metric = std::stod(match[2].str());
      } else if (std::is_same<T, uint64_t>::value) {
        metric = static_cast<uint64_t>(std::stod(match[2].str()));
      }
      metric_per_gpu[gpu_uuid] = metric;
    }
  }

  /// Union to represent the underlying triton client belonging to one of
  /// the protocols
  union TritonClient {
    TritonClient()
    {
      new (&http_client_) std::unique_ptr<tc::InferenceServerHttpClient>{};
    }
    ~TritonClient() {}

    std::unique_ptr<tc::InferenceServerHttpClient> http_client_;
    std::unique_ptr<tc::InferenceServerGrpcClient> grpc_client_;
  } client_;

  const ProtocolType protocol_{UNKNOWN};
  const grpc_compression_algorithm compression_algorithm_{GRPC_COMPRESS_NONE};
  std::shared_ptr<tc::Headers> http_headers_;
  const std::string metrics_url_{""};

#ifndef DOCTEST_CONFIG_DISABLE
  friend TestTritonClientBackend;

 protected:
  TritonClientBackend() = default;
#endif
};

//==============================================================
/// TritonInferInput is a wrapper around InferInput object of
/// triton client library.
///
class TritonInferInput : public InferInput {
 public:
  static Error Create(
      InferInput** infer_input, const std::string& name,
      const std::vector<int64_t>& dims, const std::string& datatype);
  /// Returns the raw InferInput object required by triton client library.
  tc::InferInput* Get() const { return input_.get(); }
  /// See InferInput::Shape()
  const std::vector<int64_t>& Shape() const override;
  /// See InferInput::SetShape()
  Error SetShape(const std::vector<int64_t>& shape) override;
  /// See InferInput::Reset()
  Error Reset() override;
  /// See InferInput::AppendRaw()
  Error AppendRaw(const uint8_t* input, size_t input_byte_size) override;
  /// See InferInput::SetSharedMemory()
  Error SetSharedMemory(
      const std::string& name, size_t byte_size, size_t offset = 0) override;

 private:
  explicit TritonInferInput(
      const std::string& name, const std::string& datatype);

  std::unique_ptr<tc::InferInput> input_;
};

//==============================================================
/// TritonInferRequestedOutput is a wrapper around
/// InferRequestedOutput object of triton client library.
///
class TritonInferRequestedOutput : public InferRequestedOutput {
 public:
  static Error Create(
      InferRequestedOutput** infer_output, const std::string& name,
      const size_t class_count = 0);
  /// Returns the raw InferRequestedOutput object required by triton client
  /// library.
  tc::InferRequestedOutput* Get() const { return output_.get(); }
  // See InferRequestedOutput::SetSharedMemory()
  Error SetSharedMemory(
      const std::string& region_name, const size_t byte_size,
      const size_t offset = 0) override;

 private:
  explicit TritonInferRequestedOutput(const std::string& name);

  std::unique_ptr<tc::InferRequestedOutput> output_;
};

//==============================================================
/// TritonInferResult is a wrapper around InferResult object of
/// triton client library.
///
class TritonInferResult : public InferResult {
 public:
  explicit TritonInferResult(tc::InferResult* result);
  /// See InferResult::Id()
  Error Id(std::string* id) const override;
  /// See InferResult::RequestStatus()
  Error RequestStatus() const override;
  /// See InferResult::RawData()
  Error RawData(
      const std::string& output_name, const uint8_t** buf,
      size_t* byte_size) const override;

 private:
  std::unique_ptr<tc::InferResult> result_;
};

}}}}  // namespace triton::perfanalyzer::clientbackend::tritonremote
