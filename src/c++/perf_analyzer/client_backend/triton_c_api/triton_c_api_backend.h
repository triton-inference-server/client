// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#include <string>
#include "../client_backend.h"
#include "triton_loader.h"

#define RETURN_IF_TRITON_ERROR(S)       \
  do {                                  \
    const tc::Error& status__ = (S);    \
    if (!status__.IsOk()) {             \
      return Error(status__.Message()); \
    }                                   \
  } while (false)

#define FAIL_IF_TRITON_ERR(X, MSG)                                 \
  {                                                                \
    const tc::Error err = (X);                                     \
    if (!err.IsOk()) {                                             \
      std::cerr << "error: " << (MSG) << ": " << err << std::endl; \
      exit(1);                                                     \
    }                                                              \
  }

namespace tc = triton::client;
namespace cb = triton::perfanalyzer::clientbackend;
namespace capi = triton::perfanalyzer::clientbackend::tritoncapi;

namespace triton { namespace perfanalyzer { namespace clientbackend {
namespace tritoncapi {

class InferResult;

//==============================================================================
/// TritonCApiClientBackend uses triton client C++ library to communicate with
/// triton inference service. This uses the local C++ library
///
class TritonCApiClientBackend : public ClientBackend {
 public:
  /// Create a triton client backend which can be used to interact with the
  /// server.
  /// \param triton_server_path Tritonserver library that contains
  /// lib/libtritonserver.so.
  /// \param model_repository_path The model repository.
  /// \param memory_type Type of memory used in Triton Server.
  /// \param verbose Enables the verbose mode of TritonServer.
  /// \param client_backend Returns a new TritonCApiClientBackend object.
  /// \return Error object indicating success
  /// or failure.
  static Error Create(
      const std::string& triton_server_path,
      const std::string& model_repository_path, const std::string& memory_type,
      const bool verbose, std::unique_ptr<ClientBackend>* client_backend);

  ~TritonCApiClientBackend() { TritonLoader::Delete(); }

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
      cb::InferResult** result, const InferOptions& options,
      const std::vector<InferInput*>& inputs,
      const std::vector<const InferRequestedOutput*>& outputs) override;

  /// See ClientBackend::ClientInferStat()
  Error ClientInferStat(InferStat* infer_stat) override;

  /// See ClientBackend::ModelInferenceStatistics()
  Error ModelInferenceStatistics(
      std::map<ModelIdentifier, ModelStatistics>* model_stats,
      const std::string& model_name = "",
      const std::string& model_version = "") override;

 private:
  TritonCApiClientBackend() : ClientBackend(BackendKind::TRITON_C_API) {}
  void ParseInferInputToTriton(
      const std::vector<InferInput*>& inputs,
      std::vector<tc::InferInput*>* triton_inputs);
  void ParseInferRequestedOutputToTriton(
      const std::vector<const InferRequestedOutput*>& outputs,
      std::vector<const tc::InferRequestedOutput*>* triton_outputs);
  void ParseInferOptionsToTriton(
      const InferOptions& options, tc::InferOptions* triton_options);
  void ParseStatistics(
      const rapidjson::Document& infer_stat,
      std::map<ModelIdentifier, ModelStatistics>* model_stats);
  void ParseInferStat(
      const tc::InferStat& triton_infer_stat, InferStat* infer_stat);
};

//==============================================================
/// TritonCApiInferInput is a wrapper around InferInput object of
/// triton client library.
///
class TritonCApiInferInput : public InferInput {
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

 private:
  explicit TritonCApiInferInput(
      const std::string& name, const std::string& datatype);

  std::unique_ptr<tc::InferInput> input_;
};

//==============================================================
/// TritonCApiInferRequestedOutput is a wrapper around
/// InferRequestedOutput object of triton client library.
///
class TritonCApiInferRequestedOutput : public InferRequestedOutput {
 public:
  static Error Create(
      InferRequestedOutput** infer_output, const std::string& name,
      const size_t class_count = 0);
  /// Returns the raw InferRequestedOutput object required by triton client
  /// library.
  tc::InferRequestedOutput* Get() const { return output_.get(); }

 private:
  explicit TritonCApiInferRequestedOutput(const std::string& name);

  std::unique_ptr<tc::InferRequestedOutput> output_;
};

//==============================================================
/// TritonCApiInferResult is a wrapper around InferResult object of
/// the C API library.
///
class TritonCApiInferResult : public cb::InferResult {
 public:
  explicit TritonCApiInferResult(capi::InferResult* result);
  /// See InferResult::Id()
  Error Id(std::string* id) const override;
  /// See InferResult::RequestStatus()
  Error RequestStatus() const override;
  /// See InferResult::RawData()
  Error RawData(
      const std::string& output_name, const uint8_t** buf,
      size_t* byte_size) const override;

 private:
  std::unique_ptr<capi::InferResult> result_;
};

}}}}  // namespace triton::perfanalyzer::clientbackend::tritoncapi
