// Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

#include "openai_client_backend.h"

#include "openai_infer_input.h"

namespace triton { namespace perfanalyzer { namespace clientbackend {
namespace openai {

//==============================================================================

Error
OpenAiClientBackend::Create(
    const std::string& url, const ProtocolType protocol,
    std::shared_ptr<Headers> http_headers, const bool verbose,
    std::unique_ptr<ClientBackend>* client_backend)
{
  if (protocol == ProtocolType::GRPC) {
    return Error(
        "perf_analyzer does not support gRPC protocol with OpenAI endpoints");
  }
  std::unique_ptr<OpenAiClientBackend> openai_client_backend(
      new OpenAiClientBackend(http_headers));

  openai_client_backend->http_client_.reset(
      new ChatCompletionClient(url, verbose));

  *client_backend = std::move(openai_client_backend);

  return Error::Success;
}

Error
OpenAiClientBackend::AsyncInfer(
    OnCompleteFn callback, const InferOptions& options,
    const std::vector<InferInput*>& inputs,
    const std::vector<const InferRequestedOutput*>& outputs)
{
  if (inputs.size() != 1) {
    return Error("Only expecting one input");
  }

  auto raw_input = dynamic_cast<OpenAiInferInput*>(inputs[0]);
  raw_input->PrepareForRequest();
  RETURN_IF_CB_ERROR(
      http_client_->AsyncInfer(callback, raw_input->DataString(), options.request_id_));
  return Error::Success;
}


Error
OpenAiClientBackend::ClientInferStat(InferStat* infer_stat)
{
  *infer_stat = http_client_->ClientInferStat();
  return Error::Success;
}

//==============================================================================

Error
OpenAiInferRequestedOutput::Create(
    InferRequestedOutput** infer_output, const std::string& name)
{
  OpenAiInferRequestedOutput* local_infer_output =
      new OpenAiInferRequestedOutput(name);

  tc::InferRequestedOutput* openai_infer_output;
  RETURN_IF_TRITON_ERROR(
      tc::InferRequestedOutput::Create(&openai_infer_output, name));
  local_infer_output->output_.reset(openai_infer_output);

  *infer_output = local_infer_output;

  return Error::Success;
}

OpenAiInferRequestedOutput::OpenAiInferRequestedOutput(const std::string& name)
    : InferRequestedOutput(BackendKind::OPENAI, name)
{
}

//==============================================================================


}}}}  // namespace triton::perfanalyzer::clientbackend::openai
