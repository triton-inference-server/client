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
#pragma once

#include "../client_backend.h"
#include "common.h"


namespace tc = triton::client;

namespace triton { namespace perfanalyzer { namespace clientbackend {
namespace openai {

class InferResult;
class HttpInferRequest;

//==============================================================================
/// An HttpClient object is used to perform any kind of communication with the
/// OpenAi service using <TODO: FILL IN>
///
/// \code
///   std::unique_ptr<HttpClient> client;
///   HttpClient::Create(&client, "localhost:8080");
///   ...
///   ...
/// \endcode
///
class HttpClient : public tc::InferenceServerClient {
 public:
  ~HttpClient();

  /// TODO: Adjust as needed
  /// Create a client that can be used to communicate with the server.
  /// \param client Returns a new InferenceServerHttpClient object.
  /// \param server_url The inference server name and port.
  /// \param verbose If true generate verbose output when contacting
  /// the inference server.
  /// \return Error object indicating success or failure.
  static Error Create(
      std::unique_ptr<HttpClient>* client, const std::string& server_url,
      const bool verbose);

 private:
  HttpClient(const std::string& url, bool verbose);

  // The server url
  const std::string url_;
};

//======================================================================

class InferResult {
 public:
  static Error Create(
      InferResult** infer_result,
      std::shared_ptr<HttpInferRequest> infer_request);
  Error RequestStatus() const { return Error::Success; }      // TODO FIXME TKG
  Error Id(std::string* id) const { return Error::Success; }  // TODO FIXME TKG

 private:
  InferResult(std::shared_ptr<HttpInferRequest> infer_request);

  // The status of the inference
  Error status_;
  // The pointer to the HttpInferRequest object
  std::shared_ptr<HttpInferRequest> infer_request_;
};

//======================================================================

}}}}  // namespace triton::perfanalyzer::clientbackend::openai
