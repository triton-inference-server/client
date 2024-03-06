// Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <map>
#include <memory>

#include "../client_backend.h"
#include "common.h"
#include "http_client.h"


namespace triton { namespace perfanalyzer { namespace clientbackend {
namespace openai {

class ChatCompletionResult : public InferResult {
 public:
  ChatCompletionResult(
      uint32_t http_code, std::string&& serialized_response, bool is_final,
      bool is_null, const std::string& request_id)
      : http_code_(http_code),
        serialized_response_(std::move(serialized_response)),
        is_final_(is_final), is_null_(is_null), request_id_(request_id)
  {
  }
  virtual ~ChatCompletionResult() = default;

  /// Get the id of the request which generated this response.
  /// \param id Returns the request id that generated the result.
  /// \return Error object indicating success or failure.
  Error Id(std::string* id) const override
  {
    *id = request_id_;
    return Error::Success;
  }


  /// Returns the status of the request.
  /// \return Error object indicating the success or failure of the
  /// request.
  Error RequestStatus() const override
  {
    if ((http_code_ >= 400) && (http_code_ <= 599)) {
      return Error(
          "OpenAI response returns HTTP code " + std::to_string(http_code_));
    }
    return Error::Success;
  }

  /// Returns the raw data of the output.
  /// \return Error object indicating the success or failure of the
  /// request.
  Error RawData(
      const std::string& output_name, const uint8_t** buf,
      size_t* byte_size) const override
  {
    // There is only a single output (and it has no defined name), so we can
    // disregard output_name
    *buf = reinterpret_cast<const uint8_t*>(serialized_response_.c_str());
    *byte_size = serialized_response_.size();
    return Error::Success;
  }

  /// Get final response bool for this response.
  /// \return Error object indicating the success or failure.
  Error IsFinalResponse(bool* is_final_response) const override
  {
    *is_final_response = is_final_;
    return Error::Success;
  };

  /// Get null response bool for this response.
  /// \return Error object indicating the success or failure.
  Error IsNullResponse(bool* is_null_response) const override
  {
    *is_null_response = is_null_;
    return Error::Success;
  };

 private:
  const uint32_t http_code_{200};
  const std::string serialized_response_;
  const bool is_final_{false};
  const bool is_null_{false};
  const std::string request_id_;
};


class ChatCompletionRequest : public HttpRequest {
 public:
  virtual ~ChatCompletionRequest() {}
  ChatCompletionRequest(
      std::function<void(HttpRequest*)>&& completion_callback,
      std::function<void(InferResult*)>&& response_callback,
      const std::string& request_id, const bool verbose = false)
      : HttpRequest(std::move(completion_callback), verbose),
        response_callback_(std::move(response_callback)),
        request_id_(request_id)
  {
  }
  void SendResponse(bool is_final, bool is_null);
  bool is_stream_{false};
  std::function<void(InferResult*)> response_callback_{nullptr};
  // The timers for infer request.
  triton::client::RequestTimers timer_;
  const std::string request_id_;
};

class ChatCompletionClient : public HttpClient {
 public:
  virtual ~ChatCompletionClient() = default;

  /// Create a client that can be used to communicate with the server.
  /// \param server_url The inference server name, port, optional
  /// scheme and optional base path in the following format:
  /// <scheme://>host:port/<base-path>.
  /// \param endpoint The name of the endpoint to send requests to
  /// \param verbose If true generate verbose output when contacting
  /// the inference server.
  /// \param ssl_options Specifies the settings for configuring
  /// SSL encryption and authorization. Providing these options
  /// do not ensure that SSL/TLS will be used in communication.
  /// The use of SSL/TLS depends entirely on the server endpoint.
  /// These options will be ignored if the server_url does not
  /// expose `https://` scheme.
  ChatCompletionClient(
      const std::string& server_url, const std::string& endpoint,
      bool verbose = false,
      const HttpSslOptions& ssl_options = HttpSslOptions());

  /// Simplified AsyncInfer() where the request body is expected to be
  /// prepared by the caller, the client here is responsible to communicate
  /// with a OpenAI-compatible server in both streaming and non-streaming case.
  Error AsyncInfer(
      std::function<void(InferResult*)> callback,
      std::string& serialized_request_body, const std::string& request_id);

  const InferStat& ClientInferStat() { return infer_stat_; }

 private:
  // setup curl handle
  Error PreRunProcessing(CURL* curl, ChatCompletionRequest* request);

  static size_t ResponseHandler(
      void* contents, size_t size, size_t nmemb, void* userp);
  static size_t RequestProvider(
      void* contents, size_t size, size_t nmemb, void* userp);
  static size_t ResponseHeaderHandler(
      void* contents, size_t size, size_t nmemb, void* userp);

  Error UpdateInferStat(const triton::client::RequestTimers& timer);
  InferStat infer_stat_;
};

}}}}  // namespace triton::perfanalyzer::clientbackend::openai
