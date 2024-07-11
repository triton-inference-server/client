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

// Include this first to make sure we are a friend of common classes.
#define TRITON_INFERENCE_SERVER_CLIENT_CLASS InferenceServerHttpClient
#include "openai_client.h"

#include <curl/curl.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <climits>
#include <cstdint>
#include <deque>
#include <iostream>
#include <string>
#include <utility>

#include "common.h"

#ifdef TRITON_ENABLE_ZLIB
#include <zlib.h>
#endif

extern "C" {
#include "cencode.h"
}

#ifdef _WIN32
#define strncasecmp(x, y, z) _strnicmp(x, y, z)
#undef min  // NOMINMAX did not resolve std::min compile error
#endif      //_WIN32

namespace triton { namespace perfanalyzer { namespace clientbackend {
namespace openai {

//==============================================================================

void
ChatCompletionRequest::SendResponse(bool is_final, bool is_null)
{
  final_response_sent_ = is_final;
  response_callback_(new ChatCompletionResult(
      http_code_, std::move(response_buffer_), is_final, is_null, request_id_));
}

ChatCompletionClient::ChatCompletionClient(
    const std::string& url, const std::string& endpoint, bool verbose,
    const HttpSslOptions& ssl_options)
    : HttpClient(std::string(url + "/" + endpoint), verbose, ssl_options)
{
}

size_t
ChatCompletionClient::RequestProvider(
    void* contents, size_t size, size_t nmemb, void* userp)
{
  auto request = reinterpret_cast<ChatCompletionRequest*>(userp);

  size_t input_bytes = 0;
  request->GetNextInput(
      reinterpret_cast<uint8_t*>(contents), size * nmemb, &input_bytes);

  request->timer_.CaptureTimestamp(
      triton::client::RequestTimers::Kind::SEND_END);

  return input_bytes;
}

size_t
ChatCompletionClient::ResponseHeaderHandler(
    void* contents, size_t size, size_t nmemb, void* userp)
{
  auto request = reinterpret_cast<ChatCompletionRequest*>(userp);

  char* buf = reinterpret_cast<char*>(contents);
  size_t byte_size = size * nmemb;

  std::string hdr(buf, byte_size);
  std::transform(hdr.begin(), hdr.end(), hdr.begin(), [](unsigned char c) {
    return std::tolower(c);
  });
  if (hdr.find("content-type") != std::string::npos &&
      hdr.find("text/event-stream") != std::string::npos) {
    request->is_stream_ = true;
  }
  return byte_size;
}

size_t
ChatCompletionClient::ResponseHandler(
    void* contents, size_t size, size_t nmemb, void* userp)
{
  // [TODO TMA-1666] verify if the SSE responses received are complete, or the
  // response need to be stitched first. To verify, print out the received
  // responses from SendResponse() to make sure the OpenAI server doesn't chunk
  // the HTTP responses in the way that misaligns with the SSE responses. Reason
  // of not stitching responses now is that it is a bit complicated that to make
  // the write callback bulletproof is to assume the response can be chunked at
  // arbitrary position, then bake in checking for SSE style (data:.*\n\n) by
  // iterating all received buffer character by character.
  size_t result_bytes = size * nmemb;
  // return early if the response is empty as the response handling is
  // triggered by the content of the response.
  if (result_bytes == 0) {
    return result_bytes;
  }

  auto request = reinterpret_cast<ChatCompletionRequest*>(userp);
  if (request->timer_.Timestamp(
          triton::client::RequestTimers::Kind::RECV_START) == 0) {
    request->timer_.CaptureTimestamp(
        triton::client::RequestTimers::Kind::RECV_START);
  }

  char* buf = reinterpret_cast<char*>(contents);
  request->response_buffer_.append(buf, result_bytes);
  // Send response now if streaming, otherwise wait until request has been
  // completed
  if (request->is_stream_) {
    auto done_signal =
        (request->response_buffer_.find("data: [DONE]") != std::string::npos);
    request->SendResponse(
        done_signal /* is_final */, done_signal /* is_null */);
  }

  // ResponseHandler may be called multiple times so we overwrite
  // RECV_END so that we always have the time of the last.
  request->timer_.CaptureTimestamp(
      triton::client::RequestTimers::Kind::RECV_END);

  return result_bytes;
}


Error
ChatCompletionClient::AsyncInfer(
    std::function<void(InferResult*)> callback,
    std::string& serialized_request_body, const std::string& request_id,
    const Headers& headers)
{
  if (callback == nullptr) {
    return Error(
        "Callback function must be provided along with AsyncInfer() call.");
  }

  auto completion_callback = [this](HttpRequest* req) {
    auto request = static_cast<ChatCompletionRequest*>(req);
    request->timer_.CaptureTimestamp(
        triton::client::RequestTimers::Kind::REQUEST_END);
    UpdateInferStat(request->timer_);

    // Send final response on request completion
    // if it has not already been sent.
    // (e.g. in the case of seeing [DONE] in streaming case)
    if (!request->IsFinalResponseSent()) {
      request->SendResponse(true /* is_final */, false /* is_null */);
    }
  };
  std::unique_ptr<HttpRequest> request(new ChatCompletionRequest(
      std::move(completion_callback), std::move(callback), request_id,
      verbose_));
  auto raw_request = static_cast<ChatCompletionRequest*>(request.get());
  raw_request->timer_.CaptureTimestamp(
      triton::client::RequestTimers::Kind::REQUEST_START);
  request->AddInput(
      reinterpret_cast<uint8_t*>(serialized_request_body.data()),
      serialized_request_body.size());

  CURL* multi_easy_handle = curl_easy_init();
  Error err = PreRunProcessing(multi_easy_handle, raw_request, headers);
  if (!err.IsOk()) {
    curl_easy_cleanup(multi_easy_handle);
    return err;
  }

  raw_request->timer_.CaptureTimestamp(
      triton::client::RequestTimers::Kind::SEND_START);
  Send(multi_easy_handle, std::move(request));
  return Error::Success;
}

Error
ChatCompletionClient::PreRunProcessing(
    CURL* curl, ChatCompletionRequest* request, const Headers& headers)
{
  curl_easy_setopt(curl, CURLOPT_URL, url_.c_str());
  curl_easy_setopt(curl, CURLOPT_USERAGENT, "libcurl-agent/1.0");
  curl_easy_setopt(curl, CURLOPT_POST, 1L);
  curl_easy_setopt(curl, CURLOPT_TCP_NODELAY, 1L);

  if (verbose_) {
    curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);
  }

  const long buffer_byte_size = 16 * 1024 * 1024;
  curl_easy_setopt(curl, CURLOPT_UPLOAD_BUFFERSIZE, buffer_byte_size);
  curl_easy_setopt(curl, CURLOPT_BUFFERSIZE, buffer_byte_size);

  // request data provided by RequestProvider()
  curl_easy_setopt(curl, CURLOPT_READFUNCTION, RequestProvider);
  curl_easy_setopt(curl, CURLOPT_READDATA, request);

  // response headers handled by ResponseHeaderHandler()
  curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, ResponseHeaderHandler);
  curl_easy_setopt(curl, CURLOPT_HEADERDATA, request);

  // response data handled by ResponseHandler()
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, ResponseHandler);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, request);

  const curl_off_t post_byte_size = request->total_input_byte_size_;
  curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE_LARGE, post_byte_size);

  SetSSLCurlOptions(curl);

  struct curl_slist* list = nullptr;
  list = curl_slist_append(list, "Expect:");
  list = curl_slist_append(list, "Content-Type: application/json");

  for (const auto& pr : headers) {
    std::string hdr = pr.first + ": " + pr.second;
    list = curl_slist_append(list, hdr.c_str());
  }

  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, list);

  // The list will be freed when the request is destructed
  request->header_list_ = list;

  return Error::Success;
}

Error
ChatCompletionClient::UpdateInferStat(
    const triton::client::RequestTimers& timer)
{
  const uint64_t request_time_ns = timer.Duration(
      triton::client::RequestTimers::Kind::REQUEST_START,
      triton::client::RequestTimers::Kind::REQUEST_END);
  const uint64_t send_time_ns = timer.Duration(
      triton::client::RequestTimers::Kind::SEND_START,
      triton::client::RequestTimers::Kind::SEND_END);
  const uint64_t recv_time_ns = timer.Duration(
      triton::client::RequestTimers::Kind::RECV_START,
      triton::client::RequestTimers::Kind::RECV_END);

  if ((request_time_ns == std::numeric_limits<uint64_t>::max()) ||
      (send_time_ns == std::numeric_limits<uint64_t>::max()) ||
      (recv_time_ns == std::numeric_limits<uint64_t>::max())) {
    return Error(
        "Timer not set correctly." +
        ((timer.Timestamp(triton::client::RequestTimers::Kind::REQUEST_START) >
          timer.Timestamp(triton::client::RequestTimers::Kind::REQUEST_END))
             ? (" Request time from " +
                std::to_string(timer.Timestamp(
                    triton::client::RequestTimers::Kind::REQUEST_START)) +
                " to " +
                std::to_string(timer.Timestamp(
                    triton::client::RequestTimers::Kind::REQUEST_END)) +
                ".")
             : "") +
        ((timer.Timestamp(triton::client::RequestTimers::Kind::SEND_START) >
          timer.Timestamp(triton::client::RequestTimers::Kind::SEND_END))
             ? (" Send time from " +
                std::to_string(timer.Timestamp(
                    triton::client::RequestTimers::Kind::SEND_START)) +
                " to " +
                std::to_string(timer.Timestamp(
                    triton::client::RequestTimers::Kind::SEND_END)) +
                ".")
             : "") +
        ((timer.Timestamp(triton::client::RequestTimers::Kind::RECV_START) >
          timer.Timestamp(triton::client::RequestTimers::Kind::RECV_END))
             ? (" Receive time from " +
                std::to_string(timer.Timestamp(
                    triton::client::RequestTimers::Kind::RECV_START)) +
                " to " +
                std::to_string(timer.Timestamp(
                    triton::client::RequestTimers::Kind::RECV_END)) +
                ".")
             : ""));
  }

  infer_stat_.completed_request_count++;
  infer_stat_.cumulative_total_request_time_ns += request_time_ns;
  infer_stat_.cumulative_send_time_ns += send_time_ns;
  infer_stat_.cumulative_receive_time_ns += recv_time_ns;

  return Error::Success;
}

//==============================================================================

}}}}  // namespace triton::perfanalyzer::clientbackend::openai
