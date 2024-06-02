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

#include "http_client.h"

#include <functional>
#include <iostream>

namespace triton { namespace perfanalyzer { namespace clientbackend {
namespace openai {

HttpRequest::HttpRequest(
    std::function<void(HttpRequest*)>&& completion_callback, const bool verbose)
    : completion_callback_(std::move(completion_callback)), verbose_(verbose)
{
}

HttpRequest::~HttpRequest()
{
  if (header_list_ != nullptr) {
    curl_slist_free_all(header_list_);
    header_list_ = nullptr;
  }
}

void
HttpRequest::AddInput(uint8_t* buf, size_t byte_size)
{
  data_buffers_.push_back(std::pair<uint8_t*, size_t>(buf, byte_size));
  total_input_byte_size_ += byte_size;
}

void
HttpRequest::GetNextInput(uint8_t* buf, size_t size, size_t* input_bytes)
{
  *input_bytes = 0;

  while (!data_buffers_.empty() && size > 0) {
    const size_t csz = std::min(data_buffers_.front().second, size);
    if (csz > 0) {
      const uint8_t* input_ptr = data_buffers_.front().first;
      std::copy(input_ptr, input_ptr + csz, buf);
      size -= csz;
      buf += csz;
      *input_bytes += csz;

      data_buffers_.front().first += csz;
      data_buffers_.front().second -= csz;
    }
    if (data_buffers_.front().second == 0) {
      data_buffers_.pop_front();
    }
  }
}

std::mutex HttpClient::curl_init_mtx_{};
HttpClient::HttpClient(
    const std::string& server_url, bool verbose,
    const HttpSslOptions& ssl_options)
    : url_(server_url), verbose_(verbose), ssl_options_(ssl_options)
{
  // [TODO TMA-1670] uncomment below and remove class-wise mutex once confirm
  // curl >= 7.84.0 will always be used
  // auto* ver = curl_version_info(CURLVERSION_NOW);
  // if (ver->features & CURL_VERSION_THREADSAFE == 0) {
  //   throw std::runtime_error(
  //       "HTTP client has dependency on CURL library to have thread-safe "
  //       "support (CURL_VERSION_THREADSAFE set)");
  // }
  {
    std::lock_guard<std::mutex> lk(curl_init_mtx_);
    if (curl_global_init(CURL_GLOBAL_ALL) != 0) {
      throw std::runtime_error("CURL global initialization failed");
    }
  }

  multi_handle_ = curl_multi_init();

  worker_ = std::thread(&HttpClient::AsyncTransfer, this);
}

HttpClient::~HttpClient()
{
  {
    std::lock_guard<std::mutex> lock(mutex_);
    exiting_ = true;
  }

  curl_multi_wakeup(multi_handle_);

  // thread not joinable if AsyncInfer() is not called
  // (it is default constructed thread before the first AsyncInfer() call)
  if (worker_.joinable()) {
    worker_.join();
  }

  curl_multi_cleanup(multi_handle_);

  {
    std::lock_guard<std::mutex> lk(curl_init_mtx_);
    curl_global_cleanup();
  }
}

const std::string&
HttpClient::ParseSslCertType(HttpSslOptions::CERTTYPE cert_type)
{
  static std::string pem_str{"PEM"};
  static std::string der_str{"DER"};
  switch (cert_type) {
    case HttpSslOptions::CERTTYPE::CERT_PEM:
      return pem_str;
    case HttpSslOptions::CERTTYPE::CERT_DER:
      return der_str;
  }
  throw std::runtime_error(
      "Unexpected SSL certificate type encountered. Only PEM and DER are "
      "supported.");
}

const std::string&
HttpClient::ParseSslKeyType(HttpSslOptions::KEYTYPE key_type)
{
  static std::string pem_str{"PEM"};
  static std::string der_str{"DER"};
  switch (key_type) {
    case HttpSslOptions::KEYTYPE::KEY_PEM:
      return pem_str;
    case HttpSslOptions::KEYTYPE::KEY_DER:
      return der_str;
  }
  throw std::runtime_error(
      "unsupported SSL key type encountered. Only PEM and DER are "
      "supported.");
}

void
HttpClient::SetSSLCurlOptions(CURL* curl_handle)
{
  curl_easy_setopt(
      curl_handle, CURLOPT_SSL_VERIFYPEER, ssl_options_.verify_peer);
  curl_easy_setopt(
      curl_handle, CURLOPT_SSL_VERIFYHOST, ssl_options_.verify_host);
  if (!ssl_options_.ca_info.empty()) {
    curl_easy_setopt(curl_handle, CURLOPT_CAINFO, ssl_options_.ca_info.c_str());
  }
  const auto& curl_cert_type = ParseSslCertType(ssl_options_.cert_type);
  curl_easy_setopt(curl_handle, CURLOPT_SSLCERTTYPE, curl_cert_type.c_str());
  if (!ssl_options_.cert.empty()) {
    curl_easy_setopt(curl_handle, CURLOPT_SSLCERT, ssl_options_.cert.c_str());
  }
  const auto& curl_key_type = ParseSslKeyType(ssl_options_.key_type);
  curl_easy_setopt(curl_handle, CURLOPT_SSLKEYTYPE, curl_key_type.c_str());
  if (!ssl_options_.key.empty()) {
    curl_easy_setopt(curl_handle, CURLOPT_SSLKEY, ssl_options_.key.c_str());
  }
}

void
HttpClient::Send(CURL* handle, std::unique_ptr<HttpRequest>&& request)
{
  {
    std::lock_guard<std::mutex> lock(mutex_);

    if (exiting_) {
      return;
    }

    auto insert_result = new_async_requests_.emplace(std::make_pair(
        reinterpret_cast<uintptr_t>(handle), std::move(request)));
    if (!insert_result.second) {
      curl_easy_cleanup(handle);
      throw std::runtime_error(
          "Failed to insert new asynchronous request context.");
    }
  }
  curl_multi_wakeup(multi_handle_);
}

void
HttpClient::AsyncTransfer()
{
  int messages_in_queue = 0;
  int still_running = 0;
  int numfds = 0;
  CURLMsg* msg = nullptr;
  AsyncReqMap ongoing_async_requests;

  do {
    {
      // Check for new requests and add them to ongoing requests

      std::lock_guard<std::mutex> lock(mutex_);

      for (auto& pair : new_async_requests_) {
        curl_multi_add_handle(
            multi_handle_, reinterpret_cast<CURL*>(pair.first));

        ongoing_async_requests[pair.first] = std::move(pair.second);
      }
      new_async_requests_.clear();
    }

    CURLMcode mc = curl_multi_perform(multi_handle_, &still_running);

    if (mc != CURLM_OK) {
      std::cerr << "Unexpected error: curl_multi failed. Code:" << mc
                << std::endl;
      continue;
    }

    while ((msg = curl_multi_info_read(multi_handle_, &messages_in_queue))) {
      if (msg->msg != CURLMSG_DONE) {
        // Something wrong happened.
        std::cerr << "Unexpected error: received CURLMsg=" << msg->msg
                  << std::endl;
        continue;
      }

      uintptr_t identifier = reinterpret_cast<uintptr_t>(msg->easy_handle);
      auto itr = ongoing_async_requests.find(identifier);
      // This shouldn't happen
      if (itr == ongoing_async_requests.end()) {
        std::cerr << "Unexpected error: received completed request that is not "
                     "in the list of asynchronous requests"
                  << std::endl;
        curl_multi_remove_handle(multi_handle_, msg->easy_handle);
        curl_easy_cleanup(msg->easy_handle);
        continue;
      }

      uint32_t http_code = 400;
      if (msg->data.result == CURLE_OK) {
        curl_easy_getinfo(msg->easy_handle, CURLINFO_RESPONSE_CODE, &http_code);
      } else if (msg->data.result == CURLE_OPERATION_TIMEDOUT) {
        http_code = 499;
      }

      itr->second->http_code_ = http_code;
      itr->second->completion_callback_(itr->second.get());
      ongoing_async_requests.erase(itr);
      curl_multi_remove_handle(multi_handle_, msg->easy_handle);
      curl_easy_cleanup(msg->easy_handle);
    }


    // Wait for activity on existing requests or
    // explicit curl_multi_wakeup call
    //
    // If there are no descriptors in the multi_handle_
    // then curl_multi_poll will wait until curl_multi_wakeup
    // is called
    //
    // curl_multi_wakeup is called when adding a new request
    // or exiting

    mc = curl_multi_poll(multi_handle_, NULL, 0, INT_MAX, &numfds);

    if (mc != CURLM_OK) {
      std::cerr << "Unexpected error: curl_multi failed. Code:" << mc
                << std::endl;
    }

  } while (!exiting_);

  for (auto& request : ongoing_async_requests) {
    CURL* easy_handle = reinterpret_cast<CURL*>(request.first);
    curl_multi_remove_handle(multi_handle_, easy_handle);
    curl_easy_cleanup(easy_handle);
  }

  for (auto& request : new_async_requests_) {
    CURL* easy_handle = reinterpret_cast<CURL*>(request.first);
    curl_easy_cleanup(easy_handle);
  }
}

}}}}  // namespace triton::perfanalyzer::clientbackend::openai
