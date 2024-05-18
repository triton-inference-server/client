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

#include <curl/curl.h>

#include <condition_variable>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <thread>

namespace triton { namespace perfanalyzer { namespace clientbackend {
namespace openai {

// The options for authorizing and authenticating SSL/TLS connections.
struct HttpSslOptions {
  enum CERTTYPE { CERT_PEM = 0, CERT_DER = 1 };
  enum KEYTYPE {
    KEY_PEM = 0,
    KEY_DER = 1
    // TODO TMA-1645: Support loading private key from crypto engine
    // KEY_ENG = 2
  };
  explicit HttpSslOptions()
      : verify_peer(1), verify_host(2), cert_type(CERTTYPE::CERT_PEM),
        key_type(KEYTYPE::KEY_PEM)
  {
  }
  // This option determines whether curl verifies the authenticity of the peer's
  // certificate. A value of 1 means curl verifies; 0 (zero) means it does not.
  // Default value is 1. See here for more details:
  // https://curl.se/libcurl/c/CURLOPT_SSL_VERIFYPEER.html
  long verify_peer;
  // This option determines whether libcurl verifies that the server cert is for
  // the server it is known as. The default value for this option is 2 which
  // means that certificate must indicate that the server is the server to which
  // you meant to connect, or the connection fails. See here for more details:
  // https://curl.se/libcurl/c/CURLOPT_SSL_VERIFYHOST.html
  long verify_host;
  // File holding one or more certificates to verify the peer with. If not
  // specified, client will look for the system path where cacert bundle is
  // assumed to be stored, as established at build time. See here for more
  // information: https://curl.se/libcurl/c/CURLOPT_CAINFO.html
  std::string ca_info;
  // The format of client certificate. By default it is CERT_PEM. See here for
  // more details: https://curl.se/libcurl/c/CURLOPT_SSLCERTTYPE.html
  CERTTYPE cert_type;
  // The file name of your client certificate. See here for more details:
  // https://curl.se/libcurl/c/CURLOPT_SSLCERT.html
  std::string cert;
  // The format of the private key. By default it is KEY_PEM. See here for more
  // details: https://curl.se/libcurl/c/CURLOPT_SSLKEYTYPE.html.
  KEYTYPE key_type;
  // The private key. See here for more details:
  // https://curl.se/libcurl/c/CURLOPT_SSLKEY.html.
  std::string key;
};

// HttpRequest object representing the context of a HTTP transaction. Currently
// it is also designed to be the placeholder for response data, but how the
// response is stored can be revisited later.
// 'completion_callback' doesn't transfer ownership of HttpRequest, caller must
// not keep the reference and access HttpRequest object after
// 'completion_callback' returns
class HttpRequest {
 public:
  HttpRequest(
      std::function<void(HttpRequest*)>&& completion_callback,
      const bool verbose = false);
  virtual ~HttpRequest();

  // Adds the input data to be delivered to the server, note that the HTTP
  // request does not own the buffer.
  void AddInput(uint8_t* buf, size_t byte_size);

  // Helper function for CURL
  // Copy into 'buf' up to 'size' bytes of input data. Return the
  // actual amount copied in 'input_bytes'.
  void GetNextInput(uint8_t* buf, size_t size, size_t* input_bytes);

  // Buffer that accumulates the response body.
  std::string response_buffer_;

  size_t total_input_byte_size_{0};

  // HTTP response code for the inference request
  uint32_t http_code_{200};

  std::function<void(HttpRequest*)> completion_callback_{nullptr};

  // Pointer to the list of the HTTP request header, keep it such that it will
  // be valid during the transfer and can be freed once transfer is completed.
  struct curl_slist* header_list_{nullptr};

 protected:
  const bool verbose_{false};

  // Pointers to the input data.
  std::deque<std::pair<uint8_t*, size_t>> data_buffers_;
};

// Base class for common HTTP functionalities
class HttpClient {
 public:
  enum class CompressionType { NONE, DEFLATE, GZIP };

  virtual ~HttpClient();

 protected:
  void SetSSLCurlOptions(CURL* curl_handle);

  HttpClient(
      const std::string& server_url, bool verbose = false,
      const HttpSslOptions& ssl_options = HttpSslOptions());

  // Note that this function does not block
  void Send(CURL* handle, std::unique_ptr<HttpRequest>&& request);

 protected:
  void AsyncTransfer();

  bool exiting_{false};

  std::thread worker_;
  std::mutex mutex_;
  std::condition_variable cv_;

  // The server url
  const std::string url_;
  // The options for authorizing and authenticating SSL/TLS connections
  HttpSslOptions ssl_options_;

  using AsyncReqMap = std::map<uintptr_t, std::unique_ptr<HttpRequest>>;
  // curl multi handle for processing asynchronous requests
  void* multi_handle_;
  // map to record ongoing asynchronous requests with pointer to easy handle
  // or tag id as key
  AsyncReqMap ongoing_async_requests_;

  bool verbose_;

 private:
  const std::string& ParseSslKeyType(HttpSslOptions::KEYTYPE key_type);
  const std::string& ParseSslCertType(HttpSslOptions::CERTTYPE cert_type);
  static std::mutex curl_init_mtx_;
};
}}}}  // namespace triton::perfanalyzer::clientbackend::openai
