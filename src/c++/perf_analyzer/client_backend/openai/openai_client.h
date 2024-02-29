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

/// \file

#include <map>
#include <memory>

namespace triton { namespace perfanalyzer { namespace clientbackend {

namespace openai {

class HttpInferRequest;

/// The key-value map type to be included in the request
/// as custom headers.
typedef std::map<std::string, std::string> Headers;
/// The key-value map type to be included as URL parameters.
typedef std::map<std::string, std::string> Parameters;

// The options for authorizing and authenticating SSL/TLS connections.
struct HttpSslOptions {
  enum CERTTYPE { CERT_PEM = 0, CERT_DER = 1 };
  enum KEYTYPE {
    KEY_PEM = 0,
    KEY_DER = 1
    // TODO: Support loading private key from crypto engine
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

class HttpRequest {
 public:
  HttpRequest(
      std::function<void(HttpRequest*)>&& response_callback,
      const bool verbose = false);
  ~HttpRequest();

  // Adds the input data to be delivered to the server, note that the HTTP
  // request does not own the buffer.
  void AddInput(uint8_t* buf, size_t byte_size);

 private:
  // Helper function for CURL
  // Copy into 'buf' up to 'size' bytes of input data. Return the
  // actual amount copied in 'input_bytes'.
  void GetNextInput(uint8_t* buf, size_t size, size_t* input_bytes);

  std::function<void(HttpRequest*)> response_callback_{nullptr};
  const bool verbose_{false};

  // Pointer to the list of the HTTP request header, keep it such that it will
  // be valid during the transfer and can be freed once transfer is completed.
  struct curl_slist* header_list_{nullptr};

  // HTTP response code for the inference request
  long http_code_{400};

  size_t total_input_byte_size_{0};

  // Buffer that accumulates the response body.
  std::string response_buffer_;

  // The pointers to the input data.
  std::deque<std::pair<uint8_t*, size_t>> data_buffers_;
};

// Base class for common HTTP functionalities
class HttpClient {
 public:
  enum class CompressionType { NONE, DEFLATE, GZIP };

  ~HttpClient();

 protected:
  void SetSSLCurlOptions(CURL** curl_handle);

  HttpClient(
      const std::string& server_url, bool verbose = false,
      const HttpSslOptions& ssl_options = HttpSslOptions());

 protected:
  // The server url
  const std::string url_;
  // The options for authorizing and authenticating SSL/TLS connections
  HttpSslOptions ssl_options_;

  using AsyncReqMap = std::map<uintptr_t, std::shared_ptr<HttpInferRequest>>;
  // curl multi handle for processing asynchronous requests
  void* multi_handle_;
  // map to record ongoing asynchronous requests with pointer to easy handle
  // or tag id as key
  AsyncReqMap ongoing_async_requests_;

  bool verbose_;

 private:
  // [FIXME] should belong to SSL option struct as helper function
  const std::string& ParseSslKeyType(HttpSslOptions::KEYTYPE key_type);
  const std::string& ParseSslCertType(HttpSslOptions::CERTTYPE cert_type);
};

//==============================================================================
/// An InferenceServerHttpClient object is used to perform any kind of
/// communication with the InferenceServer using HTTP protocol. None
/// of the methods of InferenceServerHttpClient are thread safe. The
/// class is intended to be used by a single thread and simultaneously
/// calling different methods with different threads is not supported
/// and will cause undefined behavior.
///
/// \code
///   std::unique_ptr<InferenceServerHttpClient> client;
///   InferenceServerHttpClient::Create(&client, "localhost:8000");
///   bool live;
///   client->IsServerLive(&live);
///   ...
///   ...
/// \endcode
///
class ChatCompletionClient : public HttpClient {
 public:
  class Result {};

  ~ChatCompletionClient();

  /// Create a client that can be used to communicate with the server.
  /// \param client Returns a new InferenceServerHttpClient object.
  /// \param server_url The inference server name, port, optional
  /// scheme and optional base path in the following format:
  /// <scheme://>host:port/<base-path>.
  /// \param verbose If true generate verbose output when contacting
  /// the inference server.
  /// \param ssl_options Specifies the settings for configuring
  /// SSL encryption and authorization. Providing these options
  /// do not ensure that SSL/TLS will be used in communication.
  /// The use of SSL/TLS depends entirely on the server endpoint.
  /// These options will be ignored if the server_url does not
  /// expose `https://` scheme.
  /// \return Error object indicating success or failure.
  ChatCompletionClient(
      const std::string& server_url, bool verbose = false,
      const HttpSslOptions& ssl_options = HttpSslOptions());

  /// Simplified AsyncInfer() where the request body is expected to be
  /// prepared by the caller, the client here is responsible to communicate
  /// with a OpenAI-compatible server in both streaming and non-streaming case.
  std::future AsyncInfer(const std::string& serialized_request_body);

  /// [TODO] AsyncInfer() variant that prepare the request body from function
  /// arguments.
  /// Run asynchronous inference on server.
  /// Once the request is completed, the InferResult pointer will be passed to
  /// the provided 'callback' function. Upon the invocation of callback
  /// function, the ownership of InferResult object is transferred to the
  /// function caller. It is then the caller's choice on either retrieving the
  /// results inside the callback function or deferring it to a different thread
  /// so that the client is unblocked. In order to prevent memory leak, user
  /// must ensure this object gets deleted.
  /// Note: InferInput::AppendRaw() or InferInput::SetSharedMemory() calls do
  /// not copy the data buffers but hold the pointers to the data directly.
  /// It is advisable to not to disturb the buffer contents until the respective
  /// callback is invoked.
  /// \param callback The callback function to be invoked on request completion.
  /// \param options The options for inference request.
  /// \param inputs The vector of InferInput describing the model inputs.
  /// \param outputs The vector of InferRequestedOutput describing how the
  /// output must be returned. If not provided then all the outputs in the
  /// model config will be returned as default settings.
  /// \param headers Optional map specifying additional HTTP headers to include
  /// in request.
  /// \param query_params Optional map specifying parameters that must be
  /// included with URL query.
  /// \param request_compression_algorithm Optional HTTP compression algorithm
  /// to use for the request body on client side. Currently supports DEFLATE,
  /// GZIP and NONE. By default, no compression is used.
  /// \param response_compression_algorithm Optional HTTP compression algorithm
  /// to request for the response body. Note that the response may not be
  /// compressed if the server does not support the specified algorithm.
  /// Currently supports DEFLATE, GZIP and NONE. By default, no compression
  /// is used.
  /// \return Error object indicating success
  /// or failure of the request.
  // Error AsyncInfer(
  //     OnCompleteFn callback, const InferOptions& options,
  //     const std::vector<InferInput*>& inputs,
  //     const std::vector<const InferRequestedOutput*>& outputs =
  //         std::vector<const InferRequestedOutput*>(),
  //     const Headers& headers = Headers(),
  //     const Parameters& query_params = Parameters(),
  //     const CompressionType request_compression_algorithm =
  //         CompressionType::NONE,
  //     const CompressionType response_compression_algorithm =
  //         CompressionType::NONE);

 private:
  Error PreRunProcessing(
      void* curl, std::string& request_uri, const InferOptions& options,
      const std::vector<InferInput*>& inputs,
      const std::vector<const InferRequestedOutput*>& outputs,
      const Headers& headers, std::shared_ptr<HttpInferRequest>& request);
  void AsyncTransfer();

  static size_t ResponseHandler(
      void* contents, size_t size, size_t nmemb, void* userp);
  static size_t InferRequestProvider(
      void* contents, size_t size, size_t nmemb, void* userp);
  static size_t InferResponseHeaderHandler(
      void* contents, size_t size, size_t nmemb, void* userp);
  static size_t InferResponseHandler(
      void* contents, size_t size, size_t nmemb, void* userp);
};

}}}}  // namespace triton::perfanalyzer::clientbackend::openai
