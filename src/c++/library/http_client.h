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

/// \file

#include <map>
#include <memory>
#include "common.h"
#include "ipc.h"

namespace triton { namespace client {

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
class InferenceServerHttpClient : public InferenceServerClient {
 public:
  enum class CompressionType { NONE, DEFLATE, GZIP };
  ~InferenceServerHttpClient();

  /// Generate a request body for inference using the supplied 'inputs' and
  /// requesting the outputs specified by 'outputs'.
  /// \param request_body Returns the generated inference request body
  /// \param header_length Returns the length of the inference header.
  /// \param options The options for inference request.
  /// \param inputs The vector of InferInput describing the model inputs.
  /// \param outputs The vector of InferRequestedOutput describing how the
  /// output must be returned. If not provided then all the outputs in the
  /// model config will be returned as default settings.
  /// \return Error object indicating success or failure of the
  /// request.
  static Error GenerateRequestBody(
      std::vector<char>* request_body, size_t* header_length,
      const InferOptions& options, const std::vector<InferInput*>& inputs,
      const std::vector<const InferRequestedOutput*>& outputs =
          std::vector<const InferRequestedOutput*>());

  /// Generate a InferResult object from the given 'response_body'.
  /// \param result Returns the generated InferResult object.
  /// \param response_body The inference response from the server
  /// \param header_length The length of the inference header if the header
  /// does not occupy the whole response body. 0 indicates that
  /// the whole response body is the inference response header.
  /// \return Error object indicating success or failure of the
  /// request.
  static Error ParseResponseBody(
      InferResult** result, const std::vector<char>& response_body,
      const size_t header_length = 0);

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
  static Error Create(
      std::unique_ptr<InferenceServerHttpClient>* client,
      const std::string& server_url, bool verbose = false,
      const HttpSslOptions& ssl_options = HttpSslOptions());

  /// Contact the inference server and get its liveness.
  /// \param live Returns whether the server is live or not.
  /// \param headers Optional map specifying additional HTTP headers to include
  /// in request.
  /// \return Error object indicating success or failure of the request.
  Error IsServerLive(
      bool* live, const Headers& headers = Headers(),
      const Parameters& query_params = Parameters());

  /// Contact the inference server and get its readiness.
  /// \param ready Returns whether the server is ready or not.
  /// \param headers Optional map specifying additional HTTP headers to include
  /// in request.
  /// \param query_params Optional map specifying parameters that must be
  /// included with URL query.
  /// \return Error object indicating success or failure of the request.
  Error IsServerReady(
      bool* ready, const Headers& headers = Headers(),
      const Parameters& query_params = Parameters());

  /// Contact the inference server and get the readiness of specified model.
  /// \param ready Returns whether the specified model is ready or not.
  /// \param model_name The name of the model to check for readiness.
  /// \param model_version The version of the model to check for readiness.
  /// The default value is an empty string which means then the server will
  /// choose a version based on the model and internal policy.
  /// \param headers Optional map specifying additional HTTP headers to include
  /// in request.
  /// \param query_params Optional map specifying parameters that must be
  /// included with URL query.
  /// \return Error object indicating success or failure of the request.
  Error IsModelReady(
      bool* ready, const std::string& model_name,
      const std::string& model_version = "", const Headers& headers = Headers(),
      const Parameters& query_params = Parameters());

  /// Contact the inference server and get its metadata.
  /// \param server_metadata Returns JSON representation of the
  /// metadata as a string.
  /// \param headers Optional map specifying additional HTTP headers to
  /// include in request.
  /// \param query_params Optional map specifying parameters that must be
  /// included with URL query.
  /// \return Error object indicating success or failure of the request.
  Error ServerMetadata(
      std::string* server_metadata, const Headers& headers = Headers(),
      const Parameters& query_params = Parameters());

  /// Contact the inference server and get the metadata of specified model.
  /// \param model_metadata Returns JSON representation of model
  /// metadata as a string.
  /// \param model_name The name of the model to get metadata.
  /// \param model_version The version of the model to get metadata.
  /// The default value is an empty string which means then the server will
  /// choose a version based on the model and internal policy.
  /// \param headers Optional map specifying additional HTTP headers to include
  /// in request.
  /// \param query_params Optional map specifying parameters that must be
  /// included with URL query.
  /// \return Error object indicating success or failure of the request.
  Error ModelMetadata(
      std::string* model_metadata, const std::string& model_name,
      const std::string& model_version = "", const Headers& headers = Headers(),
      const Parameters& query_params = Parameters());

  /// Contact the inference server and get the configuration of specified model.
  /// \param model_config Returns JSON representation of model
  /// configuration as a string.
  /// \param model_name The name of the model to get configuration.
  /// \param model_version The version of the model to get configuration.
  /// The default value is an empty string which means then the server will
  /// choose a version based on the model and internal policy.
  /// \param headers Optional map specifying additional HTTP headers to include
  /// in request.
  /// \param query_params Optional map specifying parameters that must be
  /// included with URL query.
  /// \return Error object indicating success or failure of the request.
  Error ModelConfig(
      std::string* model_config, const std::string& model_name,
      const std::string& model_version = "", const Headers& headers = Headers(),
      const Parameters& query_params = Parameters());

  /// Contact the inference server and get the index of model repository
  /// contents.
  /// \param repository_index Returns JSON representation of the
  /// repository index as a string.
  /// \param headers Optional map specifying additional HTTP headers to include
  /// in request.
  /// \param query_params Optional map specifying parameters that must be
  /// included with URL query.
  /// \return Error object indicating success or failure of the request.
  Error ModelRepositoryIndex(
      std::string* repository_index, const Headers& headers = Headers(),
      const Parameters& query_params = Parameters());

  /// Request the inference server to load or reload specified model.
  /// \param model_name The name of the model to be loaded or reloaded.
  /// \param headers Optional map specifying additional HTTP headers to include
  /// in request.
  /// \param query_params Optional map specifying parameters that must be
  /// included with URL query.
  /// \param config Optional JSON representation of a model config provided for
  /// the load request, if provided, this config will be used for
  /// loading the model.
  /// \param files Optional map specifying file path (with "file:"
  /// prefix) in the override model directory to the file content.
  /// The files will form the model directory that the model
  /// will be loaded from. If specified, 'config' must be provided to be
  /// the model configuration of the override model directory.
  /// \return Error object indicating success or failure of the request.
  Error LoadModel(
      const std::string& model_name, const Headers& headers = Headers(),
      const Parameters& query_params = Parameters(),
      const std::string& config = std::string(),
      const std::map<std::string, std::vector<char>>& files = {});

  /// Request the inference server to unload specified model.
  /// \param model_name The name of the model to be unloaded.
  /// \param headers Optional map specifying additional HTTP headers to include
  /// in request.
  /// \param query_params Optional map specifying parameters that must be
  /// included with URL query.
  /// \return Error object indicating success or failure of the request.
  Error UnloadModel(
      const std::string& model_name, const Headers& headers = Headers(),
      const Parameters& query_params = Parameters());

  /// Contact the inference server and get the inference statistics for the
  /// specified model name and version.
  /// \param infer_stat Returns the JSON representation of the
  /// inference statistics as a string.
  /// \param model_name The name of the model to get inference statistics. The
  /// default value is an empty string which means statistics of all models will
  /// be returned in the response.
  /// \param model_version The version of the model to get inference statistics.
  /// The default value is an empty string which means then the server will
  /// choose a version based on the model and internal policy.
  /// \param headers Optional map specifying additional HTTP headers to include
  /// in request.
  /// \param query_params Optional map specifying parameters that must be
  /// included with URL query.
  /// \return Error object indicating success or failure of the request.
  Error ModelInferenceStatistics(
      std::string* infer_stat, const std::string& model_name = "",
      const std::string& model_version = "", const Headers& headers = Headers(),
      const Parameters& query_params = Parameters());

  /// Update the trace settings for the specified model name, or global trace
  /// settings if model name is not given.
  /// \param response Returns the JSON representation of the updated trace
  /// settings as a string.
  /// \param model_name The name of the model to update trace settings. The
  /// default value is an empty string which means the global trace settings
  /// will be updated.
  /// \param settings The new trace setting values. Only the settings listed
  /// will be updated. If a trace setting is listed in the map with an empty
  /// string, that setting will be cleared.
  /// \param headers Optional map specifying additional HTTP headers to include
  /// in request.
  /// \param query_params Optional map specifying parameters that must be
  /// included with URL query.
  /// \return Error object indicating success or failure of the request.
  Error UpdateTraceSettings(
      std::string* response, const std::string& model_name = "",
      const std::map<std::string, std::vector<std::string>>& settings =
          std::map<std::string, std::vector<std::string>>(),
      const Headers& headers = Headers(),
      const Parameters& query_params = Parameters());

  /// Get the trace settings for the specified model name, or global trace
  /// settings if model name is not given.
  /// \param settings Returns the JSON representation of the trace
  /// settings as a string.
  /// \param model_name The name of the model to get trace settings. The
  /// default value is an empty string which means the global trace settings
  /// will be returned.
  /// \param headers Optional map specifying additional HTTP headers to include
  /// in request.
  /// \param query_params Optional map specifying parameters that must be
  /// included with URL query.
  /// \return Error object indicating success or failure of the request.
  Error GetTraceSettings(
      std::string* settings, const std::string& model_name = "",
      const Headers& headers = Headers(),
      const Parameters& query_params = Parameters());

  /// Contact the inference server and get the status for requested system
  /// shared memory.
  /// \param status Returns the JSON representation of the system
  /// shared memory status as a string.
  /// \param region_name The name of the region to query status. The default
  /// value is an empty string, which means that the status of all active system
  /// shared memory will be returned.
  /// \param headers Optional map specifying additional HTTP headers to include
  /// in request.
  /// \param query_params Optional map specifying parameters that must be
  /// included with URL query.
  /// \return Error object indicating success or failure of the request.
  Error SystemSharedMemoryStatus(
      std::string* status, const std::string& region_name = "",
      const Headers& headers = Headers(),
      const Parameters& query_params = Parameters());

  /// Request the server to register a system shared memory with the provided
  /// details.
  /// \param name The name of the region to register.
  /// \param key The key of the underlying memory object that contains the
  /// system shared memory region.
  /// \param byte_size The size of the system shared memory region, in bytes.
  /// \param offset Offset, in bytes, within the underlying memory object to
  /// the start of the system shared memory region. The default value is zero.
  /// \param headers Optional map specifying additional HTTP headers to include
  /// in request.
  /// \param query_params Optional map specifying parameters that must be
  /// included with URL query.
  /// \return Error object indicating success or failure of the request
  Error RegisterSystemSharedMemory(
      const std::string& name, const std::string& key, const size_t byte_size,
      const size_t offset = 0, const Headers& headers = Headers(),
      const Parameters& query_params = Parameters());

  /// Request the server to unregister a system shared memory with the
  /// specified name.
  /// \param name The name of the region to unregister. The default value is
  /// empty string which means all the system shared memory regions will be
  /// unregistered.
  /// \param headers Optional map specifying additional HTTP headers to include
  /// in request.
  /// \param query_params Optional map specifying parameters that must be
  /// included with URL query.
  /// \return Error object indicating success or failure of the request
  Error UnregisterSystemSharedMemory(
      const std::string& name = "", const Headers& headers = Headers(),
      const Parameters& query_params = Parameters());

  /// Contact the inference server and get the status for requested CUDA
  /// shared memory.
  /// \param status Returns the JSON representation of the CUDA shared
  /// memory status as a string.
  /// \param region_name The name of the region to query status. The default
  /// value is an empty string, which means that the status of all active CUDA
  /// shared memory will be returned.
  /// \param headers Optional map specifying additional HTTP headers to include
  /// in request.
  /// \param query_params Optional map specifying parameters that must be
  /// included with URL query.
  /// \return Error object indicating success or failure of the request.
  Error CudaSharedMemoryStatus(
      std::string* status, const std::string& region_name = "",
      const Headers& headers = Headers(),
      const Parameters& query_params = Parameters());

  /// Request the server to register a CUDA shared memory with the provided
  /// details.
  /// \param name The name of the region to register.
  /// \param cuda_shm_handle The cudaIPC handle for the memory object.
  /// \param device_id The GPU device ID on which the cudaIPC handle was
  /// created.
  /// \param byte_size The size of the CUDA shared memory region, in
  /// bytes.
  /// \param headers Optional map specifying additional HTTP headers to include
  /// in request.
  /// \param query_params Optional map specifying parameters that must be
  /// included with URL query.
  /// \return Error object indicating success or failure of the request
  Error RegisterCudaSharedMemory(
      const std::string& name, const cudaIpcMemHandle_t& cuda_shm_handle,
      const size_t device_id, const size_t byte_size,
      const Headers& headers = Headers(),
      const Parameters& query_params = Parameters());

  /// Request the server to unregister a CUDA shared memory with the
  /// specified name.
  /// \param name The name of the region to unregister. The default value is
  /// empty string which means all the CUDA shared memory regions will be
  /// unregistered.
  /// \param headers Optional map specifying additional HTTP headers to include
  /// in request.
  /// \param query_params Optional map specifying parameters that must be
  /// included with URL query.
  /// \return Error object indicating success or failure of the request
  Error UnregisterCudaSharedMemory(
      const std::string& name = "", const Headers& headers = Headers(),
      const Parameters& query_params = Parameters());

  /// Run synchronous inference on server.
  /// \param result Returns the result of inference.
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
  /// \return Error object indicating success or failure of the
  /// request.
  Error Infer(
      InferResult** result, const InferOptions& options,
      const std::vector<InferInput*>& inputs,
      const std::vector<const InferRequestedOutput*>& outputs =
          std::vector<const InferRequestedOutput*>(),
      const Headers& headers = Headers(),
      const Parameters& query_params = Parameters(),
      const CompressionType request_compression_algorithm =
          CompressionType::NONE,
      const CompressionType response_compression_algorithm =
          CompressionType::NONE);

  /// Run asynchronous inference on server.
  /// Once the request is completed, the InferResult pointer will be passed to
  /// the provided 'callback' function. Upon the invocation of callback
  /// function, the ownership of InferResult object is transfered to the
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
  Error AsyncInfer(
      OnCompleteFn callback, const InferOptions& options,
      const std::vector<InferInput*>& inputs,
      const std::vector<const InferRequestedOutput*>& outputs =
          std::vector<const InferRequestedOutput*>(),
      const Headers& headers = Headers(),
      const Parameters& query_params = Parameters(),
      const CompressionType request_compression_algorithm =
          CompressionType::NONE,
      const CompressionType response_compression_algorithm =
          CompressionType::NONE);

  /// Run multiple synchronous inferences on server.
  /// \param results Returns the results of the inferences.
  /// \param options The options for each inference request, one set of
  /// options may be provided and it will be used for all inference requests.
  /// \param inputs The vector of InferInput objects describing the model inputs
  /// for each inference request.
  /// \param outputs Optional vector of InferRequestedOutput objects describing
  /// how the output must be returned. If not provided then all the outputs in
  /// the model config will be returned as default settings. And one set of
  /// outputs may be provided and it will be used for all inference requests.
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
  /// \return Error object indicating success or failure of the
  /// request.
  Error InferMulti(
      std::vector<InferResult*>* results,
      const std::vector<InferOptions>& options,
      const std::vector<std::vector<InferInput*>>& inputs,
      const std::vector<std::vector<const InferRequestedOutput*>>& outputs =
          std::vector<std::vector<const InferRequestedOutput*>>(),
      const Headers& headers = Headers(),
      const Parameters& query_params = Parameters(),
      const CompressionType request_compression_algorithm =
          CompressionType::NONE,
      const CompressionType response_compression_algorithm =
          CompressionType::NONE);

  /// Run multiple asynchronous inferences on server.
  /// Once all the requests are completed, the vector of InferResult pointers
  /// will be passed to the provided 'callback' function. Upon the invocation
  /// of callback function, the ownership of the InferResult objects are
  /// transfered to the function caller. It is then the caller's choice on
  /// either retrieving the results inside the callback function or deferring it
  /// to a different thread so that the client is unblocked. In order to
  /// prevent memory leak, user must ensure these objects get deleted.
  /// Note: InferInput::AppendRaw() or InferInput::SetSharedMemory() calls do
  /// not copy the data buffers but hold the pointers to the data directly.
  /// It is advisable to not to disturb the buffer contents until the respective
  /// callback is invoked.
  /// \param callback The callback function to be invoked on the completion of
  /// all requests.
  /// \param options The options for each inference request, one set of
  /// option may be provided and it will be used for all inference requests.
  /// \param inputs The vector of InferInput objects describing the model inputs
  /// for each inference request.
  /// \param outputs Optional vector of InferRequestedOutput objects describing
  /// how the output must be returned. If not provided then all the outputs in
  /// the model config will be returned as default settings. And one set of
  /// outputs may be provided and it will be used for all inference requests.
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
  Error AsyncInferMulti(
      OnMultiCompleteFn callback, const std::vector<InferOptions>& options,
      const std::vector<std::vector<InferInput*>>& inputs,
      const std::vector<std::vector<const InferRequestedOutput*>>& outputs =
          std::vector<std::vector<const InferRequestedOutput*>>(),
      const Headers& headers = Headers(),
      const Parameters& query_params = Parameters(),
      const CompressionType request_compression_algorithm =
          CompressionType::NONE,
      const CompressionType response_compression_algorithm =
          CompressionType::NONE);

 private:
  InferenceServerHttpClient(
      const std::string& url, bool verbose, const HttpSslOptions& ssl_options);

  Error PreRunProcessing(
      void* curl, std::string& request_uri, const InferOptions& options,
      const std::vector<InferInput*>& inputs,
      const std::vector<const InferRequestedOutput*>& outputs,
      const Headers& headers, const Parameters& query_params,
      const CompressionType request_compression_algorithm,
      const CompressionType response_compression_algorithm,
      std::shared_ptr<HttpInferRequest>& request);
  void AsyncTransfer();
  Error Get(
      std::string& request_uri, const Headers& headers,
      const Parameters& query_params, std::string* response,
      long* http_code = nullptr);
  Error Post(
      std::string& request_uri, const std::string& request,
      const Headers& headers, const Parameters& query_params,
      std::string* response);

  static size_t ResponseHandler(
      void* contents, size_t size, size_t nmemb, void* userp);
  static size_t InferRequestProvider(
      void* contents, size_t size, size_t nmemb, void* userp);
  static size_t InferResponseHeaderHandler(
      void* contents, size_t size, size_t nmemb, void* userp);
  static size_t InferResponseHandler(
      void* contents, size_t size, size_t nmemb, void* userp);

  // The server url
  const std::string url_;
  // The options for authorizing and authenticating SSL/TLS connections
  HttpSslOptions ssl_options_;

  using AsyncReqMap = std::map<uintptr_t, std::shared_ptr<HttpInferRequest>>;
  // curl easy handle shared for all synchronous requests
  void* easy_handle_;
  // curl multi handle for processing asynchronous requests
  void* multi_handle_;
  // map to record ongoing asynchronous requests with pointer to easy handle
  // or tag id as key
  AsyncReqMap ongoing_async_requests_;
};

}}  // namespace triton::client
