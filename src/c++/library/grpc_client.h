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

#include <grpcpp/grpcpp.h>
#include <queue>
#include "common.h"
#include "grpc_service.grpc.pb.h"
#include "ipc.h"
#include "model_config.pb.h"

namespace triton { namespace client {

/// The key-value map type to be included in the request
/// metadata
typedef std::map<std::string, std::string> Headers;

struct SslOptions {
  explicit SslOptions() {}
  // File containing the PEM encoding of the server root certificates.
  // If this parameter is empty, the default roots will be used. The
  // default roots can be overridden using the
  // GRPC_DEFAULT_SSL_ROOTS_FILE_PATH environment variable pointing
  // to a file on the file system containing the roots.
  std::string root_certificates;
  // File containing the PEM encoding of the client's private key.
  // This parameter can be empty if the client does not have a
  // private key.
  std::string private_key;
  // File containing the PEM encoding of the client's certificate chain.
  // This parameter can be empty if the client does not have a
  // certificate chain.
  std::string certificate_chain;
};

// GRPC KeepAlive: https://grpc.github.io/grpc/cpp/md_doc_keepalive.html
struct KeepAliveOptions {
  explicit KeepAliveOptions()
      : keepalive_time_ms(INT_MAX), keepalive_timeout_ms(20000),
        keepalive_permit_without_calls(false), http2_max_pings_without_data(2)
  {
  }
  // The period (in milliseconds) after which a keepalive ping is sent on the
  // transport
  int keepalive_time_ms;
  // The amount of time (in milliseconds) the sender of the keepalive ping waits
  // for an acknowledgement. If it does not receive an acknowledgment within
  // this time, it will close the connection.
  int keepalive_timeout_ms;
  // If true, allow keepalive pings to be sent even if there are no calls in
  // flight.
  bool keepalive_permit_without_calls;
  // The maximum number of pings that can be sent when there is no data/header
  // frame to be sent. gRPC Core will not continue sending pings if we run over
  // the limit. Setting it to 0 allows sending pings without such a restriction.
  int http2_max_pings_without_data;
};

//==============================================================================
/// An InferenceServerGrpcClient object is used to perform any kind of
/// communication with the InferenceServer using gRPC protocol.  Most
/// of the methods are thread-safe except Infer, AsyncInfer, StartStream
/// StopStream and AsyncStreamInfer. Calling these functions from different
/// threads will cause undefined behavior.
///
/// \code
///   std::unique_ptr<InferenceServerGrpcClient> client;
///   InferenceServerGrpcClient::Create(&client, "localhost:8001");
///   bool live;
///   client->IsServerLive(&live);
///   ...
///   ...
/// \endcode
///
class InferenceServerGrpcClient : public InferenceServerClient {
 public:
  ~InferenceServerGrpcClient();

  /// Create a client that can be used to communicate with the server.
  /// This is the expected method for most users to create a GRPC client with
  /// the options directly exposed Triton.
  /// \param client Returns a new InferenceServerGrpcClient object.
  /// \param server_url The inference server name and port.
  /// \param verbose If true generate verbose output when contacting
  /// the inference server.
  /// \param use_ssl If true use encrypted channel to the server.
  /// \param ssl_options Specifies the files required for
  /// SSL encryption and authorization.
  /// \param keepalive_options Specifies the GRPC KeepAlive options described
  /// in https://grpc.github.io/grpc/cpp/md_doc_keepalive.html
  /// \param use_cached_channel If false, a new channel is created for each
  /// new client instance. When true, re-use old channels from cache for new
  /// client instances. The default value is true.
  /// \return Error object indicating success or failure.
  static Error Create(
      std::unique_ptr<InferenceServerGrpcClient>* client,
      const std::string& server_url, bool verbose = false, bool use_ssl = false,
      const SslOptions& ssl_options = SslOptions(),
      const KeepAliveOptions& keepalive_options = KeepAliveOptions(),
      const bool use_cached_channel = true);

  /// Create a client that can be used to communicate with the server.
  /// This method is available for advanced users who need to specify custom
  /// grpc::ChannelArguments not exposed by Triton, at their own risk.
  /// \param client Returns a new InferenceServerGrpcClient object.
  /// \param channel_args Exposes user-defined grpc::ChannelArguments to
  /// be set for the client. Triton assumes that the "channel_args" passed
  /// to this method are correct and complete, and are set at the user's
  /// own risk. For example, GRPC KeepAlive options may be specified directly
  /// in this argument rather than passing a KeepAliveOptions object.
  /// \param server_url The inference server name and port.
  /// \param verbose If true generate verbose output when contacting
  /// the inference server.
  /// \param use_ssl If true use encrypted channel to the server.
  /// \param ssl_options Specifies the files required for
  /// SSL encryption and authorization.
  /// \param use_cached_channel If false, a new channel is created for each
  /// new client instance. When true, re-use old channels from cache for new
  /// client instances. The default value is true.
  /// \return Error object indicating success or failure.
  static Error Create(
      std::unique_ptr<InferenceServerGrpcClient>* client,
      const std::string& server_url, const grpc::ChannelArguments& channel_args,
      bool verbose = false, bool use_ssl = false,
      const SslOptions& ssl_options = SslOptions(),
      const bool use_cached_channel = true);

  /// Contact the inference server and get its liveness.
  /// \param live Returns whether the server is live or not.
  /// \param headers Optional map specifying additional HTTP headers to include
  /// in the metadata of gRPC request.
  /// \return Error object indicating success or failure of the request.
  Error IsServerLive(bool* live, const Headers& headers = Headers());

  /// Contact the inference server and get its readiness.
  /// \param ready Returns whether the server is ready or not.
  /// \param headers Optional map specifying additional HTTP headers to include
  /// in the metadata of gRPC request.
  /// \return Error object indicating success or failure of the request.
  Error IsServerReady(bool* ready, const Headers& headers = Headers());

  /// Contact the inference server and get the readiness of specified model.
  /// \param ready Returns whether the specified model is ready or not.
  /// \param model_name The name of the model to check for readiness.
  /// \param model_version The version of the model to check for readiness.
  /// The default value is an empty string which means then the server will
  /// choose a version based on the model and internal policy.
  /// \param headers Optional map specifying additional HTTP headers to include
  /// in the metadata of gRPC request.
  /// \return Error object indicating success or failure of the request.
  Error IsModelReady(
      bool* ready, const std::string& model_name,
      const std::string& model_version = "",
      const Headers& headers = Headers());

  /// Contact the inference server and get its metadata.
  /// \param server_metadata Returns the server metadata as
  /// SeverMetadataResponse message.
  /// \param headers Optional map specifying additional HTTP headers to include
  /// in the metadata of gRPC request.
  /// \return Error object indicating success or failure of the request.
  Error ServerMetadata(
      inference::ServerMetadataResponse* server_metadata,
      const Headers& headers = Headers());

  /// Contact the inference server and get the metadata of specified model.
  /// \param model_metadata Returns model metadata as ModelMetadataResponse
  /// message.
  /// \param model_name The name of the model to get metadata.
  /// \param model_version The version of the model to get metadata.
  /// The default value is an empty string which means then the server will
  /// choose a version based on the model and internal policy.
  /// \param headers Optional map specifying additional HTTP headers to include
  /// in the metadata of gRPC request.
  /// \return Error object indicating success or failure of the request.
  Error ModelMetadata(
      inference::ModelMetadataResponse* model_metadata,
      const std::string& model_name, const std::string& model_version = "",
      const Headers& headers = Headers());

  /// Contact the inference server and get the configuration of specified model.
  /// \param model_config Returns model config as ModelConfigResponse
  /// message.
  /// \param model_name The name of the model to get configuration.
  /// \param model_version The version of the model to get configuration.
  /// The default value is an empty string which means then the server will
  /// choose a version based on the model and internal policy.
  /// \param headers Optional map specifying additional HTTP headers to include
  /// in the metadata of gRPC request.
  /// \return Error object indicating success or failure of the request.
  Error ModelConfig(
      inference::ModelConfigResponse* model_config,
      const std::string& model_name, const std::string& model_version = "",
      const Headers& headers = Headers());

  /// Contact the inference server and get the index of model repository
  /// contents.
  /// \param repository_index Returns the repository index as
  /// RepositoryIndexRequestResponse
  /// \param headers Optional map specifying additional HTTP headers to include
  /// in the metadata of gRPC request.
  /// \return Error object indicating success or failure of the request.
  Error ModelRepositoryIndex(
      inference::RepositoryIndexResponse* repository_index,
      const Headers& headers = Headers());

  /// Request the inference server to load or reload specified model.
  /// \param model_name The name of the model to be loaded or reloaded.
  /// \param headers Optional map specifying additional HTTP headers to include
  /// in the metadata of gRPC request.
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
      const std::string& config = std::string(),
      const std::map<std::string, std::vector<char>>& files = {});

  /// Request the inference server to unload specified model.
  /// \param model_name The name of the model to be unloaded.
  /// \param headers Optional map specifying additional HTTP headers to include
  /// in the metadata of gRPC request.
  /// \return Error object indicating success or failure of the request.
  Error UnloadModel(
      const std::string& model_name, const Headers& headers = Headers());

  /// Contact the inference server and get the inference statistics for the
  /// specified model name and version.
  /// \param infer_stat The inference statistics of requested model name and
  /// version.
  /// \param model_name The name of the model to get inference statistics. The
  /// default value is an empty string which means statistics of all models will
  /// be returned in the response.
  /// \param model_version The version of the model to get inference statistics.
  /// The default value is an empty string which means then the server will
  /// choose a version based on the model and internal policy.
  /// \param headers Optional map specifying additional HTTP headers to include
  /// in the metadata of gRPC request.
  /// \return Error object indicating success or failure of the request.
  Error ModelInferenceStatistics(
      inference::ModelStatisticsResponse* infer_stat,
      const std::string& model_name = "", const std::string& model_version = "",
      const Headers& headers = Headers());

  /// Update the trace settings for the specified model name, or global trace
  /// settings if model name is not given.
  /// \param response The updated settings as TraceSettingResponse.
  /// \param model_name The name of the model to update trace settings. The
  /// default value is an empty string which means the global trace settings
  /// will be updated.
  /// \param settings The new trace setting values. Only the settings listed
  /// will be updated. If a trace setting is listed in the map with an empty
  /// string, that setting will be cleared.
  /// \param config Optional JSON representation of a model config provided for
  /// the load request, if provided, this config will be used for
  /// loading the model.
  /// \param headers Optional map specifying additional HTTP headers to include
  /// in the metadata of gRPC request.
  /// \return Error object indicating success or failure of the request.
  Error UpdateTraceSettings(
      inference::TraceSettingResponse* response,
      const std::string& model_name = "",
      const std::map<std::string, std::vector<std::string>>& settings =
          std::map<std::string, std::vector<std::string>>(),
      const Headers& headers = Headers());

  /// Get the trace settings for the specified model name, or global trace
  /// settings if model name is not given.
  /// \param settings The trace settings as TraceSettingResponse.
  /// \param model_name The name of the model to get trace settings. The
  /// default value is an empty string which means the global trace settings
  /// will be returned.
  /// \param headers Optional map specifying additional HTTP headers to include
  /// in the metadata of gRPC request.
  /// \return Error object indicating success or failure of the request.
  Error GetTraceSettings(
      inference::TraceSettingResponse* settings,
      const std::string& model_name = "", const Headers& headers = Headers());

  /// Contact the inference server and get the status for requested system
  /// shared memory.
  /// \param status The system shared memory status as
  /// SystemSharedMemoryStatusResponse
  /// \param region_name The name of the region to query status. The default
  /// value is an empty string, which means that the status of all active system
  /// shared memory will be returned.
  /// \param headers Optional map specifying additional HTTP headers to include
  /// in the metadata of gRPC request.
  /// \return Error object indicating success or failure of the request.
  Error SystemSharedMemoryStatus(
      inference::SystemSharedMemoryStatusResponse* status,
      const std::string& region_name = "", const Headers& headers = Headers());

  /// Request the server to register a system shared memory with the provided
  /// details.
  /// \param name The name of the region to register.
  /// \param key The key of the underlying memory object that contains the
  /// system shared memory region.
  /// \param byte_size The size of the system shared memory region, in bytes.
  /// \param offset Offset, in bytes, within the underlying memory object to
  /// the start of the system shared memory region. The default value is zero.
  /// \param headers Optional map specifying additional HTTP headers to include
  /// in the metadata of gRPC request.
  /// \return Error object indicating success or failure of the request
  Error RegisterSystemSharedMemory(
      const std::string& name, const std::string& key, const size_t byte_size,
      const size_t offset = 0, const Headers& headers = Headers());

  /// Request the server to unregister a system shared memory with the
  /// specified name.
  /// \param name The name of the region to unregister. The default value is
  /// empty string which means all the system shared memory regions will be
  /// unregistered.
  /// \param headers Optional map specifying additional HTTP headers to include
  /// in the metadata of gRPC request.
  /// \return Error object indicating success or failure of the request
  Error UnregisterSystemSharedMemory(
      const std::string& name = "", const Headers& headers = Headers());

  /// Contact the inference server and get the status for requested CUDA
  /// shared memory.
  /// \param status The CUDA shared memory status as
  /// CudaSharedMemoryStatusResponse
  /// \param region_name The name of the region to query status. The default
  /// value is an empty string, which means that the status of all active CUDA
  /// shared memory will be returned.
  /// \param headers Optional map specifying additional HTTP headers to include
  /// in the metadata of gRPC request.
  /// \return Error object indicating success or failure of the request.
  Error CudaSharedMemoryStatus(
      inference::CudaSharedMemoryStatusResponse* status,
      const std::string& region_name = "", const Headers& headers = Headers());

  /// Request the server to register a CUDA shared memory with the provided
  /// details.
  /// \param name The name of the region to register.
  /// \param cuda_shm_handle The cudaIPC handle for the memory object.
  /// \param device_id The GPU device ID on which the cudaIPC handle was
  /// created.
  /// \param byte_size The size of the CUDA shared memory region, in
  /// bytes.
  /// \param headers Optional map specifying additional HTTP headers to
  /// include in the metadata of gRPC request.
  /// \return Error object indicating success or failure of the request
  Error RegisterCudaSharedMemory(
      const std::string& name, const cudaIpcMemHandle_t& cuda_shm_handle,
      const size_t device_id, const size_t byte_size,
      const Headers& headers = Headers());

  /// Request the server to unregister a CUDA shared memory with the
  /// specified name.
  /// \param name The name of the region to unregister. The default value is
  /// empty string which means all the CUDA shared memory regions will be
  /// unregistered.
  /// \param headers Optional map specifying additional HTTP headers to
  /// include in the metadata of gRPC request.
  /// \return Error object indicating success or failure of the request
  Error UnregisterCudaSharedMemory(
      const std::string& name = "", const Headers& headers = Headers());

  /// Run synchronous inference on server.
  /// \param result Returns the result of inference.
  /// \param options The options for inference request.
  /// \param inputs The vector of InferInput describing the model inputs.
  /// \param outputs Optional vector of InferRequestedOutput describing how the
  /// output must be returned. If not provided then all the outputs in the model
  /// config will be returned as default settings.
  /// \param headers Optional map specifying additional HTTP headers to include
  /// in the metadata of gRPC request.
  /// \param compression_algorithm The compression algorithm to be used
  /// by gRPC when sending requests. By default compression is not used.
  /// \return Error object indicating success or failure of the
  /// request.
  Error Infer(
      InferResult** result, const InferOptions& options,
      const std::vector<InferInput*>& inputs,
      const std::vector<const InferRequestedOutput*>& outputs =
          std::vector<const InferRequestedOutput*>(),
      const Headers& headers = Headers(),
      grpc_compression_algorithm compression_algorithm = GRPC_COMPRESS_NONE);

  /// Run asynchronous inference on server.
  /// Once the request is completed, the InferResult pointer will be passed to
  /// the provided 'callback' function. Upon the invocation of callback
  /// function, the ownership of InferResult object is transfered to the
  /// function caller. It is then the caller's choice on either retrieving the
  /// results inside the callback function or deferring it to a different thread
  /// so that the client is unblocked. In order to prevent memory leak, user
  /// must ensure this object gets deleted.
  /// \param callback The callback function to be invoked on request completion.
  /// \param options The options for inference request.
  /// \param inputs The vector of InferInput describing the model inputs.
  /// \param outputs Optional vector of InferRequestedOutput describing how the
  /// output must be returned. If not provided then all the outputs in the model
  /// config will be returned as default settings.
  /// \param headers Optional map specifying additional HTTP headers to include
  /// in the metadata of gRPC request.
  /// \param compression_algorithm The compression algorithm to be used
  /// by gRPC when sending requests. By default compression is not used.
  /// \return Error object indicating success or failure of the request.
  Error AsyncInfer(
      OnCompleteFn callback, const InferOptions& options,
      const std::vector<InferInput*>& inputs,
      const std::vector<const InferRequestedOutput*>& outputs =
          std::vector<const InferRequestedOutput*>(),
      const Headers& headers = Headers(),
      grpc_compression_algorithm compression_algorithm = GRPC_COMPRESS_NONE);

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
  /// in the metadata of gRPC request.
  /// \param compression_algorithm The compression algorithm to be used
  /// by gRPC when sending requests. By default compression is not used.
  /// \return Error object indicating success or failure of the
  /// request.
  Error InferMulti(
      std::vector<InferResult*>* results,
      const std::vector<InferOptions>& options,
      const std::vector<std::vector<InferInput*>>& inputs,
      const std::vector<std::vector<const InferRequestedOutput*>>& outputs =
          std::vector<std::vector<const InferRequestedOutput*>>(),
      const Headers& headers = Headers(),
      grpc_compression_algorithm compression_algorithm = GRPC_COMPRESS_NONE);

  /// Run multiple asynchronous inferences on server.
  /// Once all the requests are completed, the vector of InferResult pointers
  /// will be passed to the provided 'callback' function. Upon the invocation
  /// of callback function, the ownership of the InferResult objects are
  /// transfered to the function caller. It is then the caller's choice on
  /// either retrieving the results inside the callback function or deferring it
  /// to a different thread so that the client is unblocked. In order to
  /// prevent memory leak, user must ensure these objects get deleted.
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
  /// in the metadata of gRPC request.
  /// \param compression_algorithm The compression algorithm to be used
  /// by gRPC when sending requests. By default compression is not used.
  /// \return Error object indicating success or failure of the request.
  Error AsyncInferMulti(
      OnMultiCompleteFn callback, const std::vector<InferOptions>& options,
      const std::vector<std::vector<InferInput*>>& inputs,
      const std::vector<std::vector<const InferRequestedOutput*>>& outputs =
          std::vector<std::vector<const InferRequestedOutput*>>(),
      const Headers& headers = Headers(),
      grpc_compression_algorithm compression_algorithm = GRPC_COMPRESS_NONE);

  /// Starts a grpc bi-directional stream to send streaming inferences.
  /// \param callback The callback function to be invoked on receiving a
  /// response at the stream.
  /// \param enable_stats Indicates whether client library should record the
  /// the client-side statistics for inference requests on stream or not.
  /// The library does not support client side statistics for decoupled
  /// streaming. Set this option false when there is no 1:1 mapping between
  /// request and response on the stream.
  /// \param stream_timeout Specifies the end-to-end timeout for the streaming
  /// connection in microseconds. The default value is 0 which means that
  /// there is no limitation on deadline. The stream will be closed once
  /// the specified time elapses.
  /// \param headers Optional map specifying additional HTTP headers to
  /// include in the metadata of gRPC request.
  /// \param compression_algorithm The compression algorithm to be used
  /// by gRPC when sending requests. By default compression is not used.
  /// \return Error object indicating success or failure of the request.
  Error StartStream(
      OnCompleteFn callback, bool enable_stats = true,
      uint32_t stream_timeout = 0, const Headers& headers = Headers(),
      grpc_compression_algorithm compression_algorithm = GRPC_COMPRESS_NONE);

  /// Stops an active grpc bi-directional stream, if one available.
  /// \return Error object indicating success or failure of the request.
  Error StopStream();

  /// Runs an asynchronous inference over gRPC bi-directional streaming
  /// API. A stream must be established with a call to StartStream()
  /// before calling this function. All the results will be provided to the
  /// callback function provided when starting the stream.
  /// \param options The options for inference request.
  /// \param inputs The vector of InferInput describing the model inputs.
  /// \param outputs Optional vector of InferRequestedOutput describing how the
  /// output must be returned. If not provided then all the outputs in the model
  /// config will be returned as default settings.
  /// \return Error object indicating success or failure of the request.
  Error AsyncStreamInfer(
      const InferOptions& options, const std::vector<InferInput*>& inputs,
      const std::vector<const InferRequestedOutput*>& outputs =
          std::vector<const InferRequestedOutput*>());

 private:
  InferenceServerGrpcClient(
      const std::string& url, bool verbose, bool use_ssl,
      const SslOptions& ssl_options, const grpc::ChannelArguments& channel_args,
      const bool use_cached_channel);

  Error PreRunProcessing(
      const InferOptions& options, const std::vector<InferInput*>& inputs,
      const std::vector<const InferRequestedOutput*>& outputs);
  void AsyncTransfer();
  void AsyncStreamTransfer();

  // The producer-consumer queue used to communicate asynchronously with
  // the GRPC runtime.
  grpc::CompletionQueue async_request_completion_queue_;

  // Required to support the grpc bi-directional streaming API.
  InferenceServerClient::OnCompleteFn stream_callback_;
  std::thread stream_worker_;
  std::shared_ptr<grpc::ClientReaderWriter<
      inference::ModelInferRequest, inference::ModelStreamInferResponse>>
      grpc_stream_;
  grpc::ClientContext grpc_context_;

  bool enable_stream_stats_;
  std::queue<std::unique_ptr<RequestTimers>> ongoing_stream_request_timers_;
  std::mutex stream_mutex_;

  // GRPC end point.
  std::shared_ptr<inference::GRPCInferenceService::Stub> stub_;
  // request for GRPC call, one request object can be used for multiple calls
  // since it can be overwritten as soon as the GRPC send finishes.
  inference::ModelInferRequest infer_request_;
};


}}  // namespace triton::client
