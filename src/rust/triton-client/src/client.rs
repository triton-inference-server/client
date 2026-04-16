// Copyright 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

//! The main Triton client implementation.
//!
//! [`TritonClient`] provides an ergonomic async API for communicating with an
//! NVIDIA Triton Inference Server over gRPC. It wraps the auto-generated
//! gRPC stubs and exposes high-level methods for health checks, metadata
//! queries, inference, model management, and more.
//!
//! # Example
//!
//! ```rust,no_run
//! # async fn example() -> triton_client::error::Result<()> {
//! use triton_client::client::TritonClient;
//! use triton_client::infer::{DataType, InferInput, InferRequestBuilder};
//!
//! let client = TritonClient::connect("http://localhost:8001").await?;
//!
//! // Check server health
//! let live = client.is_server_live().await?;
//! let ready = client.is_server_ready().await?;
//!
//! // Run inference
//! let input = InferInput::new("input0", vec![1, 16], DataType::Fp32)
//!     .with_data_f32(&[0.0; 16]);
//!
//! let request = InferRequestBuilder::new("my_model")
//!     .input(input)
//!     .output("output0")
//!     .build();
//!
//! let response = client.infer(request).await?;
//! let values = response.output_as_f32(0)?;
//! # Ok(())
//! # }
//! ```

use std::collections::HashMap;
use std::time::Duration;

use tokio_stream::{Stream, StreamExt};
use tonic::transport::{Channel, Endpoint};

use crate::error::{Error, Result};
use crate::generated::inference::{
    self, grpc_inference_service_client::GrpcInferenceServiceClient,
};
use crate::infer::{InferResponse, ModelIndex, ModelMetadata, ServerMetadata, TensorMetadata};

/// Default maximum message size for gRPC (128 MiB).
const DEFAULT_MAX_MESSAGE_SIZE: usize = 128 * 1024 * 1024;

/// Options for configuring the Triton client connection.
///
/// # Example
///
/// ```rust
/// use std::time::Duration;
/// use triton_client::client::ClientOptions;
///
/// let options = ClientOptions::default()
///     .connect_timeout(Duration::from_secs(10))
///     .request_timeout(Duration::from_secs(30))
///     .max_message_size(256 * 1024 * 1024);
/// ```
#[derive(Debug, Clone)]
pub struct ClientOptions {
    connect_timeout: Option<Duration>,
    request_timeout: Option<Duration>,
    max_message_size: usize,
    keep_alive_interval: Option<Duration>,
    keep_alive_timeout: Option<Duration>,
}

impl Default for ClientOptions {
    fn default() -> Self {
        Self {
            connect_timeout: Some(Duration::from_secs(5)),
            request_timeout: None,
            max_message_size: DEFAULT_MAX_MESSAGE_SIZE,
            keep_alive_interval: None,
            keep_alive_timeout: None,
        }
    }
}

impl ClientOptions {
    /// Sets the timeout for establishing the initial connection.
    #[must_use]
    pub fn connect_timeout(self, timeout: Duration) -> Self {
        Self {
            connect_timeout: Some(timeout),
            ..self
        }
    }

    /// Sets the timeout applied to each individual RPC request.
    #[must_use]
    pub fn request_timeout(self, timeout: Duration) -> Self {
        Self {
            request_timeout: Some(timeout),
            ..self
        }
    }

    /// Sets the maximum gRPC message size in bytes.
    ///
    /// Default: 128 MiB.
    #[must_use]
    pub fn max_message_size(self, size: usize) -> Self {
        Self {
            max_message_size: size,
            ..self
        }
    }

    /// Sets the HTTP/2 keep-alive interval.
    #[must_use]
    pub fn keep_alive_interval(self, interval: Duration) -> Self {
        Self {
            keep_alive_interval: Some(interval),
            ..self
        }
    }

    /// Sets the HTTP/2 keep-alive timeout.
    #[must_use]
    pub fn keep_alive_timeout(self, timeout: Duration) -> Self {
        Self {
            keep_alive_timeout: Some(timeout),
            ..self
        }
    }
}

/// A client for communicating with NVIDIA Triton Inference Server via gRPC.
///
/// The client is cheaply cloneable -- clones share the same underlying gRPC
/// channel and can be used concurrently from multiple tasks.
///
/// # Example
///
/// ```rust,no_run
/// # async fn example() -> triton_client::error::Result<()> {
/// use triton_client::client::TritonClient;
///
/// let client = TritonClient::connect("http://localhost:8001").await?;
/// let metadata = client.server_metadata().await?;
/// println!("Server: {} v{}", metadata.name, metadata.version);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct TritonClient {
    inner: GrpcInferenceServiceClient<Channel>,
}

impl TritonClient {
    /// Connects to a Triton Inference Server at the given URL with default
    /// options.
    ///
    /// # Arguments
    ///
    /// * `url` -- The gRPC endpoint URL (e.g. `"http://localhost:8001"`).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Transport`] if the connection cannot be established.
    pub async fn connect(url: &str) -> Result<Self> {
        Self::connect_with_options(url, ClientOptions::default()).await
    }

    /// Connects to a Triton Inference Server with custom options.
    ///
    /// # Arguments
    ///
    /// * `url` -- The gRPC endpoint URL (e.g. `"http://localhost:8001"`).
    /// * `options` -- Connection and transport configuration.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Transport`] if the connection cannot be established.
    pub async fn connect_with_options(url: &str, options: ClientOptions) -> Result<Self> {
        let mut endpoint = Endpoint::from_shared(url.to_owned())
            .map_err(|e| Error::Connection(format!("invalid URL: {e}")))?;

        if let Some(timeout) = options.connect_timeout {
            endpoint = endpoint.connect_timeout(timeout);
        }
        if let Some(timeout) = options.request_timeout {
            endpoint = endpoint.timeout(timeout);
        }
        if let Some(interval) = options.keep_alive_interval {
            endpoint = endpoint.keep_alive_while_idle(true);
            endpoint = endpoint.http2_keep_alive_interval(interval);
        }
        if let Some(timeout) = options.keep_alive_timeout {
            endpoint = endpoint.keep_alive_timeout(timeout);
        }

        let channel = endpoint.connect().await?;

        let inner = GrpcInferenceServiceClient::new(channel)
            .max_decoding_message_size(options.max_message_size)
            .max_encoding_message_size(options.max_message_size);

        Ok(Self { inner })
    }

    // -----------------------------------------------------------------------
    // Health checks
    // -----------------------------------------------------------------------

    /// Checks whether the server is live (i.e. the process is running).
    ///
    /// # Errors
    ///
    /// Returns an error if the gRPC call fails.
    pub async fn is_server_live(&self) -> Result<bool> {
        let response = self
            .inner
            .clone()
            .server_live(inference::ServerLiveRequest {})
            .await?;
        Ok(response.into_inner().live)
    }

    /// Checks whether the server is ready to accept inference requests.
    ///
    /// # Errors
    ///
    /// Returns an error if the gRPC call fails.
    pub async fn is_server_ready(&self) -> Result<bool> {
        let response = self
            .inner
            .clone()
            .server_ready(inference::ServerReadyRequest {})
            .await?;
        Ok(response.into_inner().ready)
    }

    /// Checks whether a specific model (and optionally version) is ready.
    ///
    /// # Arguments
    ///
    /// * `model_name` -- The name of the model.
    /// * `model_version` -- The version to check. Pass `""` for the default
    ///   version.
    ///
    /// # Errors
    ///
    /// Returns an error if the gRPC call fails.
    pub async fn is_model_ready(&self, model_name: &str, model_version: &str) -> Result<bool> {
        let response = self
            .inner
            .clone()
            .model_ready(inference::ModelReadyRequest {
                name: model_name.to_owned(),
                version: model_version.to_owned(),
            })
            .await?;
        Ok(response.into_inner().ready)
    }

    // -----------------------------------------------------------------------
    // Metadata
    // -----------------------------------------------------------------------

    /// Retrieves server metadata including name, version, and supported
    /// extensions.
    ///
    /// # Errors
    ///
    /// Returns an error if the gRPC call fails.
    pub async fn server_metadata(&self) -> Result<ServerMetadata> {
        let response = self
            .inner
            .clone()
            .server_metadata(inference::ServerMetadataRequest {})
            .await?;
        let md = response.into_inner();
        Ok(ServerMetadata {
            name: md.name,
            version: md.version,
            extensions: md.extensions,
        })
    }

    /// Retrieves metadata for a specific model.
    ///
    /// # Arguments
    ///
    /// * `model_name` -- The name of the model.
    /// * `model_version` -- The version to query. Pass `""` for the default
    ///   version.
    ///
    /// # Errors
    ///
    /// Returns an error if the gRPC call fails.
    pub async fn model_metadata(
        &self,
        model_name: &str,
        model_version: &str,
    ) -> Result<ModelMetadata> {
        let response = self
            .inner
            .clone()
            .model_metadata(inference::ModelMetadataRequest {
                name: model_name.to_owned(),
                version: model_version.to_owned(),
            })
            .await?;
        let md = response.into_inner();
        Ok(ModelMetadata {
            name: md.name,
            versions: md.versions,
            platform: md.platform,
            inputs: md
                .inputs
                .into_iter()
                .map(|t| TensorMetadata {
                    name: t.name,
                    datatype: t.datatype,
                    shape: t.shape,
                })
                .collect(),
            outputs: md
                .outputs
                .into_iter()
                .map(|t| TensorMetadata {
                    name: t.name,
                    datatype: t.datatype,
                    shape: t.shape,
                })
                .collect(),
        })
    }

    /// Retrieves the configuration for a specific model.
    ///
    /// Returns the raw protobuf [`inference::ModelConfig`] from the server.
    ///
    /// # Arguments
    ///
    /// * `model_name` -- The name of the model.
    /// * `model_version` -- The version to query. Pass `""` for the default
    ///   version.
    ///
    /// # Errors
    ///
    /// Returns an error if the gRPC call fails or the response has no config.
    pub async fn model_config(
        &self,
        model_name: &str,
        model_version: &str,
    ) -> Result<inference::ModelConfig> {
        let response = self
            .inner
            .clone()
            .model_config(inference::ModelConfigRequest {
                name: model_name.to_owned(),
                version: model_version.to_owned(),
            })
            .await?;
        response
            .into_inner()
            .config
            .ok_or_else(|| Error::UnexpectedResponse("model config response has no config".into()))
    }

    // -----------------------------------------------------------------------
    // Inference
    // -----------------------------------------------------------------------

    /// Performs a single inference request.
    ///
    /// Use [`InferRequestBuilder`](crate::infer::InferRequestBuilder) to
    /// construct the request.
    ///
    /// # Errors
    ///
    /// Returns an error if the gRPC call fails.
    pub async fn infer(&self, request: inference::ModelInferRequest) -> Result<InferResponse> {
        let response = self.inner.clone().model_infer(request).await?;
        Ok(InferResponse::new(response.into_inner()))
    }

    /// Performs streaming inference.
    ///
    /// Sends a stream of inference requests and returns a stream of responses.
    /// This is useful for sequence models or when the server produces multiple
    /// response chunks.
    ///
    /// # Errors
    ///
    /// Returns an error if the gRPC stream cannot be established. Individual
    /// stream items may also contain errors.
    pub async fn infer_stream(
        &self,
        requests: impl Stream<Item = inference::ModelInferRequest> + Send + 'static,
    ) -> Result<impl Stream<Item = Result<InferResponse>>> {
        let response = self.inner.clone().model_stream_infer(requests).await?;

        let stream = response.into_inner().map(|result| match result {
            Ok(stream_response) => {
                if !stream_response.error_message.is_empty() {
                    return Err(Error::StreamInference(stream_response.error_message));
                }
                match stream_response.infer_response {
                    Some(infer_response) => Ok(InferResponse::new(infer_response)),
                    None => Err(Error::UnexpectedResponse(
                        "stream response has no infer_response".into(),
                    )),
                }
            }
            Err(status) => Err(Error::from(status)),
        });

        Ok(stream)
    }

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    /// Retrieves cumulative inference statistics for a model.
    ///
    /// # Arguments
    ///
    /// * `model_name` -- The name of the model. Pass `""` for all models.
    /// * `model_version` -- The version to query. Pass `""` for all versions.
    ///
    /// # Errors
    ///
    /// Returns an error if the gRPC call fails.
    pub async fn model_statistics(
        &self,
        model_name: &str,
        model_version: &str,
    ) -> Result<inference::ModelStatisticsResponse> {
        let response = self
            .inner
            .clone()
            .model_statistics(inference::ModelStatisticsRequest {
                name: model_name.to_owned(),
                version: model_version.to_owned(),
            })
            .await?;
        Ok(response.into_inner())
    }

    // -----------------------------------------------------------------------
    // Model repository management
    // -----------------------------------------------------------------------

    /// Returns the index of available models in the repository.
    ///
    /// # Errors
    ///
    /// Returns an error if the gRPC call fails.
    pub async fn repository_index(&self) -> Result<Vec<ModelIndex>> {
        let response = self
            .inner
            .clone()
            .repository_index(inference::RepositoryIndexRequest {
                repository_name: String::new(),
                ready: false,
            })
            .await?;
        Ok(response
            .into_inner()
            .models
            .into_iter()
            .map(|m| ModelIndex {
                name: m.name,
                version: m.version,
                state: m.state,
                reason: m.reason,
            })
            .collect())
    }

    /// Loads or reloads a model from the model repository.
    ///
    /// # Errors
    ///
    /// Returns an error if the gRPC call fails (e.g. model not found).
    pub async fn load_model(&self, model_name: &str) -> Result<()> {
        self.inner
            .clone()
            .repository_model_load(inference::RepositoryModelLoadRequest {
                repository_name: String::new(),
                model_name: model_name.to_owned(),
                parameters: Default::default(),
            })
            .await?;
        Ok(())
    }

    /// Unloads a model, freeing its resources on the server.
    ///
    /// # Errors
    ///
    /// Returns an error if the gRPC call fails.
    pub async fn unload_model(&self, model_name: &str) -> Result<()> {
        self.inner
            .clone()
            .repository_model_unload(inference::RepositoryModelUnloadRequest {
                repository_name: String::new(),
                model_name: model_name.to_owned(),
                parameters: Default::default(),
            })
            .await?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Shared memory (system)
    // -----------------------------------------------------------------------

    /// Gets the status of registered system shared memory regions.
    ///
    /// # Arguments
    ///
    /// * `name` -- The region name. Pass `""` for all regions.
    ///
    /// # Errors
    ///
    /// Returns an error if the gRPC call fails.
    pub async fn system_shared_memory_status(
        &self,
        name: &str,
    ) -> Result<inference::SystemSharedMemoryStatusResponse> {
        let response = self
            .inner
            .clone()
            .system_shared_memory_status(inference::SystemSharedMemoryStatusRequest {
                name: name.to_owned(),
            })
            .await?;
        Ok(response.into_inner())
    }

    /// Registers a system shared memory region.
    ///
    /// # Errors
    ///
    /// Returns an error if the gRPC call fails.
    pub async fn system_shared_memory_register(
        &self,
        name: &str,
        key: &str,
        offset: u64,
        byte_size: u64,
    ) -> Result<()> {
        self.inner
            .clone()
            .system_shared_memory_register(inference::SystemSharedMemoryRegisterRequest {
                name: name.to_owned(),
                key: key.to_owned(),
                offset,
                byte_size,
            })
            .await?;
        Ok(())
    }

    /// Unregisters a system shared memory region.
    ///
    /// # Arguments
    ///
    /// * `name` -- The region name. Pass `""` to unregister all regions.
    ///
    /// # Errors
    ///
    /// Returns an error if the gRPC call fails.
    pub async fn system_shared_memory_unregister(&self, name: &str) -> Result<()> {
        self.inner
            .clone()
            .system_shared_memory_unregister(inference::SystemSharedMemoryUnregisterRequest {
                name: name.to_owned(),
            })
            .await?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Shared memory (CUDA)
    // -----------------------------------------------------------------------

    /// Gets the status of registered CUDA shared memory regions.
    ///
    /// # Arguments
    ///
    /// * `name` -- The region name. Pass `""` for all regions.
    ///
    /// # Errors
    ///
    /// Returns an error if the gRPC call fails.
    pub async fn cuda_shared_memory_status(
        &self,
        name: &str,
    ) -> Result<inference::CudaSharedMemoryStatusResponse> {
        let response = self
            .inner
            .clone()
            .cuda_shared_memory_status(inference::CudaSharedMemoryStatusRequest {
                name: name.to_owned(),
            })
            .await?;
        Ok(response.into_inner())
    }

    /// Unregisters a CUDA shared memory region.
    ///
    /// # Arguments
    ///
    /// * `name` -- The region name. Pass `""` to unregister all regions.
    ///
    /// # Errors
    ///
    /// Returns an error if the gRPC call fails.
    pub async fn cuda_shared_memory_unregister(&self, name: &str) -> Result<()> {
        self.inner
            .clone()
            .cuda_shared_memory_unregister(inference::CudaSharedMemoryUnregisterRequest {
                name: name.to_owned(),
            })
            .await?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Trace settings
    // -----------------------------------------------------------------------

    /// Gets or updates the server trace settings.
    ///
    /// Pass `None` for `model_name` to get/set global trace settings.
    ///
    /// # Errors
    ///
    /// Returns an error if the gRPC call fails.
    pub async fn trace_setting(
        &self,
        model_name: Option<&str>,
        settings: HashMap<String, inference::trace_setting_request::SettingValue>,
    ) -> Result<inference::TraceSettingResponse> {
        let response = self
            .inner
            .clone()
            .trace_setting(inference::TraceSettingRequest {
                model_name: model_name.unwrap_or_default().to_owned(),
                settings,
            })
            .await?;
        Ok(response.into_inner())
    }

    // -----------------------------------------------------------------------
    // Log settings
    // -----------------------------------------------------------------------

    /// Gets or updates the server log settings.
    ///
    /// # Errors
    ///
    /// Returns an error if the gRPC call fails.
    pub async fn log_settings(
        &self,
        settings: HashMap<String, inference::log_settings_request::SettingValue>,
    ) -> Result<inference::LogSettingsResponse> {
        let response = self
            .inner
            .clone()
            .log_settings(inference::LogSettingsRequest { settings })
            .await?;
        Ok(response.into_inner())
    }
}
