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

//! Error types for the Triton client library.
//!
//! This module defines [`Error`] -- the unified error type returned by all
//! fallible operations -- along with the [`Result`] type alias used throughout
//! the crate.

/// Convenience alias for `std::result::Result<T, Error>`.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors that may occur when communicating with a Triton Inference Server.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Failed to establish or maintain a gRPC connection.
    #[error("connection error: {0}")]
    Connection(String),

    /// The gRPC transport layer returned an error.
    #[error("transport error: {0}")]
    Transport(#[from] tonic::transport::Error),

    /// The server returned a gRPC status error.
    #[error("gRPC error (code={code}): {message}")]
    Grpc {
        /// The gRPC status code.
        code: tonic::Code,
        /// The error message from the server.
        message: String,
    },

    /// An inference input or request was constructed with invalid parameters.
    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// The requested model is not in a ready state on the server.
    #[error("model not ready: {model_name} (version: {model_version})")]
    ModelNotReady {
        /// Name of the model that was not ready.
        model_name: String,
        /// Version of the model that was not ready.
        model_version: String,
    },

    /// The server itself is not in a ready state.
    #[error("server not ready")]
    ServerNotReady,

    /// A streaming inference response contained an error message.
    #[error("stream inference error: {0}")]
    StreamInference(String),

    /// The server returned a response that could not be interpreted.
    #[error("unexpected response: {0}")]
    UnexpectedResponse(String),
}

impl From<tonic::Status> for Error {
    fn from(status: tonic::Status) -> Self {
        Self::Grpc {
            code: status.code(),
            message: status.message().to_owned(),
        }
    }
}
