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

//! Rust client library for NVIDIA Triton Inference Server.
//!
//! This crate provides a type-safe, async Rust API for communicating with
//! [Triton Inference Server](https://github.com/triton-inference-server/server)
//! over gRPC. It wraps the Triton gRPC protocol with ergonomic builder
//! patterns and strong typing while providing zero-cost access to the
//! underlying protobuf types when needed.
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use triton_client::client::TritonClient;
//! use triton_client::infer::{DataType, InferInput, InferRequestBuilder};
//!
//! # async fn example() -> triton_client::error::Result<()> {
//! // Connect to Triton
//! let client = TritonClient::connect("http://localhost:8001").await?;
//!
//! // Check server health
//! assert!(client.is_server_live().await?);
//! assert!(client.is_server_ready().await?);
//!
//! // Build an inference request
//! let input = InferInput::new("input0", vec![1, 16], DataType::Fp32)
//!     .with_data_f32(&[0.0; 16]);
//!
//! let request = InferRequestBuilder::new("my_model")
//!     .model_version("1")
//!     .input(input)
//!     .output("output0")
//!     .build();
//!
//! // Run inference
//! let response = client.infer(request).await?;
//! let output_data = response.output_as_f32(0)?;
//! println!("Output: {:?}", output_data);
//! # Ok(())
//! # }
//! ```
//!
//! # Modules
//!
//! - [`client`] -- The main [`TritonClient`](client::TritonClient) and
//!   connection options.
//! - [`infer`] -- Builder types for inference requests and response wrappers.
//! - [`error`] -- Error types and the [`Result`](error::Result) alias.
//! - [`generated`] -- Raw protobuf/gRPC generated types for advanced usage.

pub mod client;
pub mod error;
pub mod generated;
pub mod infer;

/// Re-export of the main client type for convenience.
pub use client::TritonClient;
