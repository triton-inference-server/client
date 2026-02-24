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

//! Simple gRPC inference example.
//!
//! Demonstrates connecting to a Triton Inference Server, checking health,
//! querying model metadata, and running a basic inference request.
//!
//! This example assumes a Triton server is running at `localhost:8001` with
//! a model named `simple` that accepts two INT32 inputs (`INPUT0`, `INPUT1`)
//! of shape `[1, 16]` and produces two INT32 outputs (`OUTPUT0`, `OUTPUT1`).
//!
//! # Usage
//!
//! ```bash
//! cargo run --example simple_grpc_infer
//! ```
//!
//! Optionally pass a custom server URL:
//!
//! ```bash
//! cargo run --example simple_grpc_infer -- http://triton-server:8001
//! ```

use triton_client::client::TritonClient;
use triton_client::error::Result;
use triton_client::infer::{DataType, InferInput, InferRequestBuilder};

#[tokio::main]
async fn main() -> Result<()> {
    let url = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "http://localhost:8001".to_owned());

    println!("Connecting to Triton at {url}...");
    let client = TritonClient::connect(&url).await?;

    // -- Health checks -------------------------------------------------------

    let live = client.is_server_live().await?;
    println!("Server live: {live}");

    let ready = client.is_server_ready().await?;
    println!("Server ready: {ready}");

    // -- Server metadata -----------------------------------------------------

    let metadata = client.server_metadata().await?;
    println!(
        "Server: {} v{} (extensions: {:?})",
        metadata.name, metadata.version, metadata.extensions
    );

    // -- Model repository index ----------------------------------------------

    let models = client.repository_index().await?;
    println!("Available models:");
    for model in &models {
        println!("  {} v{} [{}]", model.name, model.version, model.state);
    }

    // -- Model metadata ------------------------------------------------------

    let model_name = "simple";
    let model_version = "";

    let model_ready = client.is_model_ready(model_name, model_version).await?;
    println!("\nModel '{model_name}' ready: {model_ready}");

    if !model_ready {
        println!("Model is not ready, skipping inference.");
        return Ok(());
    }

    let model_meta = client.model_metadata(model_name, model_version).await?;
    println!(
        "Model: {} (platform: {})",
        model_meta.name, model_meta.platform
    );
    for input in &model_meta.inputs {
        println!(
            "  Input: {} ({}) {:?}",
            input.name, input.datatype, input.shape
        );
    }
    for output in &model_meta.outputs {
        println!(
            "  Output: {} ({}) {:?}",
            output.name, output.datatype, output.shape
        );
    }

    // -- Inference -----------------------------------------------------------

    // Create input data: two INT32 tensors of shape [1, 16].
    let input_data: Vec<i32> = (0..16).collect();

    let input0 = InferInput::new("INPUT0", vec![1, 16], DataType::Int32).with_data_i32(&input_data);

    let input1 = InferInput::new("INPUT1", vec![1, 16], DataType::Int32).with_data_i32(&input_data);

    let request = InferRequestBuilder::new(model_name)
        .request_id("example-001")
        .input(input0)
        .input(input1)
        .output("OUTPUT0")
        .output("OUTPUT1")
        .build();

    println!("\nRunning inference...");
    let response = client.infer(request).await?;

    println!("Response id: {}", response.id());
    println!(
        "Model: {} v{}",
        response.model_name(),
        response.model_version()
    );

    for (i, output) in response.outputs().iter().enumerate() {
        println!(
            "Output {}: {} ({}) {:?}",
            i, output.name, output.datatype, output.shape
        );
        let data = response.output_as_i32(i)?;
        println!("  Data: {:?}", &data[..data.len().min(8)]);
    }

    println!("\nDone!");
    Ok(())
}
