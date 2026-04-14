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

//! Integration tests for the Triton client library.
//!
//! Tests that require a running Triton server are gated behind the
//! `TRITON_TEST_URL` environment variable. When the variable is not set,
//! only offline tests (builder patterns, error types, etc.) are executed.

use triton_client::client::{ClientOptions, TritonClient};
use triton_client::error::Error;
use triton_client::infer::{
    DataType, InferInput, InferRequestBuilder, InferRequestedOutput, InferResponse,
};

/// Helper to get the Triton server URL from the environment.
fn triton_url() -> Option<String> {
    std::env::var("TRITON_TEST_URL").ok()
}

// ---------------------------------------------------------------------------
// Offline tests (no server required)
// ---------------------------------------------------------------------------

#[test]
fn client_options_defaults() {
    let options = ClientOptions::default();
    // Ensure default construction does not panic.
    let _options = options
        .connect_timeout(std::time::Duration::from_secs(10))
        .request_timeout(std::time::Duration::from_secs(30))
        .max_message_size(256 * 1024 * 1024)
        .keep_alive_interval(std::time::Duration::from_secs(60))
        .keep_alive_timeout(std::time::Duration::from_secs(20));
}

#[test]
fn error_display_messages() {
    let err = Error::Connection("refused".into());
    assert!(format!("{err}").contains("refused"));

    let err = Error::InvalidInput("bad shape".into());
    assert!(format!("{err}").contains("bad shape"));

    let err = Error::ModelNotReady {
        model_name: "resnet".into(),
        model_version: "1".into(),
    };
    assert!(format!("{err}").contains("resnet"));

    let err = Error::ServerNotReady;
    assert!(format!("{err}").contains("server not ready"));

    let err = Error::StreamInference("timeout".into());
    assert!(format!("{err}").contains("timeout"));

    let err = Error::UnexpectedResponse("no data".into());
    assert!(format!("{err}").contains("no data"));
}

#[test]
fn error_from_tonic_status() {
    let status = tonic::Status::not_found("model not found");
    let err: Error = status.into();
    match &err {
        Error::Grpc { code, message } => {
            assert_eq!(*code, tonic::Code::NotFound);
            assert!(message.contains("model not found"));
        }
        other => panic!("expected Grpc error, got: {other}"),
    }
}

#[test]
fn data_type_completeness() {
    // Verify all DataType variants have a string form.
    let all_types = [
        DataType::Bool,
        DataType::Uint8,
        DataType::Uint16,
        DataType::Uint32,
        DataType::Uint64,
        DataType::Int8,
        DataType::Int16,
        DataType::Int32,
        DataType::Int64,
        DataType::Fp16,
        DataType::Fp32,
        DataType::Fp64,
        DataType::Bytes,
        DataType::Bf16,
    ];
    for dt in &all_types {
        let s = dt.as_str();
        assert!(!s.is_empty());
        assert_eq!(DataType::parse(s), Some(*dt));
    }
}

#[test]
fn infer_input_builder_chain() {
    let input = InferInput::new("features", vec![1, 768], DataType::Fp32)
        .with_data_f32(&vec![0.1f32; 768])
        .with_string_parameter("key", "value")
        .with_int_parameter("priority", 5)
        .with_bool_parameter("flag", false);

    assert_eq!(input.name(), "features");
    assert_eq!(input.shape(), &[1, 768]);
    assert_eq!(input.datatype(), DataType::Fp32);
}

#[test]
fn infer_input_raw_data() {
    let raw_fp16: Vec<u8> = vec![0x00, 0x3C, 0x00, 0x40]; // FP16: 1.0, 2.0
    let input = InferInput::new("input", vec![1, 2], DataType::Fp16).with_data_raw(raw_fp16);

    // Verify via the builder -- the raw bytes end up in raw_input_contents.
    let request = InferRequestBuilder::new("model").input(input).build();
    assert_eq!(request.inputs[0].datatype, "FP16");
    assert_eq!(request.raw_input_contents[0].len(), 4);
}

#[test]
fn infer_request_builder_full() {
    let input0 = InferInput::new("input0", vec![2, 4], DataType::Fp32).with_data_f32(&[0.0; 8]);
    let input1 = InferInput::new("input1", vec![2, 4], DataType::Fp32).with_data_f32(&[1.0; 8]);

    let output = InferRequestedOutput::new("output0").with_string_parameter("class_count", "3");

    let request = InferRequestBuilder::new("ensemble_model")
        .model_version("2")
        .request_id("batch-042")
        .input(input0)
        .input(input1)
        .output_with(output)
        .output("output1")
        .string_parameter("sequence_id", "seq-1")
        .int_parameter("sequence_start", 1)
        .bool_parameter("priority_flag", true)
        .build();

    assert_eq!(request.model_name, "ensemble_model");
    assert_eq!(request.model_version, "2");
    assert_eq!(request.id, "batch-042");
    assert_eq!(request.inputs.len(), 2);
    assert_eq!(request.outputs.len(), 2);
    assert_eq!(request.raw_input_contents.len(), 2);
    assert_eq!(request.parameters.len(), 3);
}

#[test]
fn infer_response_f32_round_trip() {
    let values = vec![1.5f32, 2.5, -3.0, 0.0];
    let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

    let proto = triton_client::generated::inference::ModelInferResponse {
        model_name: "test".into(),
        model_version: "1".into(),
        id: "req-1".into(),
        parameters: Default::default(),
        outputs: vec![
            triton_client::generated::inference::model_infer_response::InferOutputTensor {
                name: "output0".into(),
                datatype: "FP32".into(),
                shape: vec![1, 4],
                parameters: Default::default(),
                contents: None,
            },
        ],
        raw_output_contents: vec![raw],
    };

    let response = InferResponse::new(proto);
    assert_eq!(response.model_name(), "test");
    assert_eq!(response.model_version(), "1");

    let result = response.output_as_f32(0).unwrap();
    assert_eq!(result, values);
}

#[test]
fn infer_response_i64_round_trip() {
    let values = vec![i64::MIN, -1, 0, 1, i64::MAX];
    let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

    let proto = triton_client::generated::inference::ModelInferResponse {
        model_name: String::new(),
        model_version: String::new(),
        id: String::new(),
        parameters: Default::default(),
        outputs: vec![],
        raw_output_contents: vec![raw],
    };

    let response = InferResponse::new(proto);
    let result = response.output_as_i64(0).unwrap();
    assert_eq!(result, values);
}

#[test]
fn infer_response_output_lookup_by_name() {
    let proto = triton_client::generated::inference::ModelInferResponse {
        model_name: String::new(),
        model_version: String::new(),
        id: String::new(),
        parameters: Default::default(),
        outputs: vec![
            triton_client::generated::inference::model_infer_response::InferOutputTensor {
                name: "alpha".into(),
                datatype: "FP32".into(),
                shape: vec![1],
                parameters: Default::default(),
                contents: None,
            },
            triton_client::generated::inference::model_infer_response::InferOutputTensor {
                name: "beta".into(),
                datatype: "INT64".into(),
                shape: vec![2],
                parameters: Default::default(),
                contents: None,
            },
        ],
        raw_output_contents: vec![],
    };

    let response = InferResponse::new(proto);
    assert!(response.output("alpha").is_some());
    assert!(response.output("beta").is_some());
    assert!(response.output("gamma").is_none());
}

// ---------------------------------------------------------------------------
// Online tests (require TRITON_TEST_URL)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn connect_to_invalid_url_fails() {
    // Use TEST-NET (RFC 5737) address which is guaranteed non-routable,
    // with a short timeout to avoid long waits.
    let options = ClientOptions::default()
        .connect_timeout(std::time::Duration::from_millis(200));
    let result = TritonClient::connect_with_options("http://192.0.2.1:1", options).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn online_server_health() {
    let Some(url) = triton_url() else {
        eprintln!("Skipping online test: TRITON_TEST_URL not set");
        return;
    };
    let client = TritonClient::connect(&url).await.unwrap();
    let live = client.is_server_live().await.unwrap();
    assert!(live, "Expected server to be live");
    let ready = client.is_server_ready().await.unwrap();
    assert!(ready, "Expected server to be ready");
}

#[tokio::test]
async fn online_server_metadata() {
    let Some(url) = triton_url() else {
        eprintln!("Skipping online test: TRITON_TEST_URL not set");
        return;
    };
    let client = TritonClient::connect(&url).await.unwrap();
    let metadata = client.server_metadata().await.unwrap();
    assert!(!metadata.name.is_empty(), "Server name should not be empty");
    assert!(
        !metadata.version.is_empty(),
        "Server version should not be empty"
    );
}

#[tokio::test]
async fn online_repository_index() {
    let Some(url) = triton_url() else {
        eprintln!("Skipping online test: TRITON_TEST_URL not set");
        return;
    };
    let client = TritonClient::connect(&url).await.unwrap();
    let models = client.repository_index().await.unwrap();
    // Just verify the call succeeds; the list may be empty.
    eprintln!("Repository contains {} models", models.len());
}

#[tokio::test]
async fn online_infer_identity_fp32() {
    let Some(url) = triton_url() else {
        eprintln!("Skipping online test: TRITON_TEST_URL not set");
        return;
    };
    let client = TritonClient::connect(&url).await.unwrap();

    // Send a known FP32 vector through the identity backend and verify
    // the output matches the input exactly.
    let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let input = InferInput::new("INPUT0", vec![5], DataType::Fp32)
        .with_data_f32(&input_data);

    let request = InferRequestBuilder::new("identity_fp32")
        .model_version("1")
        .request_id("integration-test-001")
        .input(input)
        .output("OUTPUT0")
        .build();

    let response = client.infer(request).await.unwrap();
    assert_eq!(response.model_name(), "identity_fp32");

    let output_data = response.output_as_f32(0).unwrap();
    assert_eq!(
        output_data, input_data,
        "Identity model should return input unchanged"
    );
    eprintln!(
        "Identity inference passed: {:?} -> {:?}",
        input_data, output_data
    );
}

#[tokio::test]
async fn online_model_metadata() {
    let Some(url) = triton_url() else {
        eprintln!("Skipping online test: TRITON_TEST_URL not set");
        return;
    };
    let client = TritonClient::connect(&url).await.unwrap();

    let metadata = client.model_metadata("identity_fp32", "1").await.unwrap();
    assert_eq!(metadata.name, "identity_fp32");
    assert!(!metadata.inputs.is_empty(), "Model should have inputs");
    assert!(!metadata.outputs.is_empty(), "Model should have outputs");
    eprintln!(
        "Model metadata: {} inputs, {} outputs",
        metadata.inputs.len(),
        metadata.outputs.len()
    );
}

#[tokio::test]
async fn online_model_ready() {
    let Some(url) = triton_url() else {
        eprintln!("Skipping online test: TRITON_TEST_URL not set");
        return;
    };
    let client = TritonClient::connect(&url).await.unwrap();

    let ready = client
        .is_model_ready("identity_fp32", "1")
        .await
        .unwrap();
    assert!(ready, "identity_fp32 model should be ready");

    let ready = client.is_model_ready("simple", "1").await.unwrap();
    assert!(ready, "simple model should be ready");
}
