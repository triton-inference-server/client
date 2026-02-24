# triton-client

Rust client library for [NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server).

This crate provides a type-safe, async Rust API for communicating with Triton Inference Server over gRPC. It wraps the Triton gRPC protocol with ergonomic builder patterns and strong typing while providing zero-cost access to the underlying protobuf types when needed.

## Features

- **Async/await** -- Built on `tokio` and `tonic` for efficient async I/O.
- **Type-safe builders** -- `InferInput` and `InferRequestBuilder` catch errors at compile time.
- **High performance** -- Uses raw byte tensor encoding for minimal serialization overhead.
- **Streaming inference** -- First-class support for `ModelStreamInfer`.
- **Full API coverage** -- Health checks, metadata, model management, shared memory, tracing, and logging.
- **Zero-cost escape hatch** -- Access the raw protobuf types via the `generated` module.

## Quick Start

Add `triton-client` to your `Cargo.toml`:

```toml
[dependencies]
triton-client = { path = "src/rust/triton-client" }
tokio = { version = "1", features = ["full"] }
```

### Basic Example

```rust
use triton_client::client::TritonClient;
use triton_client::infer::{DataType, InferInput, InferRequestBuilder};

#[tokio::main]
async fn main() -> triton_client::error::Result<()> {
    // Connect to Triton
    let client = TritonClient::connect("http://localhost:8001").await?;

    // Check health
    assert!(client.is_server_live().await?);
    assert!(client.is_server_ready().await?);

    // Query server metadata
    let metadata = client.server_metadata().await?;
    println!("Server: {} v{}", metadata.name, metadata.version);

    // Build an inference request
    let input = InferInput::new("input0", vec![1, 16], DataType::Fp32)
        .with_data_f32(&[0.0; 16]);

    let request = InferRequestBuilder::new("my_model")
        .model_version("1")
        .input(input)
        .output("output0")
        .build();

    // Run inference
    let response = client.infer(request).await?;
    let output = response.output_as_f32(0)?;
    println!("Output: {:?}", output);

    Ok(())
}
```

### Connection Options

```rust
use std::time::Duration;
use triton_client::client::{ClientOptions, TritonClient};

let options = ClientOptions::default()
    .connect_timeout(Duration::from_secs(10))
    .request_timeout(Duration::from_secs(60))
    .max_message_size(256 * 1024 * 1024)
    .keep_alive_interval(Duration::from_secs(30))
    .keep_alive_timeout(Duration::from_secs(10));

let client = TritonClient::connect_with_options("http://localhost:8001", options).await?;
```

### Model Management

```rust
// List available models
let models = client.repository_index().await?;
for model in &models {
    println!("{} v{} [{}]", model.name, model.version, model.state);
}

// Load / unload models
client.load_model("my_model").await?;
client.unload_model("my_model").await?;
```

### Streaming Inference

```rust
use tokio_stream::StreamExt;

let requests = tokio_stream::iter(vec![request1, request2, request3]);
let mut stream = client.infer_stream(requests).await?;

while let Some(result) = stream.next().await {
    let response = result?;
    println!("Stream response: {}", response.model_name());
}
```

## API Reference

### `TritonClient`

| Method | Description |
|--------|-------------|
| `connect(url)` | Connect with default options |
| `connect_with_options(url, options)` | Connect with custom options |
| `is_server_live()` | Check if the server process is running |
| `is_server_ready()` | Check if the server is ready for inference |
| `is_model_ready(name, version)` | Check if a model is ready |
| `server_metadata()` | Get server name, version, extensions |
| `model_metadata(name, version)` | Get model inputs/outputs metadata |
| `model_config(name, version)` | Get full model configuration |
| `infer(request)` | Run a single inference request |
| `infer_stream(requests)` | Run streaming inference |
| `model_statistics(name, version)` | Get inference statistics |
| `repository_index()` | List models in the repository |
| `load_model(name)` | Load a model |
| `unload_model(name)` | Unload a model |
| `system_shared_memory_status(name)` | Query system shared memory |
| `cuda_shared_memory_status(name)` | Query CUDA shared memory |
| `trace_setting(model, settings)` | Get/set trace settings |
| `log_settings(settings)` | Get/set log settings |

### `InferInput`

Builder for input tensors. Supports all Triton data types:

```rust
// Numeric data
InferInput::new("input", vec![1, 4], DataType::Fp32).with_data_f32(&[1.0, 2.0, 3.0, 4.0])
InferInput::new("input", vec![1, 4], DataType::Int64).with_data_i64(&[1, 2, 3, 4])

// String / bytes data
InferInput::new("text", vec![1, 2], DataType::Bytes).with_data_bytes(&[b"hello", b"world"])

// Raw data (FP16, BF16, or any format)
InferInput::new("input", vec![1, 2], DataType::Fp16).with_data_raw(raw_bytes)
```

## Testing

```bash
# Unit tests (no server required)
cargo test

# Integration tests (requires running Triton server)
TRITON_TEST_URL=http://localhost:8001 cargo test
```

## Building

Requires `protoc` (Protocol Buffers compiler) to be installed:

```bash
# macOS
brew install protobuf

# Ubuntu/Debian
apt-get install protobuf-compiler
```

Then build:

```bash
cargo build
```

## License

BSD-3-Clause. See the license header in each source file for details.
