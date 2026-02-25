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

//! Builder types for constructing inference requests and processing responses.
//!
//! This module provides an ergonomic, type-safe API for building inference
//! requests. The main entry points are [`InferInput`] for describing input
//! tensors and [`InferRequestBuilder`] for assembling a complete request.
//!
//! # Example
//!
//! ```rust
//! use triton_client::infer::{DataType, InferInput, InferRequestBuilder};
//!
//! let input = InferInput::new("input0", vec![1, 16], DataType::Fp32)
//!     .with_data_f32(&[1.0; 16]);
//!
//! let request = InferRequestBuilder::new("my_model")
//!     .model_version("1")
//!     .request_id("req-001")
//!     .input(input)
//!     .output("output0")
//!     .build();
//! ```

use std::collections::HashMap;

use crate::error::{Error, Result};
use crate::generated::inference;

// ---------------------------------------------------------------------------
// DataType
// ---------------------------------------------------------------------------

/// Triton data types corresponding to the protocol's tensor data types.
///
/// These map to the string representations expected by the Triton gRPC
/// protocol (e.g. `"FP32"`, `"INT64"`, `"BYTES"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataType {
    /// Boolean values.
    Bool,
    /// Unsigned 8-bit integers.
    Uint8,
    /// Unsigned 16-bit integers.
    Uint16,
    /// Unsigned 32-bit integers.
    Uint32,
    /// Unsigned 64-bit integers.
    Uint64,
    /// Signed 8-bit integers.
    Int8,
    /// Signed 16-bit integers.
    Int16,
    /// Signed 32-bit integers.
    Int32,
    /// Signed 64-bit integers.
    Int64,
    /// IEEE 754 half-precision (16-bit) floating point.
    Fp16,
    /// IEEE 754 single-precision (32-bit) floating point.
    Fp32,
    /// IEEE 754 double-precision (64-bit) floating point.
    Fp64,
    /// Variable-length byte sequences (strings).
    Bytes,
    /// Brain floating point (16-bit).
    Bf16,
}

impl DataType {
    /// Returns the Triton protocol string representation of this data type.
    ///
    /// # Example
    ///
    /// ```rust
    /// use triton_client::infer::DataType;
    /// assert_eq!(DataType::Fp32.as_str(), "FP32");
    /// assert_eq!(DataType::Int64.as_str(), "INT64");
    /// ```
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Bool => "BOOL",
            Self::Uint8 => "UINT8",
            Self::Uint16 => "UINT16",
            Self::Uint32 => "UINT32",
            Self::Uint64 => "UINT64",
            Self::Int8 => "INT8",
            Self::Int16 => "INT16",
            Self::Int32 => "INT32",
            Self::Int64 => "INT64",
            Self::Fp16 => "FP16",
            Self::Fp32 => "FP32",
            Self::Fp64 => "FP64",
            Self::Bytes => "BYTES",
            Self::Bf16 => "BF16",
        }
    }

    /// Parses a Triton data type string into a [`DataType`].
    ///
    /// Returns `None` if the string does not correspond to a known type.
    ///
    /// # Example
    ///
    /// ```rust
    /// use triton_client::infer::DataType;
    /// assert_eq!(DataType::parse("FP32"), Some(DataType::Fp32));
    /// assert_eq!(DataType::parse("UNKNOWN"), None);
    /// ```
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "BOOL" => Some(Self::Bool),
            "UINT8" => Some(Self::Uint8),
            "UINT16" => Some(Self::Uint16),
            "UINT32" => Some(Self::Uint32),
            "UINT64" => Some(Self::Uint64),
            "INT8" => Some(Self::Int8),
            "INT16" => Some(Self::Int16),
            "INT32" => Some(Self::Int32),
            "INT64" => Some(Self::Int64),
            "FP16" => Some(Self::Fp16),
            "FP32" => Some(Self::Fp32),
            "FP64" => Some(Self::Fp64),
            "BYTES" => Some(Self::Bytes),
            "BF16" => Some(Self::Bf16),
            _ => None,
        }
    }
}

impl std::fmt::Display for DataType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Error returned when parsing an unknown data type string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseDataTypeError(String);

impl std::fmt::Display for ParseDataTypeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "unknown Triton data type: {}", self.0)
    }
}

impl std::error::Error for ParseDataTypeError {}

impl std::str::FromStr for DataType {
    type Err = ParseDataTypeError;

    /// Parses a Triton data type string.
    ///
    /// # Example
    ///
    /// ```rust
    /// use triton_client::infer::DataType;
    /// let dt: DataType = "FP32".parse().unwrap();
    /// assert_eq!(dt, DataType::Fp32);
    /// ```
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        DataType::parse(s).ok_or_else(|| ParseDataTypeError(s.to_owned()))
    }
}

// ---------------------------------------------------------------------------
// InferInput
// ---------------------------------------------------------------------------

/// Describes an input tensor for an inference request.
///
/// Use the `with_data_*` methods to attach tensor data in the appropriate
/// format. Data is stored as raw bytes for maximum efficiency over gRPC.
///
/// # Example
///
/// ```rust
/// use triton_client::infer::{DataType, InferInput};
///
/// let input = InferInput::new("images", vec![1, 3, 224, 224], DataType::Fp32)
///     .with_data_f32(&vec![0.0_f32; 3 * 224 * 224]);
/// ```
#[derive(Debug, Clone)]
pub struct InferInput {
    name: String,
    shape: Vec<i64>,
    datatype: DataType,
    data: Option<Vec<u8>>,
    parameters: HashMap<String, inference::InferParameter>,
}

impl InferInput {
    /// Creates a new inference input descriptor.
    ///
    /// # Arguments
    ///
    /// * `name` -- The tensor name as defined in the model configuration.
    /// * `shape` -- The shape of the tensor (e.g. `vec![1, 3, 224, 224]`).
    /// * `datatype` -- The element data type.
    #[must_use]
    pub fn new(name: impl Into<String>, shape: Vec<i64>, datatype: DataType) -> Self {
        Self {
            name: name.into(),
            shape,
            datatype,
            data: None,
            parameters: HashMap::new(),
        }
    }

    /// Attaches boolean tensor data.
    #[must_use]
    pub fn with_data_bool(self, data: &[bool]) -> Self {
        let raw: Vec<u8> = data.iter().map(|&b| u8::from(b)).collect();
        Self {
            data: Some(raw),
            ..self
        }
    }

    /// Attaches unsigned 8-bit integer tensor data.
    ///
    /// Also suitable for any data already in raw byte form where each
    /// element is a single byte.
    #[must_use]
    pub fn with_data_u8(self, data: &[u8]) -> Self {
        Self {
            data: Some(data.to_vec()),
            ..self
        }
    }

    /// Attaches signed 8-bit integer tensor data.
    #[must_use]
    pub fn with_data_i8(self, data: &[i8]) -> Self {
        let raw: Vec<u8> = data.iter().map(|&v| v as u8).collect();
        Self {
            data: Some(raw),
            ..self
        }
    }

    /// Attaches unsigned 16-bit integer tensor data.
    #[must_use]
    pub fn with_data_u16(self, data: &[u16]) -> Self {
        let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        Self {
            data: Some(raw),
            ..self
        }
    }

    /// Attaches signed 16-bit integer tensor data.
    #[must_use]
    pub fn with_data_i16(self, data: &[i16]) -> Self {
        let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        Self {
            data: Some(raw),
            ..self
        }
    }

    /// Attaches 32-bit floating-point tensor data.
    #[must_use]
    pub fn with_data_f32(self, data: &[f32]) -> Self {
        let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        Self {
            data: Some(raw),
            ..self
        }
    }

    /// Attaches 64-bit floating-point tensor data.
    #[must_use]
    pub fn with_data_f64(self, data: &[f64]) -> Self {
        let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        Self {
            data: Some(raw),
            ..self
        }
    }

    /// Attaches signed 32-bit integer tensor data.
    #[must_use]
    pub fn with_data_i32(self, data: &[i32]) -> Self {
        let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        Self {
            data: Some(raw),
            ..self
        }
    }

    /// Attaches signed 64-bit integer tensor data.
    #[must_use]
    pub fn with_data_i64(self, data: &[i64]) -> Self {
        let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        Self {
            data: Some(raw),
            ..self
        }
    }

    /// Attaches unsigned 32-bit integer tensor data.
    #[must_use]
    pub fn with_data_u32(self, data: &[u32]) -> Self {
        let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        Self {
            data: Some(raw),
            ..self
        }
    }

    /// Attaches unsigned 64-bit integer tensor data.
    #[must_use]
    pub fn with_data_u64(self, data: &[u64]) -> Self {
        let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        Self {
            data: Some(raw),
            ..self
        }
    }

    /// Attaches raw byte data.
    ///
    /// This is the most general form and can be used for any data type,
    /// including FP16 and BF16 which lack native Rust types. The caller
    /// is responsible for ensuring the bytes are in the correct format
    /// (little-endian, row-major order).
    #[must_use]
    pub fn with_data_raw(self, data: Vec<u8>) -> Self {
        Self {
            data: Some(data),
            ..self
        }
    }

    /// Attaches variable-length byte sequences (strings).
    ///
    /// Each byte slice is prepended with its 4-byte little-endian length,
    /// following the Triton BYTES tensor encoding.
    #[must_use]
    pub fn with_data_bytes(self, data: &[&[u8]]) -> Self {
        let mut raw = Vec::new();
        for item in data {
            #[allow(clippy::cast_possible_truncation)]
            let len = u32::try_from(item.len())
                .expect("byte sequence length exceeds u32::MAX");
            raw.extend_from_slice(&len.to_le_bytes());
            raw.extend_from_slice(item);
        }
        Self {
            data: Some(raw),
            ..self
        }
    }

    /// Adds a string parameter to this input tensor.
    #[must_use]
    pub fn with_string_parameter(
        mut self,
        key: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        self.parameters.insert(
            key.into(),
            inference::InferParameter {
                parameter_choice: Some(inference::infer_parameter::ParameterChoice::StringParam(
                    value.into(),
                )),
            },
        );
        self
    }

    /// Adds an integer parameter to this input tensor.
    #[must_use]
    pub fn with_int_parameter(mut self, key: impl Into<String>, value: i64) -> Self {
        self.parameters.insert(
            key.into(),
            inference::InferParameter {
                parameter_choice: Some(inference::infer_parameter::ParameterChoice::Int64Param(
                    value,
                )),
            },
        );
        self
    }

    /// Adds a boolean parameter to this input tensor.
    #[must_use]
    pub fn with_bool_parameter(mut self, key: impl Into<String>, value: bool) -> Self {
        self.parameters.insert(
            key.into(),
            inference::InferParameter {
                parameter_choice: Some(inference::infer_parameter::ParameterChoice::BoolParam(
                    value,
                )),
            },
        );
        self
    }

    /// Returns the tensor name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the tensor shape.
    #[must_use]
    pub fn shape(&self) -> &[i64] {
        &self.shape
    }

    /// Returns the tensor data type.
    #[must_use]
    pub fn datatype(&self) -> DataType {
        self.datatype
    }

    /// Converts this input into the protobuf tensor and optional raw bytes.
    pub(crate) fn into_proto(
        self,
    ) -> (
        inference::model_infer_request::InferInputTensor,
        Option<Vec<u8>>,
    ) {
        let tensor = inference::model_infer_request::InferInputTensor {
            name: self.name,
            datatype: self.datatype.as_str().to_owned(),
            shape: self.shape,
            parameters: self.parameters,
            contents: None, // We use raw_input_contents for performance.
        };
        (tensor, self.data)
    }
}

// ---------------------------------------------------------------------------
// InferRequestedOutput
// ---------------------------------------------------------------------------

/// Describes a requested output tensor for an inference request.
///
/// Specifying outputs is optional. When no outputs are requested, the server
/// returns all outputs defined in the model configuration.
#[derive(Debug, Clone)]
pub struct InferRequestedOutput {
    name: String,
    parameters: HashMap<String, inference::InferParameter>,
}

impl InferRequestedOutput {
    /// Creates a new requested output for the tensor with the given name.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            parameters: HashMap::new(),
        }
    }

    /// Adds a string parameter to this output request.
    #[must_use]
    pub fn with_string_parameter(
        mut self,
        key: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        self.parameters.insert(
            key.into(),
            inference::InferParameter {
                parameter_choice: Some(inference::infer_parameter::ParameterChoice::StringParam(
                    value.into(),
                )),
            },
        );
        self
    }

    /// Returns the output tensor name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Converts this output descriptor into the protobuf type.
    pub(crate) fn into_proto(self) -> inference::model_infer_request::InferRequestedOutputTensor {
        inference::model_infer_request::InferRequestedOutputTensor {
            name: self.name,
            parameters: self.parameters,
        }
    }
}

// ---------------------------------------------------------------------------
// InferRequestBuilder
// ---------------------------------------------------------------------------

/// Builder for constructing [`inference::ModelInferRequest`] messages.
///
/// # Example
///
/// ```rust
/// use triton_client::infer::{DataType, InferInput, InferRequestBuilder};
///
/// let request = InferRequestBuilder::new("resnet50")
///     .model_version("1")
///     .request_id("batch-001")
///     .input(
///         InferInput::new("input", vec![1, 3, 224, 224], DataType::Fp32)
///             .with_data_f32(&vec![0.0_f32; 3 * 224 * 224]),
///     )
///     .output("output")
///     .build();
/// ```
#[derive(Debug)]
pub struct InferRequestBuilder {
    model_name: String,
    model_version: String,
    request_id: String,
    inputs: Vec<InferInput>,
    outputs: Vec<InferRequestedOutput>,
    parameters: HashMap<String, inference::InferParameter>,
}

impl InferRequestBuilder {
    /// Creates a new builder targeting the specified model.
    #[must_use]
    pub fn new(model_name: impl Into<String>) -> Self {
        Self {
            model_name: model_name.into(),
            model_version: String::new(),
            request_id: String::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            parameters: HashMap::new(),
        }
    }

    /// Sets the model version to use for inference.
    ///
    /// If not set, the server uses the latest version according to its policy.
    #[must_use]
    pub fn model_version(self, version: impl Into<String>) -> Self {
        Self {
            model_version: version.into(),
            ..self
        }
    }

    /// Sets an optional request identifier.
    ///
    /// When specified, the server echoes this identifier in the response.
    #[must_use]
    pub fn request_id(self, id: impl Into<String>) -> Self {
        Self {
            request_id: id.into(),
            ..self
        }
    }

    /// Adds an input tensor to the request.
    #[must_use]
    pub fn input(mut self, input: InferInput) -> Self {
        self.inputs.push(input);
        self
    }

    /// Adds multiple input tensors to the request.
    #[must_use]
    pub fn inputs(mut self, inputs: impl IntoIterator<Item = InferInput>) -> Self {
        self.inputs.extend(inputs);
        self
    }

    /// Adds a requested output by name.
    ///
    /// This is a convenience method that creates an [`InferRequestedOutput`]
    /// with no additional parameters.
    #[must_use]
    pub fn output(mut self, name: impl Into<String>) -> Self {
        self.outputs.push(InferRequestedOutput::new(name));
        self
    }

    /// Adds a fully-configured requested output.
    #[must_use]
    pub fn output_with(mut self, output: InferRequestedOutput) -> Self {
        self.outputs.push(output);
        self
    }

    /// Adds a string inference parameter.
    #[must_use]
    pub fn string_parameter(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.parameters.insert(
            key.into(),
            inference::InferParameter {
                parameter_choice: Some(inference::infer_parameter::ParameterChoice::StringParam(
                    value.into(),
                )),
            },
        );
        self
    }

    /// Adds an integer inference parameter.
    #[must_use]
    pub fn int_parameter(mut self, key: impl Into<String>, value: i64) -> Self {
        self.parameters.insert(
            key.into(),
            inference::InferParameter {
                parameter_choice: Some(inference::infer_parameter::ParameterChoice::Int64Param(
                    value,
                )),
            },
        );
        self
    }

    /// Adds a boolean inference parameter.
    #[must_use]
    pub fn bool_parameter(mut self, key: impl Into<String>, value: bool) -> Self {
        self.parameters.insert(
            key.into(),
            inference::InferParameter {
                parameter_choice: Some(inference::infer_parameter::ParameterChoice::BoolParam(
                    value,
                )),
            },
        );
        self
    }

    /// Consumes the builder and produces a [`inference::ModelInferRequest`].
    ///
    /// Input data is placed into `raw_input_contents` for optimal performance
    /// over gRPC, following the Triton protocol recommendation.
    #[must_use]
    pub fn build(self) -> inference::ModelInferRequest {
        let mut input_tensors = Vec::with_capacity(self.inputs.len());
        let mut raw_input_contents = Vec::with_capacity(self.inputs.len());

        for input in self.inputs {
            let (tensor, raw_data) = input.into_proto();
            input_tensors.push(tensor);
            raw_input_contents.push(raw_data.unwrap_or_default());
        }

        let output_tensors: Vec<_> = self
            .outputs
            .into_iter()
            .map(InferRequestedOutput::into_proto)
            .collect();

        inference::ModelInferRequest {
            model_name: self.model_name,
            model_version: self.model_version,
            id: self.request_id,
            parameters: self.parameters,
            inputs: input_tensors,
            outputs: output_tensors,
            raw_input_contents,
        }
    }
}

// ---------------------------------------------------------------------------
// InferResponse
// ---------------------------------------------------------------------------

/// Ergonomic wrapper around the raw protobuf inference response.
///
/// Provides convenient accessors for output tensor data without requiring
/// direct interaction with the generated protobuf types.
#[derive(Debug, Clone)]
pub struct InferResponse {
    inner: inference::ModelInferResponse,
}

impl InferResponse {
    /// Creates a new `InferResponse` from the raw protobuf message.
    ///
    /// This is useful for testing or when constructing responses from raw
    /// protobuf data obtained outside of the normal client flow.
    #[must_use]
    pub fn new(inner: inference::ModelInferResponse) -> Self {
        Self { inner }
    }

    /// Returns the model name that produced this response.
    #[must_use]
    pub fn model_name(&self) -> &str {
        &self.inner.model_name
    }

    /// Returns the model version that produced this response.
    #[must_use]
    pub fn model_version(&self) -> &str {
        &self.inner.model_version
    }

    /// Returns the request identifier, if one was set in the request.
    #[must_use]
    pub fn id(&self) -> &str {
        &self.inner.id
    }

    /// Returns the list of output tensors in the response.
    #[must_use]
    pub fn outputs(&self) -> &[inference::model_infer_response::InferOutputTensor] {
        &self.inner.outputs
    }

    /// Finds an output tensor by name.
    #[must_use]
    pub fn output(
        &self,
        name: &str,
    ) -> Option<&inference::model_infer_response::InferOutputTensor> {
        self.inner.outputs.iter().find(|o| o.name == name)
    }

    /// Returns the raw output contents as byte slices.
    ///
    /// The raw contents are ordered to correspond 1-to-1 with the output
    /// tensors returned in [`outputs()`](Self::outputs). This is the
    /// high-performance representation recommended by the Triton protocol.
    #[must_use]
    pub fn raw_output_contents(&self) -> &[Vec<u8>] {
        &self.inner.raw_output_contents
    }

    /// Interprets the raw output bytes for the given output index as `f32`
    /// values.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidInput`] if the index is out of bounds or the
    /// byte length is not a multiple of 4.
    pub fn output_as_f32(&self, index: usize) -> Result<Vec<f32>> {
        let raw =
            self.inner.raw_output_contents.get(index).ok_or_else(|| {
                Error::InvalidInput(format!("output index {index} out of bounds"))
            })?;
        if raw.len() % 4 != 0 {
            return Err(Error::InvalidInput(format!(
                "raw output at index {index} has {} bytes, which is not a multiple of 4",
                raw.len()
            )));
        }
        Ok(raw
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect())
    }

    /// Interprets the raw output bytes for the given output index as `f64`
    /// values.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidInput`] if the index is out of bounds or the
    /// byte length is not a multiple of 8.
    pub fn output_as_f64(&self, index: usize) -> Result<Vec<f64>> {
        let raw =
            self.inner.raw_output_contents.get(index).ok_or_else(|| {
                Error::InvalidInput(format!("output index {index} out of bounds"))
            })?;
        if raw.len() % 8 != 0 {
            return Err(Error::InvalidInput(format!(
                "raw output at index {index} has {} bytes, which is not a multiple of 8",
                raw.len()
            )));
        }
        Ok(raw
            .chunks_exact(8)
            .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
            .collect())
    }

    /// Interprets the raw output bytes for the given output index as `i32`
    /// values.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidInput`] if the index is out of bounds or the
    /// byte length is not a multiple of 4.
    pub fn output_as_i32(&self, index: usize) -> Result<Vec<i32>> {
        let raw =
            self.inner.raw_output_contents.get(index).ok_or_else(|| {
                Error::InvalidInput(format!("output index {index} out of bounds"))
            })?;
        if raw.len() % 4 != 0 {
            return Err(Error::InvalidInput(format!(
                "raw output at index {index} has {} bytes, which is not a multiple of 4",
                raw.len()
            )));
        }
        Ok(raw
            .chunks_exact(4)
            .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
            .collect())
    }

    /// Interprets the raw output bytes for the given output index as `i64`
    /// values.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidInput`] if the index is out of bounds or the
    /// byte length is not a multiple of 8.
    pub fn output_as_i64(&self, index: usize) -> Result<Vec<i64>> {
        let raw =
            self.inner.raw_output_contents.get(index).ok_or_else(|| {
                Error::InvalidInput(format!("output index {index} out of bounds"))
            })?;
        if raw.len() % 8 != 0 {
            return Err(Error::InvalidInput(format!(
                "raw output at index {index} has {} bytes, which is not a multiple of 8",
                raw.len()
            )));
        }
        Ok(raw
            .chunks_exact(8)
            .map(|chunk| i64::from_le_bytes(chunk.try_into().unwrap()))
            .collect())
    }

    /// Interprets the raw output bytes for the given output index as `u32`
    /// values.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidInput`] if the index is out of bounds or the
    /// byte length is not a multiple of 4.
    pub fn output_as_u32(&self, index: usize) -> Result<Vec<u32>> {
        let raw =
            self.inner.raw_output_contents.get(index).ok_or_else(|| {
                Error::InvalidInput(format!("output index {index} out of bounds"))
            })?;
        if raw.len() % 4 != 0 {
            return Err(Error::InvalidInput(format!(
                "raw output at index {index} has {} bytes, which is not a multiple of 4",
                raw.len()
            )));
        }
        Ok(raw
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect())
    }

    /// Interprets the raw output bytes for the given output index as `u64`
    /// values.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidInput`] if the index is out of bounds or the
    /// byte length is not a multiple of 8.
    pub fn output_as_u64(&self, index: usize) -> Result<Vec<u64>> {
        let raw =
            self.inner.raw_output_contents.get(index).ok_or_else(|| {
                Error::InvalidInput(format!("output index {index} out of bounds"))
            })?;
        if raw.len() % 8 != 0 {
            return Err(Error::InvalidInput(format!(
                "raw output at index {index} has {} bytes, which is not a multiple of 8",
                raw.len()
            )));
        }
        Ok(raw
            .chunks_exact(8)
            .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()))
            .collect())
    }

    /// Returns the raw bytes for the output at the given index.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidInput`] if the index is out of bounds.
    pub fn output_as_raw(&self, index: usize) -> Result<&[u8]> {
        self.inner
            .raw_output_contents
            .get(index)
            .map(Vec::as_slice)
            .ok_or_else(|| Error::InvalidInput(format!("output index {index} out of bounds")))
    }

    /// Consumes this wrapper and returns the underlying protobuf response.
    #[must_use]
    pub fn into_inner(self) -> inference::ModelInferResponse {
        self.inner
    }
}

// ---------------------------------------------------------------------------
// Response wrapper types
// ---------------------------------------------------------------------------

/// Metadata about the Triton Inference Server.
#[derive(Debug, Clone)]
pub struct ServerMetadata {
    /// The server name.
    pub name: String,
    /// The server version.
    pub version: String,
    /// The protocol extensions supported by the server.
    pub extensions: Vec<String>,
}

/// Metadata about a specific model hosted on the server.
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    /// The model name.
    pub name: String,
    /// The available model versions.
    pub versions: Vec<String>,
    /// The model platform (e.g. `"tensorrt_plan"`, `"onnxruntime_onnx"`).
    pub platform: String,
    /// Input tensor metadata.
    pub inputs: Vec<TensorMetadata>,
    /// Output tensor metadata.
    pub outputs: Vec<TensorMetadata>,
}

/// Metadata for a single tensor (input or output).
#[derive(Debug, Clone)]
pub struct TensorMetadata {
    /// The tensor name.
    pub name: String,
    /// The tensor data type as a string (e.g. `"FP32"`, `"INT64"`).
    pub datatype: String,
    /// The tensor shape. Variable-size dimensions are represented as `-1`.
    pub shape: Vec<i64>,
}

/// An entry in the model repository index.
#[derive(Debug, Clone)]
pub struct ModelIndex {
    /// The model name.
    pub name: String,
    /// The model version.
    pub version: String,
    /// The model state (e.g. `"READY"`, `"UNAVAILABLE"`).
    pub state: String,
    /// The reason for the current state, if any.
    pub reason: String,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn data_type_round_trip() {
        let types = [
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
        for dt in &types {
            let s = dt.as_str();
            let parsed = DataType::parse(s).unwrap();
            assert_eq!(*dt, parsed, "Round-trip failed for {s}");
        }
    }

    #[test]
    fn data_type_display() {
        assert_eq!(format!("{}", DataType::Fp32), "FP32");
        assert_eq!(format!("{}", DataType::Int64), "INT64");
    }

    #[test]
    fn data_type_unknown_returns_none() {
        assert!(DataType::parse("UNKNOWN").is_none());
        assert!(DataType::parse("").is_none());
    }

    #[test]
    fn infer_input_with_f32_data() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let input = InferInput::new("input0", vec![1, 4], DataType::Fp32).with_data_f32(&data);

        assert_eq!(input.name(), "input0");
        assert_eq!(input.shape(), &[1, 4]);
        assert_eq!(input.datatype(), DataType::Fp32);

        let raw = input.data.as_ref().unwrap();
        assert_eq!(raw.len(), 16); // 4 floats * 4 bytes
    }

    #[test]
    fn infer_input_with_i64_data() {
        let data = vec![10i64, 20, 30];
        let input = InferInput::new("ids", vec![1, 3], DataType::Int64).with_data_i64(&data);

        let raw = input.data.as_ref().unwrap();
        assert_eq!(raw.len(), 24); // 3 ints * 8 bytes
    }

    #[test]
    fn infer_input_with_bool_data() {
        let data = vec![true, false, true];
        let input = InferInput::new("mask", vec![1, 3], DataType::Bool).with_data_bool(&data);

        let raw = input.data.as_ref().unwrap();
        assert_eq!(raw, &[1, 0, 1]);
    }

    #[test]
    fn infer_input_with_u8_data() {
        let data: Vec<u8> = vec![0, 127, 255];
        let input = InferInput::new("quantized", vec![1, 3], DataType::Uint8).with_data_u8(&data);

        let raw = input.data.as_ref().unwrap();
        assert_eq!(raw, &[0, 127, 255]);
    }

    #[test]
    fn infer_input_with_i8_data() {
        let data: Vec<i8> = vec![-128, 0, 127];
        let input = InferInput::new("quantized", vec![1, 3], DataType::Int8).with_data_i8(&data);

        let raw = input.data.as_ref().unwrap();
        assert_eq!(raw.len(), 3);
    }

    #[test]
    fn infer_input_with_u16_data() {
        let data: Vec<u16> = vec![0, 1000, 65535];
        let input = InferInput::new("input", vec![1, 3], DataType::Uint16).with_data_u16(&data);

        let raw = input.data.as_ref().unwrap();
        assert_eq!(raw.len(), 6); // 3 u16 * 2 bytes
    }

    #[test]
    fn infer_input_with_i16_data() {
        let data: Vec<i16> = vec![-32768, 0, 32767];
        let input = InferInput::new("input", vec![1, 3], DataType::Int16).with_data_i16(&data);

        let raw = input.data.as_ref().unwrap();
        assert_eq!(raw.len(), 6); // 3 i16 * 2 bytes
    }

    #[test]
    fn infer_input_with_bytes_data() {
        let strings: Vec<&[u8]> = vec![b"hello", b"world"];
        let input = InferInput::new("text", vec![1, 2], DataType::Bytes).with_data_bytes(&strings);

        let raw = input.data.as_ref().unwrap();
        // "hello" = 4-byte length (5) + 5 bytes = 9 bytes
        // "world" = 4-byte length (5) + 5 bytes = 9 bytes
        assert_eq!(raw.len(), 18);
        // Check length prefix for first string
        assert_eq!(&raw[..4], &5u32.to_le_bytes());
    }

    #[test]
    fn infer_input_with_parameters() {
        let input = InferInput::new("input0", vec![1, 4], DataType::Fp32)
            .with_string_parameter("key", "value")
            .with_int_parameter("count", 42)
            .with_bool_parameter("flag", true);

        let (tensor, _) = input.into_proto();
        assert_eq!(tensor.parameters.len(), 3);
    }

    #[test]
    fn infer_request_builder_basic() {
        let input = InferInput::new("input0", vec![1, 4], DataType::Fp32).with_data_f32(&[1.0; 4]);

        let request = InferRequestBuilder::new("my_model")
            .model_version("1")
            .request_id("test-001")
            .input(input)
            .output("output0")
            .build();

        assert_eq!(request.model_name, "my_model");
        assert_eq!(request.model_version, "1");
        assert_eq!(request.id, "test-001");
        assert_eq!(request.inputs.len(), 1);
        assert_eq!(request.outputs.len(), 1);
        assert_eq!(request.raw_input_contents.len(), 1);
        assert_eq!(request.raw_input_contents[0].len(), 16); // 4 * f32
    }

    #[test]
    fn infer_request_builder_multiple_inputs() {
        let input1 = InferInput::new("input0", vec![1, 4], DataType::Fp32).with_data_f32(&[1.0; 4]);
        let input2 = InferInput::new("input1", vec![1, 2], DataType::Int64).with_data_i64(&[1, 2]);

        let request = InferRequestBuilder::new("multi_input_model")
            .inputs(vec![input1, input2])
            .output("output0")
            .output("output1")
            .build();

        assert_eq!(request.inputs.len(), 2);
        assert_eq!(request.outputs.len(), 2);
        assert_eq!(request.raw_input_contents.len(), 2);
    }

    #[test]
    fn infer_request_builder_with_parameters() {
        let request = InferRequestBuilder::new("model")
            .string_parameter("sequence_id", "abc")
            .int_parameter("priority", 1)
            .bool_parameter("sequence_start", true)
            .build();

        assert_eq!(request.parameters.len(), 3);
    }

    #[test]
    fn infer_request_builder_no_data() {
        let input = InferInput::new("input0", vec![1, 4], DataType::Fp32);

        let request = InferRequestBuilder::new("model").input(input).build();

        // raw_input_contents should have an empty entry.
        assert_eq!(request.raw_input_contents.len(), 1);
        assert!(request.raw_input_contents[0].is_empty());
    }

    #[test]
    fn infer_request_builder_default_version() {
        let request = InferRequestBuilder::new("model").build();
        assert!(request.model_version.is_empty());
        assert!(request.id.is_empty());
    }

    #[test]
    fn infer_response_accessors() {
        let raw_data: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        let proto_response = inference::ModelInferResponse {
            model_name: "test_model".into(),
            model_version: "1".into(),
            id: "req-001".into(),
            parameters: Default::default(),
            outputs: vec![inference::model_infer_response::InferOutputTensor {
                name: "output0".into(),
                datatype: "FP32".into(),
                shape: vec![1, 4],
                parameters: Default::default(),
                contents: None,
            }],
            raw_output_contents: vec![raw_data],
        };

        let response = InferResponse::new(proto_response);

        assert_eq!(response.model_name(), "test_model");
        assert_eq!(response.model_version(), "1");
        assert_eq!(response.id(), "req-001");
        assert_eq!(response.outputs().len(), 1);
        assert!(response.output("output0").is_some());
        assert!(response.output("nonexistent").is_none());

        let f32_data = response.output_as_f32(0).unwrap();
        assert_eq!(f32_data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn infer_response_output_as_i32() {
        let raw_data: Vec<u8> = [10i32, 20, 30]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        let proto_response = inference::ModelInferResponse {
            model_name: String::new(),
            model_version: String::new(),
            id: String::new(),
            parameters: Default::default(),
            outputs: vec![],
            raw_output_contents: vec![raw_data],
        };

        let response = InferResponse::new(proto_response);
        let data = response.output_as_i32(0).unwrap();
        assert_eq!(data, vec![10, 20, 30]);
    }

    #[test]
    fn infer_response_output_as_i64() {
        let raw_data: Vec<u8> = [100i64, 200].iter().flat_map(|v| v.to_le_bytes()).collect();

        let proto_response = inference::ModelInferResponse {
            model_name: String::new(),
            model_version: String::new(),
            id: String::new(),
            parameters: Default::default(),
            outputs: vec![],
            raw_output_contents: vec![raw_data],
        };

        let response = InferResponse::new(proto_response);
        let data = response.output_as_i64(0).unwrap();
        assert_eq!(data, vec![100, 200]);
    }

    #[test]
    fn infer_response_output_out_of_bounds() {
        let proto_response = inference::ModelInferResponse {
            model_name: String::new(),
            model_version: String::new(),
            id: String::new(),
            parameters: Default::default(),
            outputs: vec![],
            raw_output_contents: vec![],
        };

        let response = InferResponse::new(proto_response);
        assert!(response.output_as_f32(0).is_err());
        assert!(response.output_as_raw(0).is_err());
    }

    #[test]
    fn infer_response_invalid_byte_length() {
        let proto_response = inference::ModelInferResponse {
            model_name: String::new(),
            model_version: String::new(),
            id: String::new(),
            parameters: Default::default(),
            outputs: vec![],
            raw_output_contents: vec![vec![0, 1, 2]], // 3 bytes, not a multiple of 4
        };

        let response = InferResponse::new(proto_response);
        assert!(response.output_as_f32(0).is_err());
        assert!(response.output_as_i32(0).is_err());
    }

    #[test]
    fn infer_response_into_inner() {
        let proto_response = inference::ModelInferResponse {
            model_name: "model".into(),
            model_version: String::new(),
            id: String::new(),
            parameters: Default::default(),
            outputs: vec![],
            raw_output_contents: vec![],
        };

        let response = InferResponse::new(proto_response);
        let inner = response.into_inner();
        assert_eq!(inner.model_name, "model");
    }

    #[test]
    fn requested_output_with_parameters() {
        let output =
            InferRequestedOutput::new("output0").with_string_parameter("classification", "3");

        assert_eq!(output.name(), "output0");

        let proto = output.into_proto();
        assert_eq!(proto.parameters.len(), 1);
    }
}
