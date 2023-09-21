#!/usr/bin/env python3

# Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import grpc
from tritonclient.grpc import service_pb2
from tritonclient.utils import *


def get_error_grpc(rpc_error):
    """Convert a gRPC error to an InferenceServerException.

    Parameters
    ----------
    rpc_error : grpc.RpcError
        The gRPC error

    Returns
    -------
    InferenceServerException
    """
    return InferenceServerException(
        msg=rpc_error.details(),
        status=str(rpc_error.code()),
        debug_details=rpc_error.debug_error_string(),
    )


def get_cancelled_error(msg=None):
    """Get InferenceServerException object for a cancelled RPC.

    Returns
    -------
    InferenceServerException
    """
    if not msg:
        msg = "Locally cancelled by application!"
    return InferenceServerException(msg=msg, status="StatusCode.CANCELLED")


def raise_error_grpc(rpc_error):
    """Raise an InferenceServerException from a gRPC error.

    Parameters
    ----------
    rpc_error : grpc.RpcError
        The gRPC error

    Raises
    -------
    InferenceServerException
    """
    raise get_error_grpc(rpc_error) from None


def _get_inference_request(
    model_name,
    inputs,
    model_version,
    request_id,
    outputs,
    sequence_id,
    sequence_start,
    sequence_end,
    priority,
    timeout,
    parameters,
):
    request = service_pb2.ModelInferRequest()
    request.model_name = model_name
    request.model_version = model_version
    if request_id != "":
        request.id = request_id
    for infer_input in inputs:
        request.inputs.extend([infer_input._get_tensor()])
        if infer_input._get_content() is not None:
            request.raw_input_contents.extend([infer_input._get_content()])
    if outputs is not None:
        for infer_output in outputs:
            request.outputs.extend([infer_output._get_tensor()])
    if sequence_id != 0 and sequence_id != "":
        if isinstance(sequence_id, str):
            request.parameters["sequence_id"].string_param = sequence_id
        else:
            request.parameters["sequence_id"].int64_param = sequence_id
        request.parameters["sequence_start"].bool_param = sequence_start
        request.parameters["sequence_end"].bool_param = sequence_end
    if priority != 0:
        request.parameters["priority"].uint64_param = priority
    if timeout is not None:
        request.parameters["timeout"].int64_param = timeout

    if parameters:
        for key, value in parameters.items():
            if (
                key == "sequence_id"
                or key == "sequence_start"
                or key == "sequence_end"
                or key == "priority"
                or key == "binary_data_output"
            ):
                raise_error(
                    f'Parameter "{key}" is a reserved parameter and cannot be specified.'
                )
            else:
                if isinstance(value, str):
                    request.parameters[key].string_param = value
                elif isinstance(value, bool):
                    request.parameters[key].bool_param = value
                elif isinstance(value, int):
                    request.parameters[key].int64_param = value
                else:
                    raise_error(
                        f'The parameter datatype "{type(value)}" for key "{key}" is not supported.'
                    )

    return request


def _grpc_compression_type(algorithm_str):
    if algorithm_str is None:
        return grpc.Compression.NoCompression
    elif algorithm_str.lower() == "deflate":
        return grpc.Compression.Deflate
    elif algorithm_str.lower() == "gzip":
        return grpc.Compression.Gzip

    print(
        "The provided client-side compression algorithm is not supported... "
        "using no compression"
    )
    return grpc.Compression.NoCompression
