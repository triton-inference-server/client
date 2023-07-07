#!/usr/bin/env python
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

import argparse
import sys

import grpc
import numpy as np
from tritonclient import utils
from tritonclient.grpc import service_pb2, service_pb2_grpc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        required=False,
        default=False,
        help="Enable verbose output",
    )
    parser.add_argument(
        "-u",
        "--url",
        type=str,
        required=False,
        default="localhost:8001",
        help="Inference server URL. Default is localhost:8001.",
    )

    FLAGS = parser.parse_args()

    # We use a simple model that takes 2 input tensors of 16 integers
    # each and returns 2 output tensors of 16 integers each. One
    # output tensor is the element-wise sum of the inputs and one
    # output is the element-wise difference.
    model_name = "simple_string"
    model_version = ""
    batch_size = 1

    # Create gRPC stub for communicating with the server
    channel = grpc.insecure_channel(FLAGS.url)
    grpc_stub = service_pb2_grpc.GRPCInferenceServiceStub(channel)

    # Generate the request
    request = service_pb2.ModelInferRequest()
    request.model_name = model_name
    request.model_version = model_version

    # Populate the inputs in inference request
    input0 = service_pb2.ModelInferRequest().InferInputTensor()
    input0.name = "INPUT0"
    input0.datatype = "BYTES"
    input0.shape.extend([1, 16])
    for i in range(16):
        input0.contents.bytes_contents.append(("{}".format(i)).encode("utf-8"))

    input1 = service_pb2.ModelInferRequest().InferInputTensor()
    input1.name = "INPUT1"
    input1.datatype = "BYTES"
    input1.shape.extend([1, 16])
    for i in range(16):
        input1.contents.bytes_contents.append("1".encode("utf-8"))

    request.inputs.extend([input0, input1])

    # Populate the outputs in the inference request
    output0 = service_pb2.ModelInferRequest().InferRequestedOutputTensor()
    output0.name = "OUTPUT0"

    output1 = service_pb2.ModelInferRequest().InferRequestedOutputTensor()
    output1.name = "OUTPUT1"
    request.outputs.extend([output0, output1])

    response = grpc_stub.ModelInfer(request)

    # Deserialize the output raw tensor to numpy array for proper comparison
    output_results = []
    index = 0
    for output in response.outputs:
        shape = []
        for value in output.shape:
            shape.append(value)
        output_results.append(
            utils.deserialize_bytes_tensor(response.raw_output_contents[index])
        )
        output_results[-1] = np.resize(output_results[-1], shape)
        index += 1

    if len(output_results) != 2:
        print("expected two output results")
        sys.exit(1)

    for i in range(16):
        print("{} + 1 = {}".format(i, output_results[0][0][i]))
        print("{} - 1 = {}".format(i, output_results[1][0][i]))

        if (i + 1) != int(output_results[0][0][i]):
            print("explicit string infer error: incorrect sum")
            sys.exit(1)
        if (i - 1) != int(output_results[1][0][i]):
            print("explicit string infer error: incorrect difference")
            sys.exit(1)
    print("PASS: explicit string")
