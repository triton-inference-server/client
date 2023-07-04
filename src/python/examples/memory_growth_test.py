#!/usr/bin/env python3
# Copyright 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from functools import partial

import numpy as np

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
        "-i",
        "--protocol",
        type=str,
        required=False,
        default="HTTP",
        help="Protocol (HTTP/gRPC) used to communicate with "
        + "the inference service. Default is HTTP.",
    )
    parser.add_argument(
        "-r",
        "--repetitions",
        type=int,
        required=False,
        default=100,
        help="Number of inferences to run. Default is 100.",
    )

    FLAGS = parser.parse_args()

    model_name = "custom_identity_int32"

    # Create the data for the input tensor.
    input0_data = np.arange(start=0, stop=16, dtype=np.int32)
    input0_data = np.expand_dims(input0_data, axis=0)

    if FLAGS.protocol.lower() == "grpc":
        import tritonclient.grpc as grpcclient

        create_client = partial(
            grpcclient.InferenceServerClient,
            url="localhost:8001",
            verbose=FLAGS.verbose,
        )
        create_input = partial(grpcclient.InferInput)
        create_output = partial(grpcclient.InferRequestedOutput)
    else:
        import tritonclient.http as httpclient

        create_client = partial(
            httpclient.InferenceServerClient,
            url="localhost:8000",
            verbose=FLAGS.verbose,
        )
        create_input = partial(httpclient.InferInput)
        create_output = partial(httpclient.InferRequestedOutput)

    for i in range(FLAGS.repetitions):
        triton_client = create_client()
        # Infer
        inputs = []
        outputs = []
        inputs.append(create_input("INPUT0", [1, 16], "INT32"))
        inputs[0].set_data_from_numpy(input0_data)
        outputs.append(create_output("OUTPUT0"))

        results = triton_client.infer(
            model_name=model_name, inputs=inputs, outputs=outputs
        )

        # Get the output arrays from the results and verify
        output0_data = results.as_numpy("OUTPUT0")

        if (
            (output0_data.dtype != input0_data.dtype)
            or (output0_data.shape != input0_data.shape)
            or not (np.array_equal(output0_data, input0_data))
        ):
            print("incorrect output")
            sys.exit(1)
