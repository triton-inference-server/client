#!/usr/bin/env python
# Copyright 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import numpy as np
import sys

import tritonclient.grpc as grpcclient

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    parser.add_argument('-u',
                        '--url',
                        type=str,
                        required=False,
                        default='localhost:8001',
                        help='Inference server URL. Default is localhost:8001.')

    FLAGS = parser.parse_args()
    try:
        # Example of setting custom GRPC ChannelArguments. Users can set any
        # channel_args here at your own risk, Triton will assume they are
        # correct and complete.
        channel_args = [
            # Define custom max message sizes: 1MB here is an arbitrary example.
            ('grpc.max_send_message_length', 1024 * 1024),
            ('grpc.max_receive_message_length', 1024 * 1024),
            # Example of setting KeepAlive options through generic channel_args
            ('grpc.keepalive_time_ms', 2**31 - 1),
            ('grpc.keepalive_timeout_ms', 20000),
            ('grpc.keepalive_permit_without_calls', False),
            ('grpc.http2.max_pings_without_data', 2),
            # Example arg requested for the feature
            ('grpc.dns_enable_srv_queries', 1)
        ]

        triton_client = grpcclient.InferenceServerClient(
            url=FLAGS.url, verbose=FLAGS.verbose, channel_args=channel_args)
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit()

    model_name = "simple"

    # Infer
    inputs = []
    outputs = []
    inputs.append(grpcclient.InferInput('INPUT0', [1, 16], "INT32"))
    inputs.append(grpcclient.InferInput('INPUT1', [1, 16], "INT32"))

    # Create the data for the two input tensors. Initialize the first
    # to unique integers and the second to all ones.
    input0_data = np.arange(start=0, stop=16, dtype=np.int32)
    input0_data = np.expand_dims(input0_data, axis=0)
    input1_data = np.ones(shape=(1, 16), dtype=np.int32)

    # Initialize the data
    inputs[0].set_data_from_numpy(input0_data)
    inputs[1].set_data_from_numpy(input1_data)

    outputs.append(grpcclient.InferRequestedOutput('OUTPUT0'))
    outputs.append(grpcclient.InferRequestedOutput('OUTPUT1'))

    # Test with outputs
    results = triton_client.infer(model_name=model_name,
                                  inputs=inputs,
                                  outputs=outputs,
                                  headers={'test': '1'})

    # Get the output arrays from the results
    output0_data = results.as_numpy('OUTPUT0')
    output1_data = results.as_numpy('OUTPUT1')

    for i in range(16):
        print(
            str(input0_data[0][i]) + " + " + str(input1_data[0][i]) + " = " +
            str(output0_data[0][i]))
        print(
            str(input0_data[0][i]) + " - " + str(input1_data[0][i]) + " = " +
            str(output1_data[0][i]))
        if (input0_data[0][i] + input1_data[0][i]) != output0_data[0][i]:
            print("sync infer error: incorrect sum")
            sys.exit(1)
        if (input0_data[0][i] - input1_data[0][i]) != output1_data[0][i]:
            print("sync infer error: incorrect difference")
            sys.exit(1)

    print('PASS: CustomArgs')
