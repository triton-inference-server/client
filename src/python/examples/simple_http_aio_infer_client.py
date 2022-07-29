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
import ssl
import asyncio

import tritonclient.http.aio as httpclient
from tritonclient.utils import InferenceServerException


async def test_infer(triton_client,
                     model_name,
                     input0_data,
                     input1_data,
                     headers=None,
                     request_compression_algorithm=None,
                     response_compression_algorithm=None):
    inputs = []
    outputs = []
    inputs.append(httpclient.InferInput('INPUT0', [1, 16], "INT32"))
    inputs.append(httpclient.InferInput('INPUT1', [1, 16], "INT32"))

    # Initialize the data
    inputs[0].set_data_from_numpy(input0_data, binary_data=False)
    inputs[1].set_data_from_numpy(input1_data, binary_data=True)

    outputs.append(httpclient.InferRequestedOutput('OUTPUT0', binary_data=True))
    outputs.append(httpclient.InferRequestedOutput('OUTPUT1',
                                                   binary_data=False))
    query_params = {'test_1': 1, 'test_2': 2}
    results = await triton_client.infer(
        model_name,
        inputs,
        outputs=outputs,
        query_params=query_params,
        headers=headers,
        request_compression_algorithm=request_compression_algorithm,
        response_compression_algorithm=response_compression_algorithm)

    return results


async def test_infer_no_outputs(triton_client,
                                model_name,
                                input0_data,
                                input1_data,
                                headers=None,
                                request_compression_algorithm=None,
                                response_compression_algorithm=None):
    inputs = []
    inputs.append(httpclient.InferInput('INPUT0', [1, 16], "INT32"))
    inputs.append(httpclient.InferInput('INPUT1', [1, 16], "INT32"))

    # Initialize the data
    inputs[0].set_data_from_numpy(input0_data, binary_data=False)
    inputs[1].set_data_from_numpy(input1_data, binary_data=True)

    query_params = {'test_1': 1, 'test_2': 2}
    results = await triton_client.infer(
        model_name,
        inputs,
        outputs=None,
        query_params=query_params,
        headers=headers,
        request_compression_algorithm=request_compression_algorithm,
        response_compression_algorithm=response_compression_algorithm)

    return results


def validate_result(results, input0_data, input1_data):
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


async def main(FLAGS):
    # Initialize triton_client
    try:
        if FLAGS.ssl:
            ssl_context = None  # default SSL check
            if FLAGS.insecure:
                ssl_context = False  # Skip SSL certification validation
            elif (FLAGS.key_file is not None and FLAGS.cert_file is not None) \
                    or FLAGS.ca_certs is not None:
                # Custom certification validation
                ssl_context = ssl.create_default_context()
                if FLAGS.key_file is not None and FLAGS.cert_file is not None:
                    ssl_context.load_cert_chain(certfile=FLAGS.cert_file,
                                                keyfile=FLAGS.key_file)
                if FLAGS.ca_certs is not None:
                    ssl_context.load_verify_locations(cafile=FLAGS.ca_certs)
            triton_client = httpclient.InferenceServerClient(
                url=FLAGS.url,
                verbose=FLAGS.verbose,
                ssl=True,
                ssl_context=ssl_context)
        else:
            triton_client = httpclient.InferenceServerClient(
                url=FLAGS.url, verbose=FLAGS.verbose)
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit(1)

    model_name = "simple"

    # Create the data for the two input tensors. Initialize the first
    # to unique integers and the second to all ones.
    input0_data = np.arange(start=0, stop=16, dtype=np.int32)
    input0_data = np.expand_dims(input0_data, axis=0)
    input1_data = np.full(shape=(1, 16), fill_value=-1, dtype=np.int32)

    if FLAGS.http_headers is not None:
        headers_dict = {
            l.split(':')[0]: l.split(':')[1] for l in FLAGS.http_headers
        }
    else:
        headers_dict = None

    # Infer with requested Outputs
    results = await test_infer(triton_client, model_name, input0_data,
                               input1_data, headers_dict,
                               FLAGS.request_compression_algorithm,
                               FLAGS.response_compression_algorithm)
    print(results.get_response())

    statistics = await triton_client.get_inference_statistics(
        model_name=model_name, headers=headers_dict)
    print(statistics)
    if len(statistics['model_stats']) != 1:
        print("FAILED: Inference Statistics")
        sys.exit(1)

    # Validate the results by comparing with precomputed values.
    validate_result(results, input0_data, input1_data)

    # Infer without requested Outputs
    results = await test_infer_no_outputs(triton_client, model_name,
                                          input0_data, input1_data,
                                          headers_dict,
                                          FLAGS.request_compression_algorithm,
                                          FLAGS.response_compression_algorithm)
    print(results.get_response())

    # Validate the results by comparing with precomputed values.
    validate_result(results, input0_data, input1_data)

    # Infer with incorrect model name
    try:
        (await test_infer(triton_client, "wrong_model_name",
                                     input0_data, input1_data)).get_response()
        print("expected error message for wrong model name")
        sys.exit(1)
    except InferenceServerException as ex:
        print(ex)
        if not (ex.message().startswith("Request for unknown model")):
            print("improper error message for wrong model name")
            sys.exit(1)

    # Infer in parallel
    result_1, result_2 = await asyncio.gather(
        test_infer(triton_client, model_name, input0_data, input1_data,
                   headers_dict, FLAGS.request_compression_algorithm,
                   FLAGS.response_compression_algorithm),
        test_infer_no_outputs(triton_client, model_name, input0_data,
                              input1_data, headers_dict,
                              FLAGS.request_compression_algorithm,
                              FLAGS.response_compression_algorithm))

    # Validate the results by comparing with precomputed values.
    print(result_1.get_response())
    validate_result(result_1, input0_data, input1_data)
    print(result_2.get_response())
    validate_result(result_2, input0_data, input1_data)

    await triton_client.close()

    print("PASS: infer")


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
                        default='localhost:8000',
                        help='Inference server URL. Default is localhost:8000.')
    parser.add_argument('-s',
                        '--ssl',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable encrypted link to the server using HTTPS')
    parser.add_argument(
        '--key-file',
        type=str,
        required=False,
        default=None,
        help='File holding client private key. Default is None.')
    parser.add_argument(
        '--cert-file',
        type=str,
        required=False,
        default=None,
        help='File holding client certificate. Default is None.')
    parser.add_argument('--ca-certs',
                        type=str,
                        required=False,
                        default=None,
                        help='File holding ca certificate. Default is None.')
    parser.add_argument(
        '--insecure',
        action="store_true",
        required=False,
        default=False,
        help=
        'Use no peer verification in SSL communications. Use with caution. Default is False.'
    )
    parser.add_argument(
        '-H',
        dest='http_headers',
        metavar="HTTP_HEADER",
        required=False,
        action='append',
        help='HTTP headers to add to inference server requests. ' +
        'Format is -H"Header:Value".')
    parser.add_argument(
        '--request-compression-algorithm',
        type=str,
        required=False,
        default=None,
        help=
        'The compression algorithm to be used when sending request body to server. Default is None.'
    )
    parser.add_argument(
        '--response-compression-algorithm',
        type=str,
        required=False,
        default=None,
        help=
        'The compression algorithm to be used when receiving response body from server. Default is None.'
    )
    FLAGS = parser.parse_args()
    asyncio.run(main(FLAGS))
