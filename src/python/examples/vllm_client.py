#!/usr/bin/env python
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

import argparse
import asyncio
import queue
import sys
from os import system

import numpy as np
import tritonclient.grpc.aio as grpcclient
from tritonclient.utils import *


class vLLMClient:
    class UserData:
        def __init__(self):
            self._completed_requests = queue.Queue()

    def __init__(self, url, verbose=False):
        self.url = url
        self.verbose = verbose

    async def generate(
        self,
        input_prompts,
        results_file,
        model_name="vllm",
        temperature=0.1,
        top_p=0.95,
        iterations=1,
        offset=0,
        streaming_mode=False,
    ):
        stream = streaming_mode

        with open(input_prompts, "r") as file:
            print(f"Loading inputs from `{input_prompts}`...")
            prompts = file.readlines()

        results_dict = {}

        async with grpcclient.InferenceServerClient(
            url=self.url, verbose=self.verbose
        ) as triton_client:
            # Request iterator that yields the next request
            async def async_request_iterator():
                try:
                    for iter in range(iterations):
                        for i, prompt in enumerate(prompts):
                            prompt_id = offset + (len(prompts) * iter) + i
                            results_dict[str(prompt_id)] = []
                            yield self._create_request(
                                prompt,
                                stream,
                                prompt_id,
                                model_name,
                                temperature,
                                top_p,
                            )
                except Exception as error:
                    print(f"caught error in request iterator:  {error}")

            try:
                # Start streaming
                response_iterator = triton_client.stream_infer(
                    inputs_iterator=async_request_iterator(),
                    stream_timeout=None,
                )
                # Read response from the stream
                async for response in response_iterator:
                    result, error = response
                    if error:
                        print(f"Encountered error while processing: {error}")
                    else:
                        output = result.as_numpy("TEXT")
                        for i in output:
                            results_dict[result.get_response().id].append(i)

            except InferenceServerException as error:
                print(error)
                sys.exit(1)

        with open(results_file, "w") as file:
            for id in results_dict.keys():
                for result in results_dict[id]:
                    file.write(result.decode("utf-8"))
                    file.write("\n")
                file.write("\n=========\n\n")
            print(f"Storing results into `{results_file}`...")

        if self.verbose:
            print(f"\nContents of `{results_file}` ===>")
            system(f"cat {results_file}")

        print("PASS: vLLM example")

    def _create_request(
        self, prompt, stream, request_id, model_name, temperature, top_p
    ):
        inputs = []
        prompt_data = np.array([prompt.encode("utf-8")], dtype=np.object_)
        try:
            inputs.append(grpcclient.InferInput("PROMPT", [1], "BYTES"))
            inputs[-1].set_data_from_numpy(prompt_data)
        except Exception as e:
            print(f"Encountered an error {e}")

        stream_data = np.array([stream], dtype=bool)
        inputs.append(grpcclient.InferInput("STREAM", [1], "BOOL"))
        inputs[-1].set_data_from_numpy(stream_data)

        # Add requested outputs
        outputs = []
        outputs.append(grpcclient.InferRequestedOutput("TEXT"))

        # Issue the asynchronous sequence inference.
        return {
            "model_name": model_name,
            "inputs": inputs,
            "outputs": outputs,
            "request_id": str(request_id),
            "parameters": {"temperature": str(temperature), "top_p": str(top_p)},
        }


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
        help="Inference server URL and its gRPC port. Default is localhost:8001.",
    )
    parser.add_argument(
        "-t",
        "--stream-timeout",
        type=float,
        required=False,
        default=None,
        help="Stream timeout in seconds. Default is None.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        required=False,
        default=0,
        help="Add offset to request IDs used",
    )
    parser.add_argument(
        "--input-prompts",
        type=str,
        required=False,
        default="prompts.txt",
        help="Text file with input prompts",
    )
    parser.add_argument(
        "--results-file",
        type=str,
        required=False,
        default="results.txt",
        help="The file with output results",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        required=False,
        default=1,
        help="Number of iterations through the prompts file",
    )
    parser.add_argument(
        "-s",
        "--streaming-mode",
        action="store_true",
        required=False,
        default=False,
        help="Enable streaming mode",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Name of the model to use. Default is 'vllm'.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        required=False,
        default=0.1,
        help="Sampling temperature in range 0-1. Default is 0.1.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        required=False,
        default=0.95,
        help="Top-p value for sampling in range 0-1. Default is 0.95.",
    )
    FLAGS = parser.parse_args()

    client = vLLMClient(FLAGS.url, FLAGS.verbose)
    asyncio.run(
        client.generate(
            FLAGS.input_prompts,
            FLAGS.results_file,
            FLAGS.model_name,
            FLAGS.temperature,
            FLAGS.top_p,
            FLAGS.iterations,
            FLAGS.offset,
            FLAGS.streaming_mode,
        )
    )
