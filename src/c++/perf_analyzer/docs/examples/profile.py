# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import json
import subprocess
from pathlib import Path


def calculate_avg_latencies():
    # Example json demonstrating format:
    #   see client/src/c++/perf_analyzer/docs/examples/decoupled_output_file.json
    first_token_latencies = []
    token_to_token_latencies = []
    with open("profile_export.json") as f:
        requests = json.load(f)["experiments"][0]["requests"]
        for request in requests:
            prev_response = request["response_timestamps"][0]
            first_token_latencies.append(prev_response - request["timestamp"])
            for response in request["response_timestamps"][1:]:
                token_to_token_latencies.append(response - prev_response)
                prev_response = response

    avg_first_token_latency = (
        sum(first_token_latencies) / len(first_token_latencies) / 1_000_000_000
    )
    avg_token_to_token_latency = (
        sum(token_to_token_latencies) / len(token_to_token_latencies) / 1_000_000_000
    )
    return avg_first_token_latency, avg_token_to_token_latency


def profile(args):
    command = (
        f"perf_analyzer -m {args.model} -i grpc --async --streaming "
        "--input-data=prompts.json "
        "--profile-export-file=profile_export.json "
        "--measurement-mode=count_windows "
        "--measurement-request-count=10 "
        "--stability-percentage=999"
    )
    ret = subprocess.run(args=[command], shell=True)
    ret.check_returncode()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="vllm",
        choices=["vllm"],
        help="The name of the model to profile.",
    )
    parser.add_argument(
        "--prompt-size-range",
        type=int,
        nargs=3,
        metavar=("START", "END", "STEP"),
        default=[10, 10, 1],
        help="The range of prompt sizes '<[START, END], STEP>' where END is inclusive.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="The maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Whether to ignore end-of-sequence token.",
    )
    args = parser.parse_args()

    request_parameters = f"""
    {{
        "max_tokens": {args.max_tokens},
        "ignore_eos": {"true" if args.ignore_eos else "false"}
    }}
    """
    input_data = {"data": [{"STREAM": [True]}]}
    input_data["data"][0]["SAMPLING_PARAMETERS"] = [request_parameters]

    results = []

    start, end, step = args.prompt_size_range
    for prompt_size in range(start, end + 1, step):
        prompt = ["hi"] * prompt_size  # Generate dummy prompt
        input_data["data"][0]["PROMPT"] = [" ".join(prompt)]
        with open("prompts.json", "w") as f:
            json.dump(input_data, f)

        # Clean up
        export_file = Path("profile_export.json")
        export_file.unlink(missing_ok=True)

        profile(args)
        avg_first_token_latency, avg_token_to_token_latency = calculate_avg_latencies()
        results.append(
            (prompt_size, avg_first_token_latency, avg_token_to_token_latency)
        )

    print("\n[ Benchmark Summary ]")
    for prompt_size, avg_first_token_latency, avg_token_to_token_latency in results:
        print(
            f"  Prompt size: {prompt_size}, "
            f"Average first-token latency: {avg_first_token_latency:.4f} sec, "
            f"Average token-token latency: {avg_token_to_token_latency:.4f} sec"
        )
