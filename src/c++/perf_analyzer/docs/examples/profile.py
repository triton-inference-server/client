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
from statistics import mean

TEMP_INPUT_FILE = "temp_input_data.json"


def load_profile_data():
    with open("profile_export.json") as f:
        return json.load(f)


def collect_latencies(requests):
    # Example json demonstrating format:
    #   see client/src/c++/perf_analyzer/docs/examples/decoupled_output_file.json
    first_token_latencies = []
    token_to_token_latencies = []
    requests = requests["experiments"][0]["requests"]
    for request in requests:
        first_response, *remaining_responses, _ = request["response_timestamps"]
        first_token_latencies.append(first_response - request["timestamp"])
        prev_response = first_response
        for response in remaining_responses:
            token_to_token_latencies.append(response - prev_response)
            prev_response = response
    return first_token_latencies, token_to_token_latencies


def calculate_avg_latencies():
    requests = load_profile_data()
    first_token_latencies, token_to_token_latencies = collect_latencies(requests)

    # Compute mean and convert from nanosec to sec
    avg_first_token_latency = mean(first_token_latencies) / 1_000_000_000
    if token_to_token_latencies:
        avg_token_to_token_latency = mean(token_to_token_latencies) / 1_000_000_000
    else:
        avg_token_to_token_latency = None
    return avg_first_token_latency, avg_token_to_token_latency


def profile(args, input_data_file):
    # Clean up
    export_file = Path("profile_export.json")
    export_file.unlink(missing_ok=True)

    command = (
        f"perf_analyzer -m {args.model} -i grpc --async --streaming "
        f"--input-data={input_data_file} "
        "--profile-export-file=profile_export.json "
    )
    if args.periodic_concurrency_range:
        start, end, step = args.periodic_concurrency_range
        command += (
            f"--periodic-concurrency-range={start}:{end}:{step} "
            f"--request-period={args.request_period}"
        )
    else:
        command += (
            "--measurement-mode=count_windows "
            "--measurement-request-count=10 "
            "--stability-percentage=999"
        )
    subprocess.run(args=[command], shell=True)


def generate_input_data(args, prompt_size, filename):
    request_parameters = f"""
    {{
        "max_tokens": {args.max_tokens},
        "ignore_eos": {"true" if args.ignore_eos else "false"}
    }}
    """
    input_data = {"data": [{"STREAM": [True]}]}
    input_data["data"][0]["SAMPLING_PARAMETERS"] = [request_parameters]

    prompt = ["hi"] * prompt_size  # Generate dummy prompt
    input_data["data"][0]["PROMPT"] = [" ".join(prompt)]
    with open(filename, "w") as f:
        json.dump(input_data, f)


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
        "--periodic-concurrency-range",
        type=int,
        nargs=3,
        metavar=("START", "END", "STEP"),
        help="The range of concurrency level that periodically increases until it reaches END.",
    )
    parser.add_argument(
        "--request-period",
        type=int,
        default=10,
        help="The number of responses that each request must receive before launching new requests.",
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
    parser.add_argument(
        "--input-data",
        type=str,
        help="The input data file to be used for inference request.",
    )
    args = parser.parse_args()

    results = []

    if args.input_data:
        print(f"Using input data file '{args.input_data}' for inference request.\n")
        with open(args.input_data) as f:
            input_data = json.load(f)
            prompt_size = len(input_data["data"][0]["PROMPT"][0].split())
            args.prompt_size_range = [prompt_size, prompt_size, 1]

    start, end, step = args.prompt_size_range
    for prompt_size in range(start, end + 1, step):
        if not args.input_data:
            generate_input_data(args, prompt_size, TEMP_INPUT_FILE)

        profile(args, args.input_data if args.input_data else TEMP_INPUT_FILE)
        avg_first_token_latency, avg_token_to_token_latency = calculate_avg_latencies()
        results.append(
            (prompt_size, avg_first_token_latency, avg_token_to_token_latency)
        )

    print("\n[ Benchmark Summary ]")
    for prompt_size, avg_first_token_latency, avg_token_to_token_latency in results:
        line = (
            f"  Prompt size: {prompt_size}, "
            f"Average first-token latency: {avg_first_token_latency:.4f} sec"
        )
        line += (
            f", Average token-token latency: {avg_token_to_token_latency:.4f} sec"
            if avg_token_to_token_latency
            else ""
        )
        print(line)
