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
import random
import subprocess
from pathlib import Path

RANDOM_WORDS = [
    "system",
    "plug",
    "gentle",
    "efficient",
    "library",
    "tested",
    "careful",
    "sneeze",
    "excuse",
    "zoo",
    "rock",
    "delight",
    "hammer",
    "unit",
    "happen",
    "multiply",
    "texture",
    "tired",
    "knot",
    "yawn",
]


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
    if ret.returncode == 0:
        # Example json demonstrating format:
        #   see client/src/c++/perf_analyzer/docs/examples/decoupled_output_file.json
        with open("profile_export.json") as f:
            requests = json.load(f)["experiments"][0]["requests"]
            latencies = [r["response_timestamps"][0] - r["timestamp"] for r in requests]
            avg_latency_s = sum(latencies) / len(latencies) / 1_000_000_000
        return avg_latency_s


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="vllm",
        help="The name of the model to profile.",
    )
    args = parser.parse_args()

    prompt_lengths = [10, 100, 500, 800, 1000]
    input_data = {"data": [{"STREAM": [True]}]}
    results = []

    for prompt_length in prompt_lengths:
        # Generate random prompt
        prompt = random.choices(RANDOM_WORDS, k=prompt_length)
        input_data["data"][0]["PROMPT"] = [" ".join(prompt)]
        with open("prompts.json", "w") as f:
            json.dump(input_data, f)

        # Clean up
        export_file = Path("profile_export.json")
        export_file.unlink(missing_ok=True)

        results.append(profile(args))

    print("[ Summary: First-Token Latency ]")
    for prompt_length, latency in zip(prompt_lengths, results):
        print(
            f"- Prompt Length: {prompt_length} | Average first-token latency: {latency}"
        )
