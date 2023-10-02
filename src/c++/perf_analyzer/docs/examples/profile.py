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
            avg_latency_in_sec = sum(latencies) / len(latencies) / 1_000_000_000
        return avg_latency_in_sec


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
    args = parser.parse_args()

    input_data = {"data": [{"STREAM": [True]}]}
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

        results.append((prompt_size, profile(args)))

    print("\n[ Summary: First-Token Latency ]")
    for prompt_size, latency in results:
        print(
            f"  Prompt size: {prompt_size} | Average first-token latency: {latency:.4f} sec"
        )
