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

import json

import subprocess
from pathlib import Path

# Clean up
export_file = Path('profile_export.json')
export_file.unlink(missing_ok=True)

with open('prompts.json', 'w') as f:
    json.dump({
        'data': [
            {
                'PROMPT': [ "Hello, my name is " ],
                "STREAM": [ True ],
            }
        ],
    }, f)

ret = subprocess.run(args=['perf_analyzer -m vllm -i grpc --async --streaming --input-data=prompts.json --profile-export-file=profile_export.json --measurement-mode=count_windows --measurement-request-count=10 --stability-percentage=999'], shell=True)

if ret.returncode == 0:
    with open("profile_export.json") as f:
        # example json demonstrating format:
        # https://github.com/triton-inference-server/client/blob/main/src/c%2B%2B/perf_analyzer/docs/examples/decoupled_output_file.json
        requests = json.load(f)["experiments"][0]["requests"]
        latencies = [r["response_timestamps"][0] - r["timestamp"] for r in requests]
        avg_latency_s = sum(latencies) / len(latencies) / 1000000000
    
        print("Average first-token latency: " + str(avg_latency_s) + " sec")
