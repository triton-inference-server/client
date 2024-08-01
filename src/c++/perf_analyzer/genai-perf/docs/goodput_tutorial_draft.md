<!--
Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
 * Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
 * Neither the name of NVIDIA CORPORATION nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
-->

# Tutorials

## Profile GPT2 running on Triton + vLLM <a id="triton-vllm"></a>

### Run GPT2 on Triton Inference Server using vLLM

<details>
<summary>See instructions</summary>

Run Triton Inference Server with vLLM backend container:

```bash
export RELEASE="yy.mm" # e.g. export RELEASE="24.06"


docker run -it --net=host --gpus=1 --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864 nvcr.io/nvidia/tritonserver:${RELEASE}-vllm-python-py3

# Install Triton CLI (~5 min):
pip install "git+https://github.com/triton-inference-server/triton_cli@0.0.8"

# Download model:
triton import -m gpt2 --backend vllm

# Run server:
triton start
```

</details>

### Run GenAI-Perf

Run GenAI-Perf from Triton Inference Server SDK container:

```bash
export RELEASE="yy.mm" # e.g. export RELEASE="24.06"

docker run -it --net=host --gpus=1 nvcr.io/nvidia/tritonserver:${RELEASE}-py3-sdk

# Run GenAI-Perf in the container:
genai-perf profile \
  -m gpt2 \
  --service-kind triton \
  --backend vllm \
  --num-prompts 100 \
  --random-seed 123 \
  --synthetic-input-tokens-mean 200 \
  --synthetic-input-tokens-stddev 0 \
  --streaming \
  --output-tokens-mean 100 \
  --output-tokens-stddev 0 \
  --output-tokens-mean-deterministic \
  --tokenizer hf-internal-testing/llama-tokenizer \
  --concurrency 1 \
  --measurement-interval 4000 \
  --profile-export-file my_profile_export.json \
  --url localhost:8001 \
  # Add TTFT and ITL requirements through CLI
  --goodput-constraints 8 1.6
```

Example output:

```
                                   LLM Metrics                                    
┏━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
┃                Statistic ┃    avg ┃    min ┃    max ┃    p99 ┃    p90 ┃    p75 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
│ Time to first token (ms) │   7.29 │   3.94 │  26.87 │  26.31 │  14.11 │   5.04 │
│ Inter token latency (ms) │   1.55 │   1.41 │   1.76 │   1.76 │   1.69 │   1.64 │
│     Request latency (ms) │ 179.21 │ 169.03 │ 206.93 │ 206.91 │ 198.91 │ 184.20 │
│   Output sequence length │ 112.19 │ 107.00 │ 120.00 │ 119.85 │ 118.50 │ 113.50 │
│    Input sequence length │ 200.00 │ 200.00 │ 200.00 │ 200.00 │ 200.00 │ 200.00 │
└──────────────────────────┴────────┴────────┴────────┴────────┴────────┴────────┘
Output token throughput (per sec): 1251.61
Request throughput (per sec): 11.16
Out of 16 requests, 10 are Good under the constraints of TTFT: 8.00ms, ITL: 1.60ms
Request goodput (per sec): 6.97
```