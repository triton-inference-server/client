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

# Tutorial

## Measuring Throughput and Latency of GPT2 using Triton + TensorRT-LLM

### Running GPT2 on Triton Inference Server using TensorRT-LLM

<details>
<summary>See instructions</summary>

1. Run Triton Inference Server with TensorRT-LLM backend container:

```bash
docker run -it --net=host --rm --gpus=all --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864 nvcr.io/nvidia/tritonserver:24.03-trtllm-python-py3
```

2. Install Triton CLI (~5 min):

```bash
pip install \
  --extra-index-url https://pypi.nvidia.com \
  -U \
  psutil \
  "pynvml>=11.5.0" \
  torch==2.1.2 \
  tensorrt_llm==0.8.0 \
  "git+https://github.com/triton-inference-server/triton_cli@0.0.6"
```

3. Download model:

```bash
triton import -m gpt2 --backend tensorrtllm
```

4. Run server:

```bash
triton start
```

</details>

### Running GenAI-Perf

1. Run Triton Inference Server SDK container:

```bash
docker run -it --net=host --rm --gpus=all nvcr.io/nvidia/tritonserver:24.03-py3-sdk
```

2. Run GenAI-Perf:

```bash
genai-perf \
  -m gpt2 \
  --service-kind triton \
  --output-format trtllm \
  --input-type synthetic \
  --num-of-output-prompts 100 \
  --random-seed 123 \
  --input-tokens-mean 200 \
  --input-tokens-stddev 0 \
  --streaming \
  --expected-output-tokens 100 \
  --concurrency 1 \
  --measurement-interval 4000 \
  --profile-export-file my_profile_export.json \
  --url localhost:8001
```

Example output:

```
                                                  PA LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃                Statistic ┃         avg ┃         min ┃           max ┃         p99 ┃         p90 ┃         p75 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ Time to first token (ns) │  36,698,426 │  12,495,578 │   577,754,321 │ 432,195,755 │  17,522,939 │  16,898,316 │
│ Inter token latency (ns) │   2,118,245 │      21,789 │     6,136,205 │   3,991,345 │   3,027,332 │   2,081,644 │
│     Request latency (ns) │ 525,306,940 │ 478,378,632 │ 1,069,448,663 │ 932,868,331 │ 528,386,622 │ 514,478,700 │
│         Num output token │         231 │         222 │           243 │         242 │         237 │         235 │
└──────────────────────────┴─────────────┴─────────────┴───────────────┴─────────────┴─────────────┴─────────────┘
Output token throughput (per sec): 438.18
Request throughput (per sec): 1.90
```

## Measuring Throughput and Latency of GPT2 using Triton + vLLM

### Running GPT2 on Triton Inference Server using vLLM

<details>
<summary>See instructions</summary>

1. Run Triton Inference Server with vLLM backend container:

```bash
docker run -it --net=host --rm --gpus=all --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864 nvcr.io/nvidia/tritonserver:24.03-vllm-python-py3
```

2. Install Triton CLI (~5 min):

```bash
pip install "git+https://github.com/triton-inference-server/triton_cli@0.0.6"
```

3. Download model:

```bash
triton import -m gpt2 --backend vllm
```

4. Run server:

```bash
triton start
```

</details>

### Running GenAI-Perf

1. Run Triton Inference Server SDK container:

```bash
docker run -it --net=host --rm --gpus=all nvcr.io/nvidia/tritonserver:24.03-py3-sdk
```

2. Run GenAI-Perf:

```bash
genai-perf \
  -m gpt2 \
  --service-kind triton \
  --output-format vllm \
  --input-type synthetic \
  --num-of-output-prompts 100 \
  --random-seed 123 \
  --input-tokens-mean 200 \
  --input-tokens-stddev 0 \
  --streaming \
  --expected-output-tokens 100 \
  --concurrency 1 \
  --measurement-interval 4000 \
  --profile-export-file my_profile_export.json \
  --url localhost:8001
```

Example output:

```
                                              PA LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃                Statistic ┃        avg ┃        min ┃        max ┃        p99 ┃        p90 ┃        p75 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ Time to first token (ns) │ 21,596,964 │ 10,821,576 │ 37,461,056 │ 29,813,133 │ 27,020,274 │ 25,983,638 │
│ Inter token latency (ns) │  3,450,047 │    787,779 │  5,269,772 │  5,098,801 │  4,078,603 │  4,018,934 │
│     Request latency (ns) │ 73,249,884 │ 44,825,743 │ 97,904,321 │ 91,128,892 │ 86,933,826 │ 84,828,480 │
│         Num output token │         16 │          9 │         17 │         17 │         16 │         16 │
└──────────────────────────┴────────────┴────────────┴────────────┴────────────┴────────────┴────────────┘
Output token throughput (per sec): 217.84
Request throughput (per sec): 13.63
```

## Measuring Throughput and Latency of GPT2 using OpenAI API-Compatible Server

### OpenAI Chat Completions API

#### Running GPT2 on [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat)-compatible server

<details>
<summary>See instructions</summary>

1. Run the vLLM inference server:

```bash
docker run -it --net=host --rm --gpus=all vllm/vllm-openai:latest --model gpt2 --dtype float16 --max-model-len 1024
```

</details>

#### Running GenAI-Perf

1. Run Triton Inference Server SDK container:

```bash
docker run -it --net=host --rm --gpus=all nvcr.io/nvidia/tritonserver:24.03-py3-sdk
```

2. Run GenAI-Perf:

```bash
genai-perf \
  -m gpt2 \
  --service-kind openai \
  --endpoint v1/chat/completions \
  --output-format openai_chat_completions \
  --input-type synthetic \
  --num-of-output-prompts 100 \
  --random-seed 123 \
  --input-tokens-mean 200 \
  --input-tokens-stddev 0 \
  --streaming \
  --expected-output-tokens 100 \
  --concurrency 1 \
  --measurement-interval 4000 \
  --profile-export-file my_profile_export.json \
  --url localhost:8000
```

Example output:

```
                                                      PA LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃                Statistic ┃           avg ┃         min ┃           max ┃           p99 ┃           p90 ┃           p75 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ Time to first token (ns) │    16,840,986 │  14,290,564 │    29,198,314 │    28,364,666 │    20,861,831 │    15,144,878 │
│ Inter token latency (ns) │     3,086,799 │       5,865 │     6,311,792 │     4,029,362 │     3,206,890 │     3,121,300 │
│     Request latency (ns) │ 2,017,527,954 │ 592,028,473 │ 2,749,313,976 │ 2,744,787,724 │ 2,704,051,453 │ 2,662,217,510 │
│         Num output token │           648 │         197 │           866 │           865 │           856 │           847 │
└──────────────────────────┴───────────────┴─────────────┴───────────────┴───────────────┴───────────────┴───────────────┘
Output token throughput (per sec): 320.87
Request throughput (per sec): 0.50
```

### OpenAI Completions API

#### Running GPT2 on [OpenAI Completions API](https://platform.openai.com/docs/api-reference/completions)-compatible server

<details>
<summary>See instructions</summary>

1. Run the vLLM inference server:

```bash
docker run -it --net=host --rm --gpus=all vllm/vllm-openai:latest --model gpt2 --dtype float16 --max-model-len 1024
```

</details>

#### Running GenAI-Perf

1. Run Triton Inference Server SDK container:

```bash
docker run -it --net=host --rm --gpus=all nvcr.io/nvidia/tritonserver:24.03-py3-sdk
```

2. Run GenAI-Perf:

```bash
genai-perf \
  -m gpt2 \
  --service-kind openai \
  --endpoint v1/completions \
  --output-format openai_completions \
  --input-type synthetic \
  --num-of-output-prompts 100 \
  --random-seed 123 \
  --input-tokens-mean 200 \
  --input-tokens-stddev 0 \
  --expected-output-tokens 100 \
  --concurrency 1 \
  --measurement-interval 4000 \
  --profile-export-file my_profile_export.json \
  --url localhost:8000
```

Example output:

```
                                            PA LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃            Statistic ┃        avg ┃        min ┃        max ┃        p99 ┃        p90 ┃        p75 ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ Request latency (ns) │ 58,912,082 │ 47,955,979 │ 85,459,346 │ 75,050,351 │ 62,799,972 │ 58,673,666 │
│     Num output token │         16 │         12 │         17 │         17 │         16 │         16 │
└──────────────────────┴────────────┴────────────┴────────────┴────────────┴────────────┴────────────┘
Output token throughput (per sec): 269.98
Request throughput (per sec): 16.96
```
