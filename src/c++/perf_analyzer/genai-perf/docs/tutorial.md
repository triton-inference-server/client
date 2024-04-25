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
export RELEASE="mm.yy" # e.g. export RELEASE="24.03"

docker run -it --net=host --rm --gpus=all --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864 nvcr.io/nvidia/tritonserver:${RELEASE}-trtllm-python-py3
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
export RELEASE="mm.yy" # e.g. export RELEASE="24.03"

docker run -it --net=host --rm --gpus=all nvcr.io/nvidia/tritonserver:${RELEASE}-py3-sdk
```

2. Run GenAI-Perf:

```bash
genai-perf \
  -m gpt2 \
  --service-kind triton \
  --backend trtllm \
  --prompt-source synthetic \
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
  --url localhost:8001
```

Example output:

```
                                                  LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃                Statistic ┃         avg ┃         min ┃         max ┃         p99 ┃         p90 ┃         p75 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ Time to first token (ns) │  13,266,974 │  11,818,732 │  18,351,779 │  16,513,479 │  13,741,986 │  13,544,376 │
│ Inter token latency (ns) │   2,069,766 │      42,023 │  15,307,799 │   3,256,375 │   3,020,580 │   2,090,930 │
│     Request latency (ns) │ 223,532,625 │ 219,123,330 │ 241,004,192 │ 238,198,306 │ 229,676,183 │ 224,715,918 │
│         Num output token │         104 │         100 │         129 │         128 │         109 │         105 │
│          Num input token │         199 │         199 │         199 │         199 │         199 │         199 │
└──────────────────────────┴─────────────┴─────────────┴─────────────┴─────────────┴─────────────┴─────────────┘
Output token throughput (per sec): 460.42
Request throughput (per sec): 4.44
```

## Measuring Throughput and Latency of GPT2 using Triton + vLLM

### Running GPT2 on Triton Inference Server using vLLM

<details>
<summary>See instructions</summary>

1. Run Triton Inference Server with vLLM backend container:

```bash
export RELEASE="mm.yy" # e.g. export RELEASE="24.03"

docker run -it --net=host --rm --gpus=all --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864 nvcr.io/nvidia/tritonserver:${RELEASE}-vllm-python-py3
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
export RELEASE="mm.yy" # e.g. export RELEASE="24.03"

docker run -it --net=host --rm --gpus=all nvcr.io/nvidia/tritonserver:${RELEASE}-py3-sdk
```

2. Run GenAI-Perf:

```bash
genai-perf \
  -m gpt2 \
  --service-kind triton \
  --backend vllm \
  --prompt-source synthetic \
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
  --url localhost:8001
```

Example output:

```
                                                  LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃                Statistic ┃         avg ┃         min ┃         max ┃         p99 ┃         p90 ┃         p75 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ Time to first token (ns) │  15,786,560 │  11,437,189 │  49,550,549 │  40,129,652 │  21,248,091 │  17,824,695 │
│ Inter token latency (ns) │   3,543,380 │     591,898 │  10,013,690 │   6,152,260 │   5,039,278 │   4,060,982 │
│     Request latency (ns) │ 388,415,721 │ 312,552,612 │ 528,229,817 │ 518,189,390 │ 484,281,365 │ 459,417,637 │
│         Num output token │         113 │         105 │         123 │         122 │         119 │         115 │
│          Num input token │         199 │         199 │         199 │         199 │         199 │         199 │
└──────────────────────────┴─────────────┴─────────────┴─────────────┴─────────────┴─────────────┴─────────────┘
Output token throughput (per sec): 290.24
Request throughput (per sec): 2.57
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
  --endpoint-type chat \
  --prompt-source synthetic \
  --num-prompts 100 \
  --random-seed 123 \
  --synthetic-input-tokens-mean 200 \
  --synthetic-input-tokens-stddev 0 \
  --streaming \
  --output-tokens-mean 100 \
  --output-tokens-stddev 0 \
  --tokenizer hf-internal-testing/llama-tokenizer \
  --concurrency 1 \
  --measurement-interval 4000 \
  --profile-export-file my_profile_export.json \
  --url localhost:8000
```

Example output:

```
                                                  LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃                Statistic ┃         avg ┃         min ┃         max ┃         p99 ┃         p90 ┃         p75 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ Time to first token (ns) │  13,546,815 │   9,821,658 │  48,317,756 │  34,361,913 │  16,541,625 │  14,612,026 │
│ Inter token latency (ns) │   2,560,813 │     457,703 │   6,507,334 │   3,754,617 │   3,059,158 │   2,953,540 │
│     Request latency (ns) │ 283,597,027 │ 240,098,890 │ 361,730,568 │ 349,164,037 │ 323,279,761 │ 306,507,562 │
│         Num output token │         114 │         103 │         142 │         136 │         122 │         119 │
│          Num input token │         199 │         199 │         199 │         199 │         199 │         199 │
└──────────────────────────┴─────────────┴─────────────┴─────────────┴─────────────┴─────────────┴─────────────┘
Output token throughput (per sec): 401.62
Request throughput (per sec): 3.52
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
  --endpoint-type completions \
  --prompt-source synthetic \
  --num-prompts 100 \
  --random-seed 123 \
  --synthetic-input-tokens-mean 200 \
  --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean 100 \
  --output-tokens-stddev 0 \
  --tokenizer hf-internal-testing/llama-tokenizer \
  --concurrency 1 \
  --measurement-interval 4000 \
  --profile-export-file my_profile_export.json \
  --url localhost:8000
```

Example output:

```
                                                LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃            Statistic ┃         avg ┃        min ┃         max ┃         p99 ┃         p90 ┃         p75 ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ Request latency (ns) │ 296,990,497 │ 43,312,449 │ 332,788,242 │ 327,475,292 │ 317,392,767 │ 310,343,333 │
│     Num output token │         109 │         11 │         158 │         142 │         118 │         113 │
│      Num input token │           1 │          1 │           1 │           1 │           1 │           1 │
└──────────────────────┴─────────────┴────────────┴─────────────┴─────────────┴─────────────┴─────────────┘
Output token throughput (per sec): 366.78
Request throughput (per sec): 3.37
```
