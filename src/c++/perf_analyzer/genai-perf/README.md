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

# GenAI-Perf

GenAI-Perf is a command line tool for measuring the throughput and latency  of
generative AI models served through inference servers. For large language models
(LLMs) specifically, GenAI-Perf provides various metrics such as output token
throughput, time to first token, inter-token latency, request throughput, and
more.

GenAI-Perf requires that the generative AI model is already loaded in an
inference server and accessible over a network. Users specify a model name and
the inference server URL. Users can also specify if they want inputs to the
model to be synthetically generated or pulled from a dataset. Users can also
specify how they want the load to be generated (e.g. how many concurrent
requests are sent at a time or a particular request send rate).

> [!Note]
> Note: GenAI-Perf is currently in early release and under rapid development.
> While we will try to remain consistent, command line options are subject to
> change until the software hits 1.0.
> Known issues will also be documented as the tool matures.

# Installation

## Triton SDK Container

Available starting with the 24.03 release of the
[Triton Server SDK container](https://ngc.nvidia.com/catalog/containers/nvidia:tritonserver).

Run the Triton Inference Server SDK docker container:

```bash
docker run -it --net=host --gpus=all  nvcr.io/nvidia/tritonserver:24.03-py3-sdk
```

Run GenAI-Perf:

```bash
genai-perf --help
```

## From Source

This method requires that Perf Analyzer is installed in your development
environment.

```bash
pip install "git+https://github.com/triton-inference-server/client.git@r24.03#subdirectory=src/c++/perf_analyzer/genai-perf"
```

Run GenAI-Perf:

```bash
genai-perf --help
```

# Basic Usage

The server and model will already need to be running/accessible before running
GenAI-Perf.

After running GenAI-Perf, the results are printed to the command line and saved
in a `profile_export_genai_perf.csv` file.

## Triton Inference Server with TensorRT-LLM Backend

<details>
<summary>Here are example instructions on running the GPT2 model on Triton
Inference Server with TensorRT-LLM:</summary>

1. Run the Triton TensorRT-LLM container:

```bash
docker run -it --net=host --rm --gpus=all --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864 nvcr.io/nvidia/tritonserver:24.03-trtllm-python-py3
```

2. Install model dependencies (~5 min):

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
</br>

GenAI-Perf can be run from wherever it was installed. If it’s on the same
machine as the server container, you can run GenAI-Perf like this:

```bash
genai-perf -m gpt2 --service-kind triton --output-format trtllm --input-tokens-mean 200 --input-tokens-stddev 0 --streaming --concurrency 1
```

Example output:

```
                                              PA LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃                Statistic ┃        avg ┃        min ┃        max ┃        p99 ┃        p90 ┃        p75 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ Time to first token (ns) │ 11,182,408 │ 10,706,503 │ 13,875,938 │ 12,017,948 │ 11,878,155 │ 11,789,258 │
│ Inter token latency (ns) │  1,013,174 │    136,721 │  4,068,603 │  1,550,931 │  1,497,520 │  1,022,919 │
│     Request latency (ns) │ 53,280,063 │ 30,740,044 │ 59,664,925 │ 55,859,882 │ 53,987,990 │ 53,867,980 │
│         Num output token │         46 │         18 │         53 │         51 │         49 │         47 │
└──────────────────────────┴────────────┴────────────┴────────────┴────────────┴────────────┴────────────┘
Output token throughput (per sec): 859.02
Request throughput (per sec): 18.72
```

## Triton Inference Server with vLLM Backend

<details>
<summary>Here are example instructions on running the GPT2 model on Triton with
vLLM:</summary>

1. Run the Triton vLLM container:

```bash
docker run -it --net=host --rm --gpus=all --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864 nvcr.io/nvidia/tritonserver:24.03-vllm-python-py3
```

2. Install model dependencies (~5 min):

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
</br>

GenAI-Perf can be run from wherever it was installed. If it’s on the same
machine as the server container, you can run GenAI-Perf like this:

```bash
genai-perf -m gpt2 --service-kind triton --output-format vllm --input-tokens-mean 200 --input-tokens-stddev 0 --streaming --concurrency 1
```

Example output:

```
                                              PA LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃                Statistic ┃        avg ┃        min ┃        max ┃        p99 ┃        p90 ┃        p75 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ Time to first token (ns) │ 13,214,478 │ 10,764,205 │ 38,550,060 │ 22,862,108 │ 16,303,985 │ 13,836,136 │
│ Inter token latency (ns) │  2,936,136 │      3,869 │ 11,923,633 │  4,093,398 │  3,139,745 │  3,033,398 │
│     Request latency (ns) │ 57,227,234 │ 37,765,913 │ 88,951,099 │ 79,935,677 │ 64,973,767 │ 58,820,014 │
│         Num output token │         16 │          8 │         19 │         16 │         16 │         16 │
└──────────────────────────┴────────────┴────────────┴────────────┴────────────┴────────────┴────────────┘
Output token throughput (per sec): 279.15
Request throughput (per sec): 17.45
```

## OpenAI Chat Completions Compatible APIs

### OpenAI Chat Completions

<details>
<summary>Here are example instructions on running the GPT2 model on vLLM via its
<a href=https://platform.openai.com/docs/api-reference/chat>OpenAI-chat-completions-compatible API</a>:
</summary>

1. Run the vLLM OpenAI-compatible API server:

```bash
docker run -it --net=host --rm --gpus=all vllm/vllm-openai:latest --model gpt2 --dtype float16 --max-model-len 1024
```

</details>
</br>

GenAI-Perf can be run from wherever it was installed. If it’s on the same
machine as the server container, you can run GenAI-Perf like this:

```bash
genai-perf -m gpt2 --service-kind openai --endpoint v1/chat/completions --output-format openai_chat_completions --input-tokens-mean 200 --input-tokens-stddev 0 --streaming --concurrency 1
```

Example output:

```
                                                      PA LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃                Statistic ┃           avg ┃         min ┃           max ┃           p99 ┃           p90 ┃           p75 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ Time to first token (ns) │     8,608,119 │   8,301,517 │     9,927,047 │     9,655,734 │     8,729,382 │     8,628,921 │
│ Inter token latency (ns) │     2,408,936 │       3,805 │     5,064,721 │     2,795,042 │     2,507,276 │     2,472,543 │
│     Request latency (ns) │ 1,413,884,921 │ 182,823,305 │ 2,094,378,983 │ 2,090,798,466 │ 2,074,160,653 │ 2,068,209,175 │
│         Num output token │           583 │          77 │           861 │           860 │           853 │           847 │
└──────────────────────────┴───────────────┴─────────────┴───────────────┴───────────────┴───────────────┴───────────────┘
Output token throughput (per sec): 412.04
Request throughput (per sec): 0.71
```

### OpenAI Completions

<details>
<summary>Here are example instructions on running the GPT2 model on vLLM via its
<a href=https://platform.openai.com/docs/api-reference/completions>OpenAI-completions-compatible API</a>:
</summary>

1. Run the vLLM OpenAI-compatible API server:

```bash
docker run -it --net=host --rm --gpus=all vllm/vllm-openai:latest --model gpt2 --dtype float16 --max-model-len 1024
```

</details>
</br>

GenAI-Perf can be run from wherever it was installed. If it’s on the same
machine as the server container, you can run GenAI-Perf like this:

```bash
genai-perf -m gpt2 --service-kind openai --endpoint v1/completions --output-format openai_completions --input-tokens-mean 200 --input-tokens-stddev 0 --concurrency 1
```

Example output:

```
                                            PA LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃            Statistic ┃        avg ┃        min ┃        max ┃        p99 ┃        p90 ┃        p75 ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ Request latency (ns) │ 41,221,138 │ 17,522,610 │ 50,452,443 │ 47,291,315 │ 41,620,663 │ 41,396,763 │
│     Num output token │         16 │          4 │         17 │         16 │         16 │         16 │
└──────────────────────┴────────────┴────────────┴────────────┴────────────┴────────────┴────────────┘
Output token throughput (per sec): 384.39
Request throughput (per sec): 24.23
```

# Model Inputs

GenAI-Perf supports model input prompts from either synthetically generated
inputs, or from the HuggingFace
[OpenOrca](https://huggingface.co/datasets/Open-Orca/OpenOrca) or
[CNN_DailyMail](https://huggingface.co/datasets/cnn_dailymail) datasets. This is
specified using the `--dataset` CLI option.

When the dataset is synthetic you can specify the following options:
* `--num-of-output-prompts=<int>`: The number of unique prompts to generate, >= 1.
* `--input-tokens-mean=<int>`: The mean number of tokens of synthetic input data, >= 1.
* `--input-tokens-stddev=<int>`: The standard deviation number of tokens of synthetic input data, >= 0.
* `--synthetic-requested-output-tokens`: The number of output tokens to ask the model
  to return in the response, >= 1.
* `--random-seed=<int>`: The seed used to generate random values, >= 0.

When the dataset is coming from HuggingFace you can specify the following
options:
* `--dataset`: HuggingFace dataset to use for benchmarking.
* `--num-of-output-prompts`: The number of synthetic output prompts to generate.

# Metrics

GenAI-Perf collects a diverse set of metrics that captures the performance of
the inference server.

| Metric | Description | Aggregations |
| - | - | - |
| Time to First Token | Time between when a request is sent and when its first response is received, one value per request in benchmark | Avg, min, max, p99, p90, p75 |
| Inter Token Latency | Time between intermediate responses for a single request divided by the number of generated tokens of the latter response, one value per response per request in benchmark | Avg, min, max, p99, p90, p75 |
| Request Latency | Time between when a request is sent and when its final response is received, one value per request in benchmark | Avg, min, max, p99, p90, p75 |
| Number of Output Tokens | Total number of output tokens of a request, one value per request in benchmark | Avg, min, max, p99, p90, p75 |
| Output Token Throughput | Total number of output tokens from benchmark divided by benchmark duration | None–one value per benchmark |
| Request Throughput | Number of final responses from benchmark divided by the benchmark duration | None–one value per benchmark |

# CLI

##### `-h`
##### `--help`

Show the help message and exit.

##### `-v`
##### `--verbose`

Enables verbose mode.

##### `--version`

Prints the version and exits.

##### `--expected-output-tokens <int>`

The number of tokens to expect in the output. This is used to determine the
length of the prompt. The prompt will be generated such that the output will be
approximately this many tokens.

##### `--input-type {url,file,synthetic}`

The source of the input data.

##### `--input-tokens-mean <int>`

The mean of the number of tokens of synthetic input data.

##### `--input-tokens-stddev <int>`

The standard deviation of number of tokens of synthetic input data.

##### `-m <str>`
##### `--model <str>`

The name of the model to benchmark.

##### `--num-of-output-prompts <int>`

The number of synthetic output prompts to generate

##### `--output-format {openai_chat_completions,openai_completions,trtllm,vllm}`

The format of the data sent to triton.

##### `--random-seed <int>`
Seed used to generate random values

##### `--concurrency <int>`
Sets the concurrency value to benchmark.

##### `--input-data <file>`
Path to the input data json file that contains the list of requests.

##### `-p <int>`
##### `--measurement-interval <int>`

Indicates the time interval used for each measurement in milliseconds. The perf
analyzer will sample a time interval specified by -p and take measurement over
the requests completed within that time interval. The default value is `5000`
msec.

##### `--profile-export-file <file>`

Specifies the path where the perf_analyzer profile export will be generated. By
default, the profile export will be to `profile_export.json`. The genai-perf
file will be exported to `<profile_export_file>_genai_perf.csv`. For example,
if the profile export file is `profile_export.json`, the genai-perf file will be
exported to `profile_export_genai_perf.csv`.

##### `--request-rate <float>`

Sets the request rate for the load generated by PA.

##### `--service-kind {triton,openai}`

Describes the kind of service perf_analyzer will generate load for. The options
are `triton` and `openai`. Note in order to use `openai` you must specify an
endpoint via `--endpoint`. The default value is `triton`.

##### `-s <float>`
##### `--stability-percentage <float>`

Indicates the allowed variation in latency measurements when determining if a
result is stable. The measurement is considered as stable if the ratio of max /
min from the recent 3 measurements is within (stability percentage) in terms of
both infer per second and latency.

##### `--streaming`

Enables the use of the streaming API.

##### `--endpoint {v1/completions,v1/chat/completions}`

Describes what endpoint to send requests to on the server. This is required when
using `openai` service-kind. This is ignored in other cases.

##### `-u <url>`
##### `--url <url>`

URL of the endpoint to target for benchmarking.

##### `--dataset {openorca,cnn_dailymail}`

HuggingFace dataset to use for benchmarking.

# Known Issues

* GenAI-Perf can be slow to finish if a high request-rate is provided
* Token counts may not be exact
* Token output counts are much higher than reality for now when running on
triton server, because the input is reflected back into the output
