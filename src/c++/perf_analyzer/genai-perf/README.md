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

GenAI-Perf is a command line tool for measuring the throughput and latency of
generative AI models as served through an inference server. For large language
models (LLMs), GenAI-Perf provides metrics such as
[output token throughput](#output_token_throughput_metric),
[time to first token](#time_to_first_token_metric),
[inter token latency](#inter_token_latency_metric), and
[request throughput](#request_throughput_metric). For a full list of metrics
please see the [Metrics section](#metrics).

Users specify a model name, an inference server URL, the type of inputs to use
(synthetic or from dataset), and the type of load to generate (number of
concurrent requests, request rate).

GenAI-Perf generates the specified load, measures the performance of the
inference server and reports the metrics in a simple table as console output.
The tool also logs all results in a csv file that can be used to derive
additional metrics and visualizations. The inference server must already be
running when GenAI-Perf is run.

> [!Note]
> GenAI-Perf is currently in early release and under rapid development. While we
> will try to remain consistent, command line options and functionality are
> subject to change as the tool matures.

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

<details>

<summary>To install from source:</summary>

## From Source

This method requires that Perf Analyzer is installed in your development
environment and that you have at least Python 3.10 installed. To build Perf
Analyzer from source, see
[here](../docs/install.md#build-from-source).

```bash
pip install "git+https://github.com/triton-inference-server/client.git@r24.03#subdirectory=src/c++/perf_analyzer/genai-perf"
```

Run GenAI-Perf:

```bash
genai-perf --help
```

</details>
</br>

# Quick Start

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

See [Tutorial](docs/tutorial.md) for additional examples.

# Model Inputs

GenAI-Perf supports model input prompts from either synthetically generated
inputs, or from the HuggingFace
[OpenOrca](https://huggingface.co/datasets/Open-Orca/OpenOrca) or
[CNN_DailyMail](https://huggingface.co/datasets/cnn_dailymail) datasets. This is
specified using the `--dataset` CLI option.

When the dataset is synthetic, you can specify the following options:
* `--num-of-output-prompts <int>`: The number of unique prompts to generate, >= 1.
* `--input-tokens-mean <int>`: The mean number of tokens of synthetic input data, >= 1.
* `--input-tokens-stddev <int>`: The standard deviation of the number of tokens of synthetic input data, >= 0.
* `--expected-output-tokens <int>`: The number of output tokens to ask the model
  to return in the response, >= 1.
* `--random-seed <int>`: The seed used to generate random values, >= 0.

When the dataset is coming from HuggingFace, you can specify the following
options:
* `--dataset {openorca,cnn_dailymail}`: HuggingFace dataset to use for benchmarking.
* `--num-of-output-prompts <int>`: The number of unique prompts to generate, >= 1.

# Metrics

GenAI-Perf collects a diverse set of metrics that captures the performance of
the inference server.

| Metric | Description | Aggregations |
| - | - | - |
| <span id="time_to_first_token_metric">Time to First Token</span> | Time between when a request is sent and when its first response is received, one value per request in benchmark | Avg, min, max, p99, p90, p75 |
| <span id="inter_token_latency_metric">Inter Token Latency</span> | Time between intermediate responses for a single request divided by the number of generated tokens of the latter response, one value per response per request in benchmark | Avg, min, max, p99, p90, p75 |
| Request Latency | Time between when a request is sent and when its final response is received, one value per request in benchmark | Avg, min, max, p99, p90, p75 |
| Number of Output Tokens | Total number of output tokens of a request, one value per request in benchmark | Avg, min, max, p99, p90, p75 |
| <span id="output_token_throughput_metric">Output Token Throughput</span> | Total number of output tokens from benchmark divided by benchmark duration | None–one value per benchmark |
| <span id="request_throughput_metric">Request Throughput</span> | Number of final responses from benchmark divided by benchmark duration | None–one value per benchmark |

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
