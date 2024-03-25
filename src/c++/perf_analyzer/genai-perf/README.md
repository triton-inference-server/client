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

A tool to facilitate benchmarking generative AI models leveraging NVIDIA’s
[performance analyzer tool](https://github.com/triton-inference-server/client/tree/main/src/c%2B%2B/perf_analyzer).

GenAI-Perf builds upon the performant stimulus generation of the performance
analyzer to easily benchmark LLMs. Multiple endpoints are currently supported.

The GenAI-Perf workflow enables a user to
* [Generate prompts](#model-inputs) using either
  * synthetic generated data
  * open orca or CNN daily mail datasets
* Transform the prompts to a format understood by the
[chosen endpoint](#basic-usage)
  * Triton Infer
  * OpenAI
* Use Performance Analyzer to drive stimulus
* Gather LLM relevant [metrics](#metrics)
* Generate reports

all from the [command line](#cli).

> [!Note]
> GenAI-Perf is currently in early release while under rapid development.
> While we will try to remain consistent, command line options are subject to
> change until the software hits 1.0. Known issues will also be documented as the
> tool matures.

# Installation

## Triton SDK Container

Available starting with the 24.03 release of the
[Triton Server SDK container](https://ngc.nvidia.com/catalog/containers/nvidia:tritonserver).

```bash
RELEASE="24.03"

docker run -it --net=host --gpus=all  nvcr.io/nvidia/tritonserver:${RELEASE}-py3-sdk

genai-perf --help
```

## From Source

This method requires that Perf Analyzer is installed in your development
environment.

```bash
RELEASE="24.03"

pip install "git+https://github.com/triton-inference-server/client.git@r${RELEASE}#egg=genai-perf&subdirectory=src/c++/perf_analyzer/genai-perf"

genai-perf --help
```

# Basic Usage

## Triton with TRT-LLM

```bash
genai-perf -m llama-2-7b --concurrency 1 --service-kind triton --backend trtllm
```

## Triton with vLLM

```bash
genai-perf -m llama-2-7b --concurrency 1 --service-kind triton --backend vllm
```

## OpenAI Chat Completions Compatible APIs

https://platform.openai.com/docs/api-reference/chat

```bash
genai-perf -m llama-2-7b --concurrency 1 --service-kind openai --endpoint v1/chat/completions
```

## OpenAI Completions Compatible APIs

https://platform.openai.com/docs/api-reference/completions

```bash
genai-perf -m llama-2-7b --concurrency 1 --service-kind openai --endpoint v1/completions
```

# Model Inputs
GenAI-Perf supports model input prompts from either synthetically generated inputs,
or from the HuggingFace OpenOrca or CNN_DailyMail datasets. This is specified
using the `--prompt-source` CLI option.

When the dataset is synthetic you can specify the following options:
* `--num-prompts`: The number of unique prompts to generate.
* `--synthetic-tokens-mean`: The mean number of tokens of synthetic input data.
* `--synthetic-tokens-stddev`: The standard deviation number of tokens of synthetic
  input data.
* `--synthetic-requested-output-tokens`: The number of output tokens to ask the model
  to return in the response.
* `--random-seed`: The seed used to generate random values.

When the dataset is coming from HuggingFace you can specify the following
options:
* `--num-prompts`: The number of unique prompts to generate.
* `--dataset`: HuggingFace dataset to use for benchmarking.

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

##### `-v`
##### `--verbose`

Enables verbose mode.

##### `--version`

Prints the version and exits.

##### `--prompt-source {dataset,synthetic}`

The source of the input prompts.

##### `--input-dataset {openorca,cnn_dailymail}`

The HuggingFace dataset to use for prompts when prompt-source is dataset.

##### `--synthetic-requested-output-tokens <int>`
The number of tokens to request in the output. This is used when prompt-source
is synthetic to tell the LLM how many output tokens to generate in each response.

##### `--synthetic-tokens-mean <int>`

The mean of the number of tokens of synthetic input data.

##### `--synthetic-tokens-stddev <int>`

The standard deviation of number of tokens of synthetic input data.

##### `-m <str>`
##### `--model <str>`

The name of the model to benchmark.

##### `--num-prompts <int>`

The number of unique prompts to generate as stimulus.

##### `--backend {trtllm,vllm}`

When using the "triton" service-kind, this is the backend of the model.

##### `--random-seed <int>`

Seed used to generate random values.

##### `--concurrency <int>`

Sets the concurrency value to benchmark.

##### `-p <int>`
##### `--measurement-interval <int>`

Indicates the time interval used for each measurement in milliseconds. The perf
analyzer will sample a time interval specified by -p and take measurement over
the requests completed within that time interval.

The default value is `10000`.

##### `--profile-export-file <file>`

Specifies the path where the perf_analyzer profile export will be generated. By
default, the profile export will be to profile_export.json. The genai-perf file
will be exported to profile_export_file>_genai_perf.csv. For example, if the
profile export file is profile_export.json, the genai-perf file will be exported
to profile_export_genai_perf.csv.

##### `--request-rate <float>`

Sets the request rate for the load generated by PA.

##### `--service-kind {triton,openai}`

Describes the kind of service perf_analyzer will generate load for. The options
are `triton` and `openai`. Note in order to use `openai` you must specify an
endpoint via `--endpoint`.

The default value is `triton`.

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


# Known Issues

* GenAI-Perf can be slow to finish if a high request-rate is provided
* Token counts may not be exact
* Token output counts are much higher than reality for now when running on
triton server, because the input is reflected back into the output
