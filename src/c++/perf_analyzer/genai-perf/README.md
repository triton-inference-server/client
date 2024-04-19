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
genai-perf -m llama-2-7b --concurrency 1 --service-kind openai --endpoint-type chat
```

## OpenAI Completions Compatible APIs

https://platform.openai.com/docs/api-reference/completions

```bash
genai-perf -m llama-2-7b --concurrency 1 --service-kind openai --endpoint-type completions
```

> [!Note]
> GenAI-Perf uses Llama tokenizer as a default tokenizer to parse and calculate
> token metrics on the input prompts and output responses. Users can instead
> specify a custom huggingface tokenizer using `--tokenizer` command line option
> as well.


# Model Inputs
GenAI-Perf supports model input prompts from either synthetically generated inputs,
or from the HuggingFace OpenOrca or CNN_DailyMail datasets. This is specified
using the `--prompt-source` CLI option.

When the dataset is synthetic, you can specify the following options:
* `--num-prompts`: The number of unique prompts to generate.
* `--synthetic-input-tokens-mean`: The mean number of tokens of synthetic input data.
* `--synthetic-input-tokens-stddev`: The standard deviation of the number of tokens of synthetic
  input data.
* `--random-seed`: The seed used to generate random values.

When the dataset is coming from HuggingFace, you can specify the following
options:
* `--num-prompts`: The number of unique prompts to generate.
* `--dataset`: HuggingFace dataset to use for benchmarking.

For any dataset, you can specify the following options:
* `--output-tokens-mean`: The mean number of tokens to request from the model.
* `--output-tokens-mean-deterministic`: If an output token mean is supplied,
this will make the token count distribution set the max and min lengths equal.
This is supported for the Triton service-kind.
* `--output-tokens-stddev`: The standard deviation of the number of tokens to
request from the model. input data.

You can optionally set additional model inputs with the following option:
* `--extra-inputs {input_name}:{value}`: An additional input for use with the model with a singular value,
such as `stream:true` or `max_tokens:5`. This flag can be repeated to supply multiple extra inputs.


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

##### `--tokenizer <str>`

The HuggingFace tokenizer to use to interpret token metrics from prompts and
responses.

##### `--prompt-source {dataset,synthetic}`

The source of the input prompts.

##### `--input-dataset {openorca,cnn_dailymail}`

The HuggingFace dataset to use for prompts when prompt-source is dataset.

##### `--synthetic-input-tokens-mean <int>`

The mean of the number of tokens of synthetic input data.

##### `--synthetic-input-tokens-stddev <int>`

The standard deviation of the number of tokens of synthetic input data.

##### `--output-tokens-mean <int>`

The mean of the number of output tokens to request from the model.

##### `--output-tokens-mean-deterministic`

Sets the output token distribution to add a minimum token length input
(in addition to the maximum token length) to more deterministically
request the number of output tokens from the model.

##### `--output-tokens-stddev <int>`

The standard deviation of the number of output tokens to request from the model.

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

##### `--log-file <file>`

Specifies the path where a log file will be generated. By default, the log will
be genai_perf.log.

##### `--request-rate <float>`

Sets the request rate for the load generated by PA.

##### `--service-kind {triton,openai}`

Describes the kind of service perf_analyzer will generate load for. The options
are `triton` and `openai`. Note in order to use `openai` you must specify an
api via `--endpoint-type`.

The default value is `triton`.

##### `-s <float>`
##### `--stability-percentage <float>`

Indicates the allowed variation in latency measurements when determining if a
result is stable. The measurement is considered as stable if the ratio of max /
min from the recent 3 measurements is within (stability percentage) in terms of
both infer per second and latency.

##### `--streaming`

Enables the use of the streaming API.

##### `--extra-inputs`

Provides an additional input for use with the model with a singular value,
such as `stream:true` or `max_tokens:5`. This flag can be repeated to supply multiple extra inputs.


##### `--endpoint <str>`

Set a custom endpoint that differs from the OpenAI defaults. This is ignored when
not using the `openai` service-kind.

##### `--endpoint-type {completions,chat}`

Describes what api to send requests to on the server. This is required when
using `openai` service-kind. This is ignored in other cases.

##### `-u <url>`
##### `--url <url>`

URL of the endpoint to target for benchmarking.


# Known Issues

* GenAI-Perf can be slow to finish if a high request-rate is provided
* Token counts may not be exact
* Token output counts are much higher than reality for now when running on
triton server, because the input is reflected back into the output
