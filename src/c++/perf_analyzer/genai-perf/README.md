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
export RELEASE="mm.yy" # e.g. export RELEASE="24.03"

docker run -it --net=host --gpus=all  nvcr.io/nvidia/tritonserver:${RELEASE}-py3-sdk
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
export RELEASE="mm.yy" # e.g. export RELEASE="24.03"

pip install "git+https://github.com/triton-inference-server/client.git@r${RELEASE}#subdirectory=src/c++/perf_analyzer/genai-perf"
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
  --backend tensorrtllm \
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

See [Tutorial](docs/tutorial.md) for additional examples.


## Using `compare` Subcommand to Visualize Multiple Runs

The `compare` subcommand in GenAI-Perf facilitates users in comparing multiple
profile runs and visualizing the differences through plots.

### Usage
Assuming the user possesses two profile export JSON files,
namely `profile1.json` and `profile2.json`,
they can execute the `compare` subcommand using the `--files` option:

```bash
genai-perf compare --files profile1.json profile2.json
```

Executing the above command will create following files under `compare` directory:
1. Generate a YAML configuration file (e.g. `config.yaml`) containing the
metadata for each plot generated during the comparison process.
2. Automatically generate the [default set of plots](docs/compare.md#example-plots)
(e.g. TTFT vs. Number of Input Tokens) that compare the two profile runs.

```
compare
├── config.yaml
├── distribution_of_input_tokens_to_generated_tokens.jpeg
├── request_latency.jpeg
├── time_to_first_token.jpeg
├── time_to_first_token_vs_number_of_input_tokens.jpeg
├── token-to-token_latency_vs_output_token_position.jpeg
└── ...
```

### Customization
Users have the flexibility to iteratively modify the generated YAML configuration
file to suit their specific requirements.
They can make alterations to the plots according to their preferences and execute
the command with the `--config` option followed by the path to the modified
configuration file:

```bash
genai-perf compare --config compare/config.yaml
```

This command will regenerate the plots based on the updated configuration settings,
enabling users to refine the visual representation of the comparison results as
per their needs.

See [Compare documentation](docs/compare.md) for more details.

</br>

# Model Inputs

GenAI-Perf supports model input prompts from either synthetically generated
inputs, or from the HuggingFace
[OpenOrca](https://huggingface.co/datasets/Open-Orca/OpenOrca) or
[CNN_DailyMail](https://huggingface.co/datasets/cnn_dailymail) datasets. This is
specified using the `--input-dataset` CLI option.

When the dataset is synthetic, you can specify the following options:
* `--num-prompts <int>`: The number of unique prompts to generate as stimulus, >= 1.
* `--synthetic-input-tokens-mean <int>`: The mean of number of tokens in the
  generated prompts when using synthetic data, >= 1.
* `--synthetic-input-tokens-stddev <int>`: The standard deviation of number of
  tokens in the generated prompts when using synthetic data, >= 0.
* `--random-seed <int>`: The seed used to generate random values, >= 0.

When the dataset is coming from HuggingFace, you can specify the following
options:
* `--input-dataset {openorca,cnn_dailymail}`: HuggingFace dataset to use for
  benchmarking.
* `--num-prompts <int>`: The number of unique prompts to generate as stimulus, >= 1.

When the dataset is coming from a file, you can specify the following
options:
* `--input-file <path>`: The input file containing the single prompt to
  use for benchmarking.

For any dataset, you can specify the following options:
* `--output-tokens-mean <int>`: The mean number of tokens in each output. Ensure
  the `--tokenizer` value is set correctly, >= 1.
* `--output-tokens-stddev <int>`: The standard deviation of the number of tokens
  in each output. This is only used when output-tokens-mean is provided, >= 1.
* `--output-tokens-mean-deterministic`: When using `--output-tokens-mean`, this
  flag can be set to improve precision by setting the minimum number of tokens
  equal to the requested number of tokens. This is currently supported with the
  Triton service-kind. Note that there is still some variability in the
  requested number of output tokens, but GenAi-Perf attempts its best effort
  with your model to get the right number of output tokens.

You can optionally set additional model inputs with the following option:
* `--extra-inputs <input_name>:<value>`: An additional input for use with the
  model with a singular value, such as `stream:true` or `max_tokens:5`. This
  flag can be repeated to supply multiple extra inputs.

</br>

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

</br>

# Command Line Options

##### `-h`
##### `--help`

Show the help message and exit.

## Endpoint Options:

##### `-m <str>`
##### `--model <str>`

The name of the model to benchmark. (default: `None`)

##### `--backend {tensorrtllm,vllm}`

When using the "triton" service-kind, this is the backend of the model. For the
TRT-LLM backend, you currently must set `exclude_input_in_output` to true in the
model config to not echo the input tokens in the output. (default: tensorrtllm)

##### `--endpoint <str>`

Set a custom endpoint that differs from the OpenAI defaults. (default: `None`)

##### `--endpoint-type {chat,completions}`

The endpoint-type to send requests to on the server. This is only used with the
`openai` service-kind. (default: `None`)

##### `--service-kind {triton,openai}`

The kind of service perf_analyzer will generate load for. In order to use
`openai`, you must specify an api via `--endpoint-type`. (default: `triton`)

##### `--streaming`

An option to enable the use of the streaming API. (default: `False`)

##### `-u <url>`
##### `--url <url>`

URL of the endpoint to target for benchmarking. (default: `None`)

## Input Options

##### `--extra-inputs <str>`

Provide additional inputs to include with every request. You can repeat this
flag for multiple inputs. Inputs should be in an input_name:value format.
(default: `None`)

##### `--input-dataset {openorca,cnn_dailymail}`

The HuggingFace dataset to use for prompts.
(default: `openorca`)

##### `--input-file <path>`

The input file containing the single prompt to use for profiling.

##### `--num-prompts <int>`

The number of unique prompts to generate as stimulus. (default: `100`)

##### `--output-tokens-mean <int>`

The mean number of tokens in each output. Ensure the `--tokenizer` value is set
correctly. (default: `-1`)

##### `--output-tokens-mean-deterministic`

When using `--output-tokens-mean`, this flag can be set to improve precision by
setting the minimum number of tokens equal to the requested number of tokens.
This is currently supported with the Triton service-kind. Note that there is
still some variability in the requested number of output tokens, but GenAi-Perf
attempts its best effort with your model to get the right number of output
tokens. (default: `False`)

##### `--output-tokens-stddev <int>`

The standard deviation of the number of tokens in each output. This is only used
when `--output-tokens-mean` is provided. (default: `0`)

##### `--random-seed <int>`

The seed used to generate random values. (default: `0`)

##### `--synthetic-input-tokens-mean <int>`

The mean of number of tokens in the generated prompts when using synthetic
data. (default: `550`)

##### `--synthetic-input-tokens-stddev <int>`

The standard deviation of number of tokens in the generated prompts when
using synthetic data. (default: `0`)

## Profiling Options

##### `--concurrency <int>`

The concurrency value to benchmark. (default: `None`)

##### `--measurement-interval <int>`
##### `-p <int>`

The time interval used for each measurement in milliseconds. Perf Analyzer
will sample a time interval specified and take measurement over the requests
completed within that time interval. (default: `10000`)

##### `--request-rate <float>`

Sets the request rate for the load generated by PA. (default: `None`)

##### `-s <float>`
##### `--stability-percentage <float>`

The allowed variation in latency measurements when determining if a result is
stable. The measurement is considered as stable if the ratio of max / min from
the recent 3 measurements is within (stability percentage) in terms of both
infer per second and latency. (default: `999`)

## Output Options

##### `--generate-plots`

An option to enable the generation of plots. (default: False)

##### `--profile-export-file <path>`

The path where the perf_analyzer profile export will be generated. By default,
the profile export will be to `profile_export.json`. The genai-perf file will be
exported to `<profile_export_file>_genai_perf.csv`. For example, if the profile
export file is `profile_export.json`, the genai-perf file will be exported to
`profile_export_genai_perf.csv`. (default: `profile_export.json`)

## Other Options

##### `--tokenizer <str>`

The HuggingFace tokenizer to use to interpret token metrics from prompts and
responses. (default: `hf-internal-testing/llama-tokenizer`)

##### `-v`
##### `--verbose`

An option to enable verbose mode. (default: `False`)

##### `--version`

An option to print the version and exit.

# Known Issues

* GenAI-Perf can be slow to finish if a high request-rate is provided
* Token counts may not be exact
