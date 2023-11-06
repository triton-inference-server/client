<!--
Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

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

# Benchmarking LLM

The following guide shows the reader how to use Triton
[Perf Analyzer](https://github.com/triton-inference-server/client/tree/main/src/c%2B%2B/perf_analyzer)
to measure and characterize the performance behaviors of Large Language Models
(LLMs) using Triton with [vLLM](https://github.com/vllm-project/vllm).

### Setup: Download and configure Triton Server environment

From [Step 1 of the Triton vLLM tutorial](https://github.com/triton-inference-server/tutorials/blob/main/Quick_Deploy/vLLM/README.md#step-1-build-a-triton-container-image-with-vllm).

```bash
git clone https://github.com/triton-inference-server/tutorials
cd tutorials/Quick_Deploy/vLLM
docker build -t tritonserver_vllm .
# wait for command to finish, might take several minutes
```

Upon successful build, run the following command to start the Triton Server container:
```bash
docker run --gpus all -it --rm -p 8001:8001 --shm-size=1G --ulimit memlock=-1 --ulimit stack=67108864 -v ${PWD}:/work -w /work tritonserver_vllm tritonserver --model-store ./model_repository
```

Next run the following command to start the Triton SDK container:
```bash
git clone https://github.com/triton-inference-server/client.git
cd client/src/c++/perf_analyzer/docs/examples
docker run --gpus all -it --rm --net host -v ${PWD}:/work -w /work nvcr.io/nvidia/tritonserver:23.10-py3-sdk
```

## Benchmark 1: Profiling the Prefill Phase

In this benchmarking scenario, we want to measure the effect of input prompt
size on first-token latency. We issue single request to the server of fixed
input sizes and request the model to compute at most one new token. This
essentially means one pass through the model.

#### Example

Inside the client container, run the following command to generate dummy prompts
of size 100, 300, and 500 and receive single token from the model for each prompt.

```bash
python profile.py -m vllm --prompt-size-range 100 500 200 --max-tokens 1

# [ BENCHMARK SUMMARY ]
# Prompt size: 100
#   * Max first token latency: 35.2451 ms
#   * Min first token latency: 11.0879 ms
#   * Avg first token latency: 18.3775 ms
#   ...
```

> **Note**
>
> Users can also run a custom prompt by providing input data JSON file using
> `--input-data` option. They can also specify input tensors or parameters to
> the model as well. However, when a parameter is defined in both input data
> JSON file and through command line option (e.g. `max_tokens`), the command
> line option value will overwrite the one in the input data JSON file.
> ```bash
> $ echo '
> {
>     "data": [
>         {
>             "PROMPT": [
>                 "Hello, my name is"  // user-provided prompt
>             ],
>             "STREAM": [
>                 true
>             ],
>             "SAMPLING_PARAMETERS": [
>                 "{ \"max_tokens\": 1 }"
>             ]
>         }
>     ]
> }
> ' > input_data.json
>
> $ python profile.py -m vllm --input-data input_data.json
> ```


## Benchmark 2: Profiling the Generation Phase

In this benchmarking scenario, we want to measure the effect of input prompt
size on token-to-token latency. We issue single request to the server of fixed
input sizes and request the model to compute a fixed amount of tokens.

#### Example

Inside the client container, run the following command to generate dummy prompts
of size 100, 300, and 500 and receive total 256 tokens from the model for each
prompts.

```bash
python profile.py -m vllm --prompt-size-range 100 500 200 --max-tokens 256 --ignore-eos

# [ BENCHMARK SUMMARY ]
# Prompt size: 100
#   * Max first token latency: 23.2899 ms
#   * Min first token latency: 11.0127 ms
#   * Avg first token latency: 16.0468 ms
#  ...
```

## Benchmark 3: Profiling In-Flight Batching

In this benchmarking scenario, we want to measure the effect of in-flight
batch size on token-to-token (T2T) latency. We systematically issue requests to
the server of fixed input sizes and request the model to compute a fixed amount
of tokens in order to increase the in-flight batch size over time.

#### Example

In this benchmark, we will run Perf Analyzer in
[periodic concurrency mode](inference_load_modes.md#periodic-concurrency-mode)
that periodically launches a new concurrent request to the model using
`--periodic-concurrency-range START END STEP` option.
In this example, Perf Analyzer starts with a single request and launches the new
ones until the total number reaches 100.
You can also specify the timing of the new requests:
Setting `--request-period` to 32 (as shown below) will make Perf Analyzer to
wait for all the requests to receive 32 responses before launching new requests.
Run the following command inside the client container.

```bash
# Install matplotlib to generate the benchmark plot
pip install matplotlib

# Run Perf Analyzer
python profile.py -m vllm --prompt-size-range 10 10 1 --periodic-concurrency-range 1 100 1 --request-period 32 --max-tokens 1024 --ignore-eos

# [ BENCHMARK SUMMARY ]
# Prompt size: 10
#   * Max first token latency: 125.7212 ms
#   * Min first token latency: 18.4281 ms
#   * Avg first token latency: 61.8372 ms
#   ...
# Saved in-flight batching benchmark plots @ 'inflight_batching_benchmark-*.png'.
```

The resulting plot will look like

<img src="examples/inflight_batching_benchmark.png" width="600">

The plot demonstrates how the average T2T latency changes across the entire
benchmark process as we increase the number of requests.
To observe the change, we first align the responses of every requests and then
split them into multiple segments of responses.
For instance, assume we ran the following benchmark command:

```bash
python profile.py -m vllm --periodic-concurrency-range 1 4 1 --request-period 32 --max-tokens 1024 --ignore-eos
```

We start from a single request and increment up to 4 requests one by one for
every 32 responses (defined by `--request-period`).
For each request, there are total 1024 generated responses (defined by `--max-tokens`).
We align these total 1024 generated responses and split them by request period,
giving us 1024/32 = 32 total segments per request as shown below:

```
          32 responses (=request period)
            ┌────┐
request 1   ──────┊──────┊──────┊──────┊─ ··· ─┊──────┊
request 2         ┊──────┊──────┊──────┊─ ··· ─┊──────┊──────┊
request 3         ┊      ┊──────┊──────┊─ ··· ─┊──────┊──────┊──────┊
request 4         ┊      ┊      ┊──────┊─ ··· ─┊──────┊──────┊──────┊──────

segment #     1      2      3       4     ···     32     33     34     35
```

Then for each segment, we compute the mean of T2T latencies of the responses.
This will allow us to visualize the change in T2T latency as the number of
requests increase, filling up the inflight batch slots, and as they terminate.
See [profile.py](examples/profile.py) for more details.

