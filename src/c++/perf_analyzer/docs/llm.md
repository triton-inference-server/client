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
(LLMs) using Triton with [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
and [vLLM](https://github.com/vllm-project/vllm).

## Setup: Download and configure Triton Server and Client environment

### Using TensorRT-LLM

1. Follow [step 1](https://github.com/triton-inference-server/tutorials/blob/main/Popular_Models_Guide/Llama2/trtllm_guide.md#installation)
of the Installation section. It includes instructions for cloning llama if you
do not already have it downloaded.

  ```
  git clone https://github.com/triton-inference-server/tensorrtllm_backend.git  --branch release/0.5.0
  # Update the submodules
  cd tensorrtllm_backend
  # Install git-lfs if needed
  sudo apt-get update && sudo apt-get install git-lfs -y --no-install-recommends
  git lfs install
  git submodule update --init --recursive
  ```

2. Launch the Triton docker container with the TensorRT-LLM backend.
This will require mounting the repo from step 1 into the docker container
and any models you plan to serve.

    For the tensorrtllm_backend repository, you need the following directories mounted:
- backend: .../tensorrtllm_backend/:/tensorrtllm_backend
- llama repo: .../llama/repo:/Llama-2-7b-hf
- engine: .../tensorrtllm_backend/tensorrt_llm/examples/llama/engine:/engines

```
docker run --rm -it --net host --shm-size=2g \
    --ulimit memlock=-1 --ulimit stack=67108864 --gpus all \
    -v $(pwd):/tensorrtllm_backend \
    -v /path/to/llama/repo:/Llama-2-7b-hf \
    -v $(pwd)/tensorrt_llm/examples/llama/engines:/engines \
    nvcr.io/nvidia/tritonserver:23.10-trtllm-python-py3 \
    bash
```

3. Follow the steps [here](https://github.com/triton-inference-server/tutorials/blob/main/Popular_Models_Guide/Llama2/trtllm_guide.md#create-engines-for-each-model-skip-this-step-if-you-already-have-an-engine)
to create the engine.

    Building the engine in the container with the `--output_dir /engines`
    flag will place the compiled `.engine` file under the expected directory set in step 1.

    Note:
    - Compiling the wheel and engine can take more than 1 hour.
    - If you get an error compiling bfloat16, you can remove it for the default
    option.


4. Serve the model with [Triton](https://github.com/triton-inference-server/tutorials/blob/main/Popular_Models_Guide/Llama2/trtllm_guide.md#serving-with-triton).

```
cp -R /tensorrtllm_backend/all_models/inflight_batcher_llm /opt/tritonserver/.
```

  After copying the model repository, use the following sed commands to set
  some required values in the config.pbtxt files.

```
sed -i 's#${tokenizer_dir}#/Llama-2-7b-hf/#' /opt/tritonserver/inflight_batcher_llm/preprocessing/config.pbtxt
sed -i 's#${tokenizer_type}#auto#' /opt/tritonserver/inflight_batcher_llm/preprocessing/config.pbtxt
sed -i 's#${tokenizer_dir}#/Llama-2-7b-hf/#' /opt/tritonserver/inflight_batcher_llm/postprocessing/config.pbtxt
sed -i 's#${tokenizer_type}#auto#' /opt/tritonserver/inflight_batcher_llm/postprocessing/config.pbtxt

sed -i 's#${decoupled_mode}#false#' /opt/tritonserver/inflight_batcher_llm/tensorrt_llm/config.pbtxt
sed -i 's#${engine_dir}#/engines/1-gpu/#' /opt/tritonserver/inflight_batcher_llm/tensorrt_llm/config.pbtxt
```

```
python3 /tensorrtllm_backend/scripts/launch_triton_server.py --world_size=<world size of the engine> --model_repo=/opt/tritonserver/inflight_batcher_llm
```

### Using vLLM

Download the pre-built Triton Server Container with vLLM backend from
[NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver)
registry.

```bash
docker pull nvcr.io/nvidia/tritonserver:23.10-vllm-python-py3
```

Run the Triton Server container with
[vLLM backend](https://github.com/triton-inference-server/vllm_backend) and
launch the server.
```bash
git clone -b r23.10 https://github.com/triton-inference-server/vllm_backend.git
cd vllm_backend

docker run --gpus all --rm -it --net host \
    --shm-size=2G --ulimit memlock=-1 --ulimit stack=67108864 \
    -v $(pwd)/samples/model_repository:/model_repository \
    nvcr.io/nvidia/tritonserver:23.10-vllm-python-py3 \
    tritonserver --model-repository /model_repository
```

### Run Triton Client SDK Container

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
python profile.py -m vllm_model --prompt-size-range 100 500 200 --max-tokens 1

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
>             "text_input": [
>                 "Hello, my name is"  // user-provided prompt
>             ],
>             "stream": [
>                 true
>             ],
>             "sampling_parameters": [
>                 "{ \"max_tokens\": 1 }"
>             ]
>         }
>     ]
> }
> ' > input_data.json
>
> $ python profile.py -m vllm_model --input-data input_data.json
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
python profile.py -m vllm_model --prompt-size-range 100 500 200 --max-tokens 256 --ignore-eos

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
python profile.py -m vllm_model --prompt-size-range 10 10 1 --periodic-concurrency-range 1 100 1 --request-period 32 --max-tokens 1024 --ignore-eos

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
python profile.py -m vllm_model --periodic-concurrency-range 1 4 1 --request-period 32 --max-tokens 1024 --ignore-eos
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

