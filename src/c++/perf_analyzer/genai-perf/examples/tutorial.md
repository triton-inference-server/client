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

# Benchmarking LLM

The following guide provides an example on how to use GenAI-Perf
to measure and characterize the performance behaviors of Large Language Models
(LLMs).

## Setup: Download and configure Triton Server and Client environment

1. In a clean working directory, start by executing the following code snippet
to create a triton model repository:

```bash
MODEL_REPO="${PWD}/models"
MODEL_NAME="gpt2_vllm"

mkdir -p $MODEL_REPO/$MODEL_NAME/1
echo '{
    "model":"gpt2",
    "disable_log_requests": "true",
    "gpu_memory_utilization": 0.5
}' >$MODEL_REPO/$MODEL_NAME/1/model.json

echo 'backend: "vllm"
instance_group [
  {
    count: 1
    kind: KIND_MODEL
  }
]' >$MODEL_REPO/$MODEL_NAME/config.pbtxt
```

2. Download the pre-built Triton Server Container with the vLLM backend from the
[NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver)
registry.

```bash
export RELEASE=<yy.mm> # e.g. to use the release from the end of March of 2024, do `export RELEASE=24.03`

docker pull nvcr.io/nvidia/tritonserver:${RELEASE}-vllm-python-py3
```

3. Launch the triton container to serve the gpt2_vllm model.

```bash
docker run -it --gpus all --net=host --rm --shm-size=1G --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v ${PWD}:/work -w /work \
    nvcr.io/nvidia/tritonserver:${RELEASE}-vllm-python-py3 \
    tritonserver --model-repository ./models
```

4. In a separate terminal window, download the pre-built Triton SDK Container
which includes GenAI-Perf and Performance Analyzer from the
[NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver)
registry and launch the container.

```bash
export RELEASE=<yy.mm>
docker run --gpus all -it --net host --shm-size=1g --ulimit stack=67108864 \
  nvcr.io/nvidia/tritonserver:${RELEASE}-py3-sdk \
  bash
```

5. Profile GPT2 via GenAI-Perf

```bash
genai-perf -m gpt2_vllm --concurrency 1 --output-format vllm --streaming
```

By default, all metrics will saved in the `profile_export_genai_perf.csv` file.