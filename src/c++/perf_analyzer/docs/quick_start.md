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

# Quick Start

The steps below will guide you on how to start using Perf Analyzer.

### Step 1: Start Triton Container

```bash
export RELEASE=<yy.mm> # e.g. to use the release from the end of February of 2023, do `export RELEASE=23.02`

docker pull nvcr.io/nvidia/tritonserver:${RELEASE}-py3

docker run --gpus all --rm -it --net host nvcr.io/nvidia/tritonserver:${RELEASE}-py3
```

### Step 2: Download `simple` Model

```bash
# inside triton container
git clone --depth 1 https://github.com/triton-inference-server/server

mkdir model_repository ; cp -r server/docs/examples/model_repository/simple model_repository
```

### Step 3: Start Triton Server

```bash
# inside triton container
tritonserver --model-repository $(pwd)/model_repository &> server.log &

# confirm server is ready, look for 'HTTP/1.1 200 OK'
curl -v localhost:8000/v2/health/ready

# detach (CTRL-p CTRL-q)
```

### Step 4: Start Triton SDK Container

```bash
docker pull nvcr.io/nvidia/tritonserver:${RELEASE}-py3-sdk

docker run --gpus all --rm -it --net host nvcr.io/nvidia/tritonserver:${RELEASE}-py3-sdk
```

### Step 5: Run Perf Analyzer

```bash
# inside sdk container
perf_analyzer -m simple
```

### Step 6: Observe and Analyze Output

```
$ perf_analyzer -m simple
*** Measurement Settings ***
  Batch size: 1
  Service Kind: Triton
  Using "time_windows" mode for stabilization
  Measurement window: 5000 msec
  Using synchronous calls for inference
  Stabilizing using average latency

Request concurrency: 1
  Client:
    Request count: 25348
    Throughput: 1407.84 infer/sec
    Avg latency: 708 usec (standard deviation 663 usec)
    p50 latency: 690 usec
    p90 latency: 881 usec
    p95 latency: 926 usec
    p99 latency: 1031 usec
    Avg HTTP time: 700 usec (send/recv 102 usec + response wait 598 usec)
  Server:
    Inference count: 25348
    Execution count: 25348
    Successful request count: 25348
    Avg request latency: 382 usec (overhead 41 usec + queue 41 usec + compute input 26 usec + compute infer 257 usec + compute output 16 usec)

Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 1407.84 infer/sec, latency 708 usec
```

We can see from the output that the model was able to complete approximately
1407.84 inferences per second, with an average latency of 708 microseconds per
inference request. Concurrency of 1 meant that Perf Analyzer attempted to always
have 1 outgoing request at all times.
