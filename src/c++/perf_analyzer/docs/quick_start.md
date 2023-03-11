<!--
Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

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

# Profiling a Model Quick Start

The steps below will guide you through using Perf Analyzer to profile a simple
PyTorch model: `add_sub`.

## Step 1: Download `add_sub` Model

```bash
git clone --depth=1 https://github.com/triton-inference-server/model_analyzer
mkdir model_repository
cp -r model_analyzer/examples/quick-start/add_sub model_repository
```

## Step 2: Start Triton Server

```bash
tritonserver --model-repository $(pwd)/model_repository &> /dev/null & 
```

## Step 3: Run Perf Analyzer

```bash
perf_analyzer -m add_sub
```

## Step 4: Observe Output

```
*** Measurement Settings ***
  Batch size: 1
  Service Kind: Triton
  Using "time_windows" mode for stabilization
  Measurement window: 5000 msec
  Using synchronous calls for inference
  Stabilizing using average latency

Request concurrency: 1
  Client: 
    Request count: 24024
    Throughput: 1334.23 infer/sec
    Avg latency: 747 usec (standard deviation 2253 usec)
    p50 latency: 737 usec
    p90 latency: 898 usec
    p95 latency: 916 usec
    p99 latency: 1022 usec
    Avg HTTP time: 737 usec (send/recv 110 usec + response wait 627 usec)
  Server: 
    Inference count: 24025
    Execution count: 24025
    Successful request count: 24025
    Avg request latency: 388 usec (overhead 52 usec + queue 50 usec + compute input 68 usec + compute infer 135 usec + compute output 81 usec)

Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 1334.23 infer/sec, latency 747 usec
```