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

# Benchmarking Triton via HTTP or gRPC endpoint

This is the default mode for Perf Analyzer.

# Benchmarking Triton directly via C API

Besides using HTTP or gRPC server endpoints to communicate with Triton, Perf
Analyzer also allows users to benchmark Triton directly using the C API. HTTP
and gRPC endpoints introduce an additional latency in the pipeline which may not
be of interest to users who are using Triton via C API within their application.
Specifically, this feature is useful to benchmark a bare minimum Triton without
additional overheads from HTTP/gRPC communication.

## Prerequisite

Pull the Triton SDK and the Triton Server container images on target machine.
Since you will need access to the `tritonserver` install, it might be easier if
you copy the `perf_analyzer` binary to the Inference Server container.

## Required parameters

Use the [`--help`](cli.md#--help) option to see a complete list of supported
command line arguments. By default, Perf Analyzer expects the Triton instance to
already be running. You can configure C API mode using the
[`--service-kind`](cli.md#--service-kindtritontriton_c_apitfservingtorchserve)
option. In additon, you will need to point Perf Analyzer to the Triton server
library path using the
[`--triton-server-directory`](cli.md#--triton-server-directorypath) option and
the model repository path using the
[`--model-repository`](cli.md#--model-repositorypath) option.

An example run would look like:

```
$ perf_analyzer -m my_model --service-kind=triton_c_api --triton-server-directory=/opt/tritonserver --model-repository=/my/model/repository
...
*** Measurement Settings ***
  Service Kind: Triton C-API
  Using "time_windows" mode for stabilization
  Measurement window: 5000 msec
  Using synchronous calls for inference
  Stabilizing using average latency

Request concurrency: 1
  Client: 
    Request count: 353
    Throughput: 19.6095 infer/sec
    Avg latency: 50951 usec (standard deviation 2265 usec)
    p50 latency: 50833 usec
    p90 latency: 50923 usec
    p95 latency: 50940 usec
    p99 latency: 50985 usec
    
  Server: 
    Inference count: 353
    Execution count: 353
    Successful request count: 353
    Avg request latency: 50841 usec (overhead 20 usec + queue 63 usec + compute input 35 usec + compute infer 50663 usec + compute output 59 usec)

Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 19.6095 infer/sec, latency 50951 usec
```

## Non-supported functionalities

There are a few functionalities that are missing from C API mode. They are:

1. Async mode ([`--async`](cli.md#--async))
2. For additonal known non-working cases, please refer to
   [qa/L0_perf_analyzer_capi/test.sh](https://github.com/triton-inference-server/server/blob/main/qa/L0_perf_analyzer_capi/test.sh#L239-L277)

# Benchmarking TensorFlow Serving

Perf Analyzer can also be used to benchmark models deployed on
[TensorFlow Serving](https://github.com/tensorflow/serving) using the
[`--service-kind=tfserving`](cli.md#--service-kindtritontriton_c_apitfservingtorchserve)
option. Only gRPC protocol is supported.

The following invocation demonstrates how to configure Perf Analyzer to issue
requests to a running instance of `tensorflow_model_server`:

```
$ perf_analyzer -m resnet50 --service-kind tfserving -i grpc -b 1 -p 5000 -u localhost:8500
*** Measurement Settings ***
  Batch size: 1
  Using "time_windows" mode for stabilization
  Measurement window: 5000 msec
  Using synchronous calls for inference
  Stabilizing using average latency
Request concurrency: 1
  Client:
    Request count: 829
    Throughput: 165.8 infer/sec
    Avg latency: 6032 usec (standard deviation 569 usec)
    p50 latency: 5863 usec
    p90 latency: 6655 usec
    p95 latency: 6974 usec
    p99 latency: 8093 usec
    Avg gRPC time: 5984 usec ((un)marshal request/response 257 usec + response wait 5727 usec)
Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 165.8 infer/sec, latency 6032 usec
```

You might have to specify a different url ([`-u`](cli.md#-u-url)) to access
wherever the server is running. The report of Perf Analyzer will only include
statistics measured at the client-side.

**NOTE:** The support is still in **beta**. Perf Analyzer does not guarantee
optimal tuning for TensorFlow Serving. However, a single benchmarking tool that
can be used to stress the inference servers in an identical manner is important
for performance analysis.

The following points are important for interpreting the results:

1. `Concurrent Request Execution`:
   TensorFlow Serving (TFS), as of version 2.8.0, by default creates threads for
   each request that individually submits requests to TensorFlow Session. There
   is a resource limit on the number of concurrent threads serving requests.
   When benchmarking at a higher request concurrency, you can see higher
   throughput because of this. Unlike TFS, by default Triton is configured with
   only a single
   [instance count](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#instance-groups).
   Hence, at a higher request concurrency, most of the requests are blocked on
   the instance availability. To configure Triton to behave like TFS, set the
   instance count to a reasonably high value and then set
   [MAX_SESSION_SHARE_COUNT](https://github.com/triton-inference-server/tensorflow_backend#parameters)
   parameter in the model `config.pbtxt` to the same value. For some context,
   the TFS sets its thread constraint to four times the num of schedulable CPUs.
2. `Different library versions`:
   The version of TensorFlow might differ between Triton and TensorFlow Serving
   being benchmarked. Even the versions of CUDA libraries might differ between
   the two solutions. The performance of models can be susceptible to the
   versions of these libraries. For a single request concurrency, if the
   `compute_infer` time reported by Perf Analyzer when benchmarking Triton is as
   large as the latency reported by Perf Analyzer when benchmarking TFS, then
   the performance difference is likely because of the difference in the
   software stack and outside the scope of Triton.
3. `CPU Optimization`:
   TFS has separate builds for CPU and GPU targets. They have target-specific
   optimization. Unlike TFS, Triton has a single build which is optimized for
   execution on GPUs. When collecting performance on CPU models on Triton, try
   running Triton with the environment variable `TF_ENABLE_ONEDNN_OPTS=1`.

# Benchmarking TorchServe

Perf Analyzer can also be used to benchmark
[TorchServe](https://github.com/pytorch/serve) using the
[`--service-kind=torchserve`](cli.md#--service-kindtritontriton_c_apitfservingtorchserve)
option. Only HTTP protocol is supported. It also requires input to be provided
via JSON file.

The following invocation demonstrates how to configure Perf Analyzer to issue
requests to a running instance of `torchserve` assuming the location holds
`kitten_small.jpg`:

```
$ perf_analyzer -m resnet50 --service-kind torchserve -i http -u localhost:8080 -b 1 -p 5000 --input-data data.json
 Successfully read data for 1 stream/streams with 1 step/steps.
*** Measurement Settings ***
  Batch size: 1
  Using "time_windows" mode for stabilization
  Measurement window: 5000 msec
  Using synchronous calls for inference
  Stabilizing using average latency
Request concurrency: 1
  Client:
    Request count: 799
    Throughput: 159.8 infer/sec
    Avg latency: 6259 usec (standard deviation 397 usec)
    p50 latency: 6305 usec
    p90 latency: 6448 usec
    p95 latency: 6494 usec
    p99 latency: 7158 usec
    Avg HTTP time: 6272 usec (send/recv 77 usec + response wait 6195 usec)
Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 159.8 infer/sec, latency 6259 usec
```

The content of `data.json`:

```json
 {
   "data" :
    [
       {
         "TORCHSERVE_INPUT" : ["kitten_small.jpg"]
       }
     ]
 }
```

You might have to specify a different url ([`-u`](cli.md#-u-url)) to access
wherever the server is running. The report of Perf Analyzer will only include
statistics measured at the client-side.

**NOTE:** The support is still in **beta**. Perf Analyzer does not guarantee
optimal tuning for TorchServe. However, a single benchmarking tool that can be
used to stress the inference servers in an identical manner is important for
performance analysis.

# Advantages of using Perf Analyzer over third-party benchmark suites

Triton Inference Server offers the entire serving solution which includes
[client libraries](https://github.com/triton-inference-server/client) that are
optimized for Triton. Using third-party benchmark suites like `jmeter` fails to
take advantage of the optimized libraries. Some of these optimizations includes
but are not limited to:

1. Using
   [binary tensor data extension](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_binary_data.md#binary-tensor-data-extension)
   with HTTP requests.
2. Effective re-use of gRPC message allocation in subsequent requests.
3. Avoiding extra memory copy via libcurl interface.

These optimizations can have a tremendous impact on overall performance. Using
Perf Analyzer for benchmarking directly allows a user to access these
optimizations in their study.

Not only that, Perf Analyzer is also very customizable and supports many Triton
features as described in this document. This, along with a detailed report,
allows a user to identify performance bottlenecks and experiment with different
features before deciding upon what works best for them.
