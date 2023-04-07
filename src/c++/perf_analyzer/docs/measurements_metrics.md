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

# Measurement Modes

Currently, Perf Analyzer has 2 measurement modes.

## Time Windows

When using time windows measurement mode
([`--measurement-mode=time_windows`](cli.md#-measurement-modetime_windowscount_windows)),
Perf Analyzer will count how many requests have completed during a window of
duration `X` (in milliseconds, via `--measurement-interval=X`, default is
`5000`). This is the default measurement mode.

## Count Windows

When using count windows measurement mode
([`--measurement-mode=count_windows`](cli.md#-measurement-modetime_windowscount_windows)),
Perf Analyzer will start the window duration at 1 second and potentially
dynamically increase it until `X` requests have completed (via
[`--measurement-request-count=X`](cli.md#-measurement-request-countn), default is `50`).

# Metrics

## How Throughput is Calculated

Perf Analyzer calculates throughput to be the total number of requests completed
during a measurement, divided by the duration of the measurement, in seconds.

## How Latency is Calculated

For each request concurrency level perf*analyzer reports latency and
throughput as seen from the \_client* (that is, as seen by
perf_analyzer) and also the average request latency on the server.

The server latency measures the total time from when the request is
received at the server until the response is sent from the
server. Because of the HTTP and GRPC libraries used to implement the
server endpoints, total server latency is typically more accurate for
HTTP requests as it measures time from first byte received until last
byte sent. For both HTTP and GRPC the total server latency is
broken-down into the following components:

- _queue_: The average time spent in the inference schedule queue by a
  request waiting for an instance of the model to become available.
- _compute_: The average time spent performing the actual inference,
  including any time needed to copy data to/from the GPU.

The client latency time is broken-down further for HTTP and GRPC as
follows:

- HTTP: _send/recv_ indicates the time on the client spent sending the
  request and receiving the response. _response wait_ indicates time
  waiting for the response from the server.
- GRPC: _(un)marshal request/response_ indicates the time spent
  marshalling the request data into the GRPC protobuf and
  unmarshalling the response data from the GRPC protobuf. _response
  wait_ indicates time writing the GRPC request to the network,
  waiting for the response, and reading the GRPC response from the
  network.

Use the verbose (-v) option to perf_analyzer to see more output,
including the stabilization passes run for each request concurrency
level.

# Reports

## Visualizing Latency vs. Throughput

The perf_analyzer provides the -f option to generate a file containing
CSV output of the results.

```
$ perf_analyzer -m inception_graphdef --concurrency-range 1:4 -f perf.csv
$ cat perf.csv
Concurrency,Inferences/Second,Client Send,Network+Server Send/Recv,Server Queue,Server Compute Input,Server Compute Infer,Server Compute Output,Client Recv,p50 latency,p90 latency,p95 latency,p99 latency
1,69.2,225,2148,64,206,11781,19,0,13891,18795,19753,21018
3,84.2,237,1768,21673,209,11742,17,0,35398,43984,47085,51701
4,84.2,279,1604,33669,233,11731,18,1,47045,56545,59225,64886
2,87.2,235,1973,9151,190,11346,17,0,21874,28557,29768,34766
```

NOTE: The rows in the CSV file are sorted in an increasing order of throughput (Inferences/Second).

You can import the CSV file into a spreadsheet to help visualize
the latency vs inferences/second tradeoff as well as see some
components of the latency. Follow these steps:

- Open [this
  spreadsheet](https://docs.google.com/spreadsheets/d/1S8h0bWBBElHUoLd2SOvQPzZzRiQ55xjyqodm_9ireiw)
- Make a copy from the File menu "Make a copy..."
- Open the copy
- Select the A1 cell on the "Raw Data" tab
- From the File menu select "Import..."
- Select "Upload" and upload the file
- Select "Replace data at selected cell" and then select the "Import data" button

## Server-side Prometheus metrics

Perf Analyzer can collect server-side metrics, such as
GPU utilization and GPU power usage. To enable the collection of these metrics,
use the [`--collect-metrics`](cli.md#-collect-metrics) CLI option.

Perf Analyzer defaults to access the metrics endpoint at
`localhost:8002/metrics`. If the metrics are accessible at a different url, use
the [`--metrics-url <url>`](cli.md#-metrics-urlurl) CLI option to specify that.

Perf Analyzer defaults to access the metrics endpoint every 1000 milliseconds.
To use a different accessing interval, use the [`--metrics-interval <interval>`](cli.md#-metrics-intervaln)
CLI option (specify in milliseconds).

Because Perf Analyzer can collect the server-side metrics multiple times per
run, these metrics are aggregated in specific ways to produce one final number
per sweep (concurrency/request rate). Here are how they are aggregated:

| Metric           | Aggregation                                                                                                                                                  |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| GPU Utilization  | Averaged from each collection taken during stable passes. We want a number representative of all stable passes.                                              |
| GPU Power Usage  | Averaged from each collection taken during stable passes. We want a number representative of all stable passes.                                              |
| GPU Used Memory  | Maximum from all collections taken during a stable pass. Users are typically curious what the peak memory usage is for determining model/hardware viability. |
| GPU Total Memory | First from any collection taken during a stable pass. All of the collections should produce the same value for total memory available on the GPU.            |

Note that all metrics are per-GPU in the case of multi-GPU systems.

To output these server-side metrics to a CSV file, use the [`-f <filename>`](cli.md#-f-path) and
[`--verbose-csv`](cli.md#-verbose-csv) CLI options. The output CSV will contain one column per metric.
The value of each column will be a `key:value` pair (`GPU UUID:metric value`).
Each `key:value` pair will be delimited by a semicolon (`;`) to indicate metric
values for each GPU accessible by the server. There is a trailing semicolon. See
below:

`<gpu-uuid-0>:<metric-value>;<gpu-uuid-1>:<metric-value>;...;`

Here is a simplified CSV output:

```bash
$ perf_analyzer -m resnet50_libtorch --collect-metrics -f output.csv --verbose-csv
$ cat output.csv
Concurrency,...,Avg GPU Utilization,Avg GPU Power Usage,Max GPU Memory Usage,Total GPU Memory
1,...,gpu_uuid_0:0.33;gpu_uuid_1:0.5;,gpu_uuid_0:55.3;gpu_uuid_1:56.9;,gpu_uuid_0:10000;gpu_uuid_1:11000;,gpu_uuid_0:50000;gpu_uuid_1:75000;,
2,...,gpu_uuid_0:0.25;gpu_uuid_1:0.6;,gpu_uuid_0:25.6;gpu_uuid_1:77.2;,gpu_uuid_0:11000;gpu_uuid_1:17000;,gpu_uuid_0:50000;gpu_uuid_1:75000;,
3,...,gpu_uuid_0:0.87;gpu_uuid_1:0.9;,gpu_uuid_0:87.1;gpu_uuid_1:71.7;,gpu_uuid_0:15000;gpu_uuid_1:22000;,gpu_uuid_0:50000;gpu_uuid_1:75000;,
```
