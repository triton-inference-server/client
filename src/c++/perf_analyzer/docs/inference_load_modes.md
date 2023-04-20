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

# Inference Load Modes

Perf Analyzer has several modes for generating inference request load for a
model.

## Concurrency Mode

In concurrency mode, Perf Analyzer attempts to send inference requests to the
server such that N requests are always outstanding during profiling. For
example, when using
[`--concurrency-range=4`](cli.md#--concurrency-rangestartendstep), Perf Analyzer
will to attempt to have 4 outgoing inference requests at all times during
profiling.

## Request Rate Mode

In request rate mode, Perf Analyzer attempts to send N inference requests per
second to the server during profiling. For example, when using
[`--request-rate-range=20](cli.md#--request-rate-rangestartendstep), Perf
Analyzer will attempt to send 20 requests per second during profiling.

## Custom Interval Mode

In custom interval mode, Perf Analyzer attempts to send inference requests
according to intervals (between requests, looping if necessary) provided by the
user in the form of a text file with one time interval (in microseconds) per
line. For example, when using
[`--request-intervals=my_intervals.txt`](cli.md#--request-intervalspath),
where `my_intervals.txt` contains:

```
100000
200000
500000
```

Perf Analyzer will attempt to send requests at the following times: 0.1s, 0.3s,
0.8s, 0.9s, 1.1s, 1.6s, and so on, during profiling.
