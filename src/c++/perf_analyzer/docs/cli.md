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

# Perf Analyzer CLI

This document details the Perf Analyzer command line interface:

- [General Options](#general-options)
- [Measurement Options](#measurement-options)
- [Sequence Model Options](#sequence-model-options)
- [Input Data Options](#input-data-options)
- [Request Options](#request-options)
- [Server Options](#server-options)
- [Prometheus Metrics Options](#prometheus-metrics-options)
- [Report Options](#report-options)
- [Trace Options](#trace-options)
- [Deprecated Options](#deprecated-options)

## General Options

#### `-?`
#### `-h`
#### `--help`

Prints a description of the Perf Analyzer command line interface.

#### `-m <string>`

Specifies the model name for Perf Analyzer to run.

This is a required option.

#### `-x <string>`

Specifies the version of the model to be used. If not specified the most
recent version (the highest numbered version) of the model will be used.

#### `--service-kind=[triton|triton_c_api|tfserving|torchserve]`

Specifies the kind of service for Perf Analyzer to generate load for. Note: in
order to use `torchserve` backend, the `--input-data` option must point to a
JSON file holding data in the following format:

```
{
  "data": [
    {
      "TORCHSERVE_INPUT": [
        "<complete path to the content file>"
      ]
    },
    {...},
    ...
  ]
}
```

The type of file here will depend on the model. In order to use `triton_c_api`
you must specify the Triton server install path and the model repository path
via the `--triton-server-directory` and `--model-repository` options.

Default is `triton`.

#### `--bls-composing-models=<string>`

Specifies the list of all BLS composing models as a comma separated list of
model names (with optional model version number after a colon for each) that may
be called by the input BLS model. For example,
`--bls-composing-models=modelA:3,modelB` would specify that modelA and modelB
are composing models that may be called by the input BLS model, and that modelA
will use version 3, while modelB's version is unspecified.

#### `--model-signature-name=<string>`

Specifies the signature name of the saved model to use.

Default is `serving_default`. This option will be ignored if `--service-kind`
is not `tfserving`.

#### `-v`

Enables verbose mode. May be specified an additional time (`-v -v`) to enable
extra verbose mode.

## Measurement Options

#### `--measurement-mode=[time_windows|count_windows]`

Specifies the mode used for stabilizing measurements. 'time_windows' will
create windows such that the duration of each window is equal to
`--measurement-interval`. 'count_windows' will create windows such that there
are at least `--measurement-request-count` requests in each window and that
the window is at least one second in duration (adding more requests if
necessary).

Default is `time_windows`.

#### `-p <n>`
#### `--measurement-interval=<n>`

Specifies the time interval used for each measurement in milliseconds when
`--measurement-mode=time_windows` is used. Perf Analyzer will sample a time
interval specified by this option and take measurement over the requests
completed within that time interval.

Default is `5000`.

#### `--measurement-request-count=<n>`

Specifies the minimum number of requests to be collected in each measurement
window when `--measurement-mode=count_windows` is used.

Default is `50`.

#### `-s <n>`
#### `--stability-percentage=<n>`

Specifies the allowed variation in latency measurements when determining if a
result is stable. The measurement is considered stable if the ratio of max /
min from the recent 3 measurements is within (stability percentage)% in terms
of both inferences per second and latency.

Default is `10`(%).

#### `--percentile=<n>`

Specifies the confidence value as a percentile that will be used to determine
if a measurement is stable. For example, a value of `85` indicates that the
85th percentile latency will be used to determine stability. The percentile
will also be reported in the results.

Default is `-1` indicating that the average latency is used to determine
stability.

#### `-r <n>`
#### `--max-trials=<n>`

Specifies the maximum number of measurements when attempting to reach stability
of inferences per second and latency for each concurrency or request rate
during the search. Perf Analyzer will terminate if the measurement is still
unstable after the maximum number of trials.

Default is `10`.

#### `--concurrency-range=<start:end:step>`

Specifies the range of concurrency levels covered by Perf Analyzer. Perf
Analyzer will start from the concurrency level of 'start' and go until 'end'
with a stride of 'step'.

Default of 'start', 'end', and 'step' are `1`. If 'end' is not specified then
Perf Analyzer will run for a single concurrency level determined by 'start'. If
'end' is set as `0`, then the concurrency limit will be incremented by 'step'
until the latency threshold is met. 'end' and `--latency-threshold` cannot
both be `0`. 'end' cannot be `0` for sequence models while using asynchronous
mode.

#### `--periodic-concurrency-range=<start:end:step>`

Specifies the range of concurrency levels in the similar but slightly different
manner as the `--concurrency-range`. Perf Analyzer will start from the
concurrency level of 'start' and increase by 'step' each time. Unlike
`--concurrency-range`, the 'end' indicates the *total* number of concurrency
since the 'start' (including) and will stop increasing once the cumulative
number of concurrent requests has reached the 'end'. The user can specify
*when* to periodically increase the concurrency level using the
`--request-period` option. The concurrency level will periodically increase for
every `n`-th response specified by `--request-period`. Since this disables
stability check in Perf Analyzer and reports response timestamps only, the user
must provide `--profile-export-file` to specify where to dump all the measured
timestamps.

The default values of 'start', 'end', and 'step' are `1`.

#### `--request-period=<n>`

Specifies the number of responses that each request must receive before new,
concurrent requests are sent when `--periodic-concurrency-range` is specified.

Default value is `10`.

#### `--request-parameter=<name:value:type>`

Specifies a custom parameter that can be sent to a Triton backend as part of
the request. For example, providing '--request-parameter max_tokens:256:int'
to the command line will set an additional parameter 'max_tokens' of type
'int' to 256 as part of the request. The --request-parameter may be specified
multiple times for different custom parameters.

Valid `type` values are: `bool`, `int`, and `string`.

> **NOTE**
>
> The `--request-parameter` is currently only supported by gRPC protocol.

#### `--request-rate-range=<start:end:step>`

Specifies the range of request rates for load generated by Perf Analyzer. This
option can take floating-point values. The search along the request rate range
is enabled only when using this option.

If not specified, then Perf Analyzer will search along the concurrency range.
Perf Analyzer will start from the request rate of 'start' and go until 'end'
with a stride of 'step'. Default values of 'start', 'end' and 'step' are all
`1.0`. If 'end' is not specified, then Perf Analyzer will run for a single
request rate as determined by 'start'. If 'end' is set as `0.0`, then the
request rate will be incremented by 'step' until the latency threshold is met.
'end' and `--latency-threshold` can not be both `0`.

#### `--request-distribution=[constant|poisson]`

Specifies the time interval distribution between dispatching inference requests
to the server. Poisson distribution closely mimics the real-world work load on
a server. This option is ignored if not using `--request-rate-range`.

Default is `constant`.

#### `-l <n>`
#### `--latency-threshold=<n>`

Specifies the limit on the observed latency, in milliseconds. Perf Analyzer
will terminate the concurrency or request rate search once the measured latency
exceeds this threshold.

Default is `0` indicating that Perf Analyzer will run for the entire
concurrency or request rate range.

#### `--binary-search`

Enables binary search on the specified search range (concurrency or request
rate). This option requires 'start' and 'end' to be expilicitly specified in
the concurrency range or request rate range. When using this option, 'step' is
more like the precision. When the 'step' is lower, there are more iterations
along the search path to find suitable convergence.

When `--binary-search` is not specified, linear search is used.

#### `--request-intervals=<path>`

Specifies a path to a file containing time intervals in microseconds. Each time
interval should be in a new line. Perf Analyzer will try to maintain time
intervals between successive generated requests to be as close as possible in
this file. This option can be used to apply custom load to server with a
certain pattern of interest. Perf Analyzer will loop around the file if the
duration of execution exceeds the amount of time specified by the intervals.
This option can not be used with `--request-rate-range` or
`--concurrency-range`.

#### `--max-threads=<n>`

Specifies the maximum number of threads that will be created for providing
desired concurrency or request rate. However, when running in synchronous mode
with `--concurrency-range` having explicit 'end' specification, this value will
be ignored.

Default is `4` if `--request-rate-range` is specified, otherwise default is
`16`.

## Sequence Model Options

#### `--num-of-sequences=<n>`

Specifies the number of concurrent sequences for sequence models. This option
is ignored when `--request-rate-range` is not specified.

Default is `4`.

#### `--sequence-length=<n>`

Specifies the base length of a sequence used for sequence models. A sequence
with length X will be composed of X requests to be sent as the elements in the
sequence. The actual length of the sequencewill be within +/- Y% of the base
length, where Y defaults to 20% and is customizable via
`--sequence-length-variation`. If sequence length is unspecified and input data
is provided, the sequence length will be the number of inputs in the
user-provided input data.

Default is `20`.

#### `--sequence-length-variation=<n>`

Specifies the percentage variation in length of sequences. This option is only
valid when not using user-provided input data or when `--sequence-length` is
specified while using user-provided input data.

Default is `20`(%).

#### `--sequence-id-range=<start:end>`

Specifies the range of sequence IDs used by Perf Analyzer. Perf Analyzer will
start from the sequence ID of 'start' and go until 'end' (excluded). If 'end'
is not specified then Perf Analyzer will generate new sequence IDs without
bounds. If 'end' is specified and the concurrency setting may result in
maintaining a number of sequences more than the range of available sequence
IDs, Perf Analyzer will exit with an error due to possible sequence ID
collisions.

The default for 'start is `1`, and 'end' is not specified (no bounds).

#### `--serial-sequences`

Enables the serial sequence mode where a maximum of one request is live per sequence.
Note: It is possible that this mode can cause the request rate mode to not achieve the
desired rate, especially if num-of-sequences is too small.

## Input Data Options

#### `--input-data=[zero|random|<path>]`

Specifies type of data that will be used for input in inference requests. The
available options are `zero`, `random`, and a path to a directory or a JSON
file.

When pointing to a JSON file, the user must adhere to the format described in
the [input data documentation](input_data.md). By specifying JSON data, users
can control data used with every request. Multiple data streams can be specified
for a sequence model, and Perf Analyzer will select a data stream in a
round-robin fashion for every new sequence. Multiple JSON files can also be
provided (`--input-data json_file1.json --input-data json_file2.json` and so on)
and Perf Analyzer will append data streams from each file. When using
`--service-kind=torchserve`, make sure this option points to a JSON file.

If the option is path to a directory then the directory must contain a binary
text file for each non-string/string input respectively, named the same as the
input. Each file must contain the data required for that input for a batch-1
request. Each binary file should contain the raw binary representation of the
input in row-major order for non-string inputs. The text file should contain
all strings needed by batch-1, each in a new line, listed in row-major order.

Default is `random`.

#### `-b <n>`

Specifies the batch size for each request sent.

Default is `1`.

#### `--shape=<string>`

Specifies the shape used for the specified input. The argument must be
specified as 'name:shape' where the shape is a comma-separated list for
dimension sizes. For example `--shape=input_name:1,2,3` indicates that the
input `input_name` has tensor shape [ 1, 2, 3 ]. `--shape` may be specified
multiple times to specify shapes for different inputs.

#### `--string-data=<string>`

Specifies the string to initialize string input buffers. Perf Analyzer will
replicate the given string to build tensors of required shape.
`--string-length` will not have any effect. This option is ignored if
`--input-data` points to a JSON file or directory.

#### `--string-length=<n>`

Specifies the length of the random strings to be generated by Perf Analyzer
for string input. This option is ignored if `--input-data` points to a
JSON file or directory.

Default is `128`.

#### `--shared-memory=[none|system|cuda]`

Specifies the type of the shared memory to use for input and output data.

Default is `none`.

#### `--output-shared-memory-size=<n>`

Specifies The size, in bytes, of the shared memory region to allocate per
output tensor. Only needed when one or more of the outputs are of string type
and/or variable shape. The value should be larger than the size of the largest
output tensor that the model is expected to return. Perf Analyzer will use the
following formula to calculate the total shared memory to allocate:
output_shared_memory_size * number_of_outputs * batch_size.

Default is `102400` (100 KB).

#### `--input-tensor-format=[binary|json]`

Specifies the Triton inference request input tensor format. Only valid when HTTP
protocol is used.

Default is `binary`.

#### `--output-tensor-format=[binary|json]`

Specifies the Triton inference response output tensor format. Only valid when
HTTP protocol is used.

Default is `binary`.

## Request Options

#### `-i [http|grpc]`

Specifies the communication protocol to use. The available protocols are HTTP
and gRPC.

Default is `http`.

#### `-a`
#### `--async`

Enables asynchronous mode in Perf Analyzer.

By default, Perf Analyzer will use a synchronous request API for inference.
However, if the model is sequential, then the default mode is asynchronous.
Specify `--sync` to operate sequential models in synchronous mode. In
synchronous mode, Perf Analyzer will start threads equal to the concurrency
level. Use asynchronous mode to limit the number of threads, yet maintain the
concurrency.

#### `--sync`

Enables synchronous mode in Perf Analyzer. Can be used to operate Perf
Analyzer with sequential model in synchronous mode.

#### `--streaming`

Enables the use of streaming API. This option is only valid with gRPC protocol.

#### `-H <string>`

Specifies the header that will be added to HTTP requests (ignored for gRPC
requests). The header must be specified as 'Header:Value'. `-H` may be
specified multiple times to add multiple headers.

#### `--grpc-compression-algorithm=[none|gzip|deflate]`

Specifies the compression algorithm to be used by gRPC when sending requests.
Only supported when gRPC protocol is being used.

Default is `none`.

## Server Options

#### `-u <url>`

Specifies the URL for the server.

Default is `localhost:8000` when using `--service-kind=triton` with HTTP.
Default is `localhost:8001` when using `--service-kind=triton` with gRPC.
Default is `localhost:8500` when using `--service-kind=tfserving`.

#### `--ssl-grpc-use-ssl`

Enables usage of an encrypted channel to the server.

#### `--ssl-grpc-root-certifications-file=<path>`

Specifies the path to file containing the PEM encoding of the server root
certificates.

#### `--ssl-grpc-private-key-file=<path>`

Specifies the path to file containing the PEM encoding of the client's private
key.

#### `--ssl-grpc-certificate-chain-file=<path>`

Specifies the path to file containing the PEM encoding of the client's
certificate chain.

#### `--ssl-https-verify-peer=[0|1]`

Specifies whether to verify the peer's SSL certificate. See
https://curl.se/libcurl/c/CURLOPT_SSL_VERIFYPEER.html for the meaning of each
value.

Default is `1`.

#### `--ssl-https-verify-host=[0|1|2]`

Specifies whether to verify the certificate's name against host. See
https://curl.se/libcurl/c/CURLOPT_SSL_VERIFYHOST.html for the meaning of each
value.

Default is `2`.

#### `--ssl-https-ca-certificates-file=<path>`

Specifies the path to Certificate Authority (CA) bundle.

#### `--ssl-https-client-certificate-file=<path>`

Specifies the path to the SSL client certificate.

#### `--ssl-https-client-certificate-type=[PEM|DER]`

Specifies the type of the client SSL certificate.

Default is `PEM`.

#### `--ssl-https-private-key-file=<path>`

Specifies the path to the private keyfile for TLS and SSL client cert.

#### `--ssl-https-private-key-type=[PEM|DER]`

Specifies the type of the private key file.

Default is `PEM`.

#### `--triton-server-directory=<path>`

Specifies the Triton server install path. Required by and only used when C API
is used (`--service-kind=triton_c_api`).

Default is `/opt/tritonserver`.

#### `--model-repository=<path>`

Specifies the model repository directory path for loading models. Required by
and only used when C API is used (`--service-kind=triton_c_api`).

## Prometheus Metrics Options

#### `--collect-metrics`

Enables the collection of server-side inference server metrics. Perf Analyzer
will output metrics in the CSV file generated with the `-f` option. Only valid
when `--verbose-csv` option also used.

#### `--metrics-url=<url>`

Specifies the URL to query for server-side inference server metrics.

Default is `localhost:8002/metrics`.

#### `--metrics-interval=<n>`

Specifies how often within each measurement window, in milliseconds, Perf
Analyzer should query for server-side inference server metrics.

Default is `1000`.

## Report Options

#### `-f <path>`

Specifies the path that the latency report file will be generated at.

When `-f` is not specified, a latency report will not be generated.

#### `--profile-export-file <path>`

Specifies the path that the profile export will be generated at.

When `--profile-export-file` is not specified, a profile export will not be
generated.

#### `--verbose-csv`

Enables additional information being output to the CSV file generated by Perf
Analyzer.

## Trace Options

#### `--trace-level=[OFF|TIMESTAMPS|TENSORS]`

Specifies a trace level. `OFF` disables tracing. `TIMESTAMPS` traces
timestamps. `TENSORS` traces tensors. It may be specified multiple times to
trace multiple information. Only used for `--service-kind=triton`.

Default is `OFF`.

#### `--trace-rate=<n>`

Specifies the trace sampling rate (traces per second).

Default is `1000`.

#### `--trace-count=<n>`

Specifies the number of traces to be sampled. If the value is `-1`, the number
of traces to be sampled will not be limited.

Default is `-1`.

#### `--log-frequency=<n>`

Specifies the trace log frequency. If the value is `0`, Triton will only log
the trace output to the trace file when shutting down.
Otherwise, Triton will log the trace output to `<trace-file>`.<idx> when it
collects the specified number of traces.
For example, if the trace file is `trace_file.log`, and if the log
frequency is `100`, when Triton collects the 100th trace, it logs the traces
to file `trace_file.log.0`, and when it collects the 200th trace, it logs the
101st to the 200th traces to file `trace_file.log.1`.

Default is `0`.

## Deprecated Options

#### `--data-directory=<path>`

**DEPRECATED**

Alias for `--input-data=<path>` where `<path>` is the path to a directory. See
`--input-data` option documentation for details.

#### `-c <n>`

**DEPRECATED**

Specifies the maximum concurrency that Perf Analyzer will search up to. Cannot
be used with `--concurrency-range`.

#### `-d`

**DEPRECATED**

Enables dynamic concurrency mode. Perf Analyzer will search along
concurrencies up to the maximum concurrency specified via `-c <n>`. Cannot be
used with `--concurrency-range`.

#### `-t <n>`

**DEPRECATED**

Specifies the number of concurrent requests. Cannot be used with
`--concurrency-range`.

Default is `1`.

#### `-z`

**DEPRECATED**

Alias for `--input-data=zero`. See `--input-data` option documentation for
details.
