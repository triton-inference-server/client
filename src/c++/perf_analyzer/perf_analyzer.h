// Copyright 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#pragma once

#include <getopt.h>
#include <signal.h>
#include <algorithm>
#include "command_line_parser.h"
#include "concurrency_manager.h"
#include "custom_load_manager.h"
#include "inference_profiler.h"
#include "model_parser.h"
#include "mpi_utils.h"
#include "perf_utils.h"

// Perf Analyzer provides various metrics to measure the performance of
// the inference server. It can either be used to measure the throughput,
// latency and time distribution under specific setting (i.e. fixed batch size
// and fixed concurrent requests), or be used to generate throughput-latency
// data point under dynamic setting (i.e. collecting throughput-latency data
// under different load level).
//
// The following data is collected and used as part of the metrics:
// - Throughput (infer/sec):
//     The number of inference processed per second as seen by the analyzer.
//     The number of inference is measured by the multiplication of the number
//     of requests and their batch size. And the total time is the time elapsed
//     from when the analyzer starts sending requests to when it received
//     all responses.
// - Latency (usec):
//     The average elapsed time between when a request is sent and
//     when the response for the request is received. If 'percentile' flag is
//     specified, the selected percentile value will be reported instead of
//     average value.
//
// Perf Analyzer determines the stability of throughput and latency by observing
// measurements in different trials. If the latency and throughput, are within
// the stability percentage (see --stability-percentage option) Perf Analyzer
// will report the average of the throughput and latency numbers observed in the
// last three trials. All the measurements gathered during the last three trials
// is aggregated to generate a single report. The number of total requests is
// the sum of all the requests in the individual measurement windows.
//
// There are broadly three ways to load server for the data collection using
// perf_analyzer:
// - Maintaining Target Concurrency:
//     In this setting, the analyzer will maintain a target number of concurrent
//     requests sent to the server (see --concurrency-range option) while
//     taking measurements.
//     The number of requests will be the total number of requests sent within
//     the time interval for measurement (see --measurement-interval option) and
//     the latency will be the average latency across all requests.
//
//     Besides throughput and latency, which is measured on client side,
//     the following data measured by the server will also be reported
//     in this setting:
//     - Concurrent request: the number of concurrent requests as specified
//         in --concurrency-range option. Note, for running perf analyzer for
//         a single concurrency, user must specify --concurrency-range
//         <'start'>, omitting 'end' and 'step' values.
//     - Batch size: the batch size of each request as specified in -b option
//     - Inference count: batch size * number of inference requests
//     - Cumulative time: the total time between request received and
//         response sent on the requests sent by perf analyzer.
//     - Average Cumulative time: cumulative time / number of inference requests
//     - Compute time: the total time it takes to run inferencing including time
//         copying input tensors to GPU memory, time executing the model,
//         and time copying output tensors from GPU memory for the requests
//         sent by perf analyzer.
//     - Average compute time: compute time / number of inference requests
//     - Queue time: the total time it takes to wait for an available model
//         instance for the requests sent by perf analyzer.
//     - Average queue time: queue time / number of inference requests
//     If all fields of --concurrency-range are specified, the analyzer will
//     perform the following procedure:
//       1. Follows the procedure in fixed concurrent request mode using
//          k concurrent requests (k starts at 'start').
//       2. Gathers data reported from step 1.
//       3. Increases k by 'step' and repeats step 1 and 2 until latency from
//          current iteration exceeds latency threshold (see --latency-threshold
//          option) or concurrency level reaches 'end'. Note, by setting
//          --latency-threshold or 'end' to 0 the effect of each threshold can
//          be removed. However, both can not be 0 simultaneously.
//     At each iteration, the data mentioned in fixed concurrent request mode
//     will be reported. Besides that, after the procedure above, a collection
//     of "throughput, latency, concurrent request count" tuples will be
//     reported in increasing load level order.
//
// - Maintaining Target Request Rate:
//     This mode is enabled only when --request-rate-range option is specified.
//     Unlike above, here the analyzer will try to maintain a target rate of
//     requests issued to the server while taking measurements. Rest of the
//     behaviour of analyzer is identical as above. It is important to note that
//     even though over a  sufficiently large interval the rate of requests
//     will tend to the target request rate, the actual request rate for a small
//     time interval will depend upon the selected request distribution
//     (--request-distribution). For 'constant' request distribution the time
//     interval between successive requests is maintained to be constant, hence
//     request rate is constant over time. However, 'poisson' request
//     distribution varies the time interval between successive requests such
//     that there are periods of bursts and nulls in request generation.
//     Additionally, 'poisson' distribution mimics the real-world traffic and
//     can be used to obtain measurements for a realistic-load.
//     With each request-rate, the analyzer also reports the 'Delayed Request
//     Count' which gives an idea of how many requests missed their schedule as
//     specified by the distribution. Users can use --max-threads to increase
//     the number of threads which might help in dispatching requests as per
//     the schedule. Also note that a very large number of threads might be
//     counter-productive with most of the time being spent on context-switching
//     the threads.
//
// - Following User Provided Request Delivery Schedule:
//     This mode is enabled only when --request-intervals option is specified.
//     In this case, analyzer will try to dispatch the requests to the server
//     with time intervals between successive requests specified in a user
//     provided file. This file should contain time intervals in microseconds in
//     each new line. Analyzer will loop around the values to produce a
//     consistent load for measurements. Once, the readings are stabilized then
//     the final statistics will be reported. The statistics will include
//     'Delayed Request Count' for the requests that missed their schedule. As
//     described before, users can tune --max-threads to allow analyzer in
//     keeping up with the schedule. This mode will help user in analyzing the
//     performance of the server under different custom settings which may be of
//     interest.
//
// By default, perf_analyzer will maintain target concurrency while measuring
// the performance.
//
// Options:
// -b: batch size for each request sent.
// --concurrency-range: The range of concurrency levels perf_analyzer will use.
//    A concurrency level indicates the number of concurrent requests in queue.
// --request-rate-range: The range of request rates perf_analyzer will use to
//    load the server.
// --request-intervals: File containing time intervals (in microseconds) to use
//    between successive requests.
// --latency-threshold: latency threshold in msec.
// --measurement-interval: time interval for each measurement window in msec.
// --async: Enables Asynchronous inference calls.
// --binary-search: Enables binary search within the specified range.
// --request-distribution: Allows user to specify the distribution for selecting
//    the time intervals between the request dispatch.
//
// For detail of the options not listed, please refer to the usage.
//
class PerfAnalyzer {
 public:
  PerfAnalyzer(pa::PAParamsPtr params);
  virtual ~PerfAnalyzer(){};

  // Main runner function for Perf Analyzer.
  void Run();

 private:
  pa::PAParamsPtr params_;
  std::unique_ptr<pa::InferenceProfiler> profiler_;
  std::unique_ptr<cb::ClientBackend> backend_;
  std::shared_ptr<pa::ModelParser> parser_;
  std::vector<pa::PerfStatus> summary_;

  //
  // Helper methods
  //

  // Parse the options out of the command line argument
  //
  void CreateAnalyzerObjects();
  void PrerunReport();
  void Profile();
  void WriteReport();
  void Finalize();
};
