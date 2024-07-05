// Copyright 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
//

#include "command_line_parser.h"

#include <getopt.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <string>

#include "perf_analyzer_exception.h"

namespace triton { namespace perfanalyzer {

PAParamsPtr
CLParser::Parse(int argc, char** argv)
{
  ParseCommandLine(argc, argv);
  VerifyOptions();

  return params_;
}

std::vector<std::string>
SplitString(const std::string& str, const std::string& delimiter = ":")
{
  std::vector<std::string> substrs;
  size_t pos = 0;
  while (pos != std::string::npos) {
    size_t colon_pos = str.find(":", pos);
    substrs.push_back(str.substr(pos, colon_pos - pos));
    if (colon_pos == std::string::npos) {
      pos = colon_pos;
    } else {
      pos = colon_pos + 1;
    }
  }
  return substrs;
}

void
ToLowerCase(std::string& s)
{
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
    return std::tolower(c);
  });
}

// Used to format the usage message
std::string
CLParser::FormatMessage(std::string str, int offset) const
{
  int width = 60;
  int current_pos = offset;
  while (current_pos + width < int(str.length())) {
    int n = str.rfind(' ', current_pos + width);
    if (n != int(std::string::npos)) {
      str.replace(n, 1, "\n\t ");
      current_pos += (width + 10);
    }
  }
  return str;
}

void
CLParser::Usage(const std::string& msg)
{
  if (!msg.empty()) {
    std::cerr << "Error: " << msg << std::endl;
  }

  std::cerr << "Usage: " << argv_[0] << " [options]" << std::endl;
  std::cerr << "==== SYNOPSIS ====\n \n";
  std::cerr << "\t--version " << std::endl;
  std::cerr << "\t-m <model name>" << std::endl;
  std::cerr << "\t-x <model version>" << std::endl;
  std::cerr << "\t--bls-composing-models <string>" << std::endl;
  std::cerr << "\t--model-signature-name <model signature name>" << std::endl;
  std::cerr
      << "\t--service-kind "
         "<\"triton\"|\"openai\"|\"tfserving\"|\"torchserve\"|\"triton_c_api\">"
      << std::endl;
  std::cerr << "\t--endpoint <string>" << std::endl;
  std::cerr << "\t-v" << std::endl;
  std::cerr << std::endl;
  std::cerr << "I. MEASUREMENT PARAMETERS: " << std::endl;
  std::cerr << "\t--async (-a)" << std::endl;
  std::cerr << "\t--sync" << std::endl;
  std::cerr << "\t--measurement-interval (-p) <measurement window (in msec)>"
            << std::endl;
  std::cerr << "\t--concurrency-range <start:end:step>" << std::endl;
  std::cerr << "\t--periodic-concurrency-range <start:end:step>" << std::endl;
  std::cerr << "\t--request-period <number of responses>" << std::endl;
  std::cerr << "\t--request-rate-range <start:end:step>" << std::endl;
  std::cerr << "\t--request-distribution <\"poisson\"|\"constant\">"
            << std::endl;
  std::cerr << "\t--request-intervals <path to file containing time intervals "
               "in microseconds>"
            << std::endl;
  std::cerr << "\t--serial-sequences" << std::endl;
  std::cerr << "\t--binary-search" << std::endl;
  std::cerr << "\t--num-of-sequences <number of concurrent sequences>"
            << std::endl;
  std::cerr << "\t--latency-threshold (-l) <latency threshold (in msec)>"
            << std::endl;
  std::cerr << "\t--max-threads <thread counts>" << std::endl;
  std::cerr << "\t--stability-percentage (-s) <deviation threshold for stable "
               "measurement (in percentage)>"
            << std::endl;
  std::cerr << "\t--max-trials (-r)  <maximum number of measurements for each "
               "profiling>"
            << std::endl;
  std::cerr << "\t--percentile <percentile>" << std::endl;
  std::cerr << "\t--request-count <number of requests>" << std::endl;
  std::cerr << "\tDEPRECATED OPTIONS" << std::endl;
  std::cerr << "\t-t <number of concurrent requests>" << std::endl;
  std::cerr << "\t-c <maximum concurrency>" << std::endl;
  std::cerr << "\t-d" << std::endl;
  std::cerr << std::endl;
  std::cerr << "II. INPUT DATA OPTIONS: " << std::endl;
  std::cerr << "\t-b <batch size>" << std::endl;
  std::cerr << "\t--input-data <\"zero\"|\"random\"|<path>>" << std::endl;
  std::cerr << "\t--shared-memory <\"system\"|\"cuda\"|\"none\">" << std::endl;
  std::cerr << "\t--output-shared-memory-size <size in bytes>" << std::endl;
  std::cerr << "\t--shape <name:shape>" << std::endl;
  std::cerr << "\t--sequence-length <length>" << std::endl;
  std::cerr << "\t--sequence-length-variation <variation>" << std::endl;
  std::cerr << "\t--sequence-id-range <start:end>" << std::endl;
  std::cerr << "\t--string-length <length>" << std::endl;
  std::cerr << "\t--string-data <string>" << std::endl;
  std::cerr << "\t--input-tensor-format [binary|json]" << std::endl;
  std::cerr << "\t--output-tensor-format [binary|json]" << std::endl;
  std::cerr << "\tDEPRECATED OPTIONS" << std::endl;
  std::cerr << "\t-z" << std::endl;
  std::cerr << "\t--data-directory <path>" << std::endl;
  std::cerr << std::endl;
  std::cerr << "III. SERVER DETAILS: " << std::endl;
  std::cerr << "\t-u <URL for inference service>" << std::endl;
  std::cerr << "\t-i <Protocol used to communicate with inference service>"
            << std::endl;
  std::cerr << "\t--ssl-grpc-use-ssl <bool>" << std::endl;
  std::cerr << "\t--ssl-grpc-root-certifications-file <path>" << std::endl;
  std::cerr << "\t--ssl-grpc-private-key-file <path>" << std::endl;
  std::cerr << "\t--ssl-grpc-certificate-chain-file <path>" << std::endl;
  std::cerr << "\t--ssl-https-verify-peer <number>" << std::endl;
  std::cerr << "\t--ssl-https-verify-host <number>" << std::endl;
  std::cerr << "\t--ssl-https-ca-certificates-file <path>" << std::endl;
  std::cerr << "\t--ssl-https-client-certificate-file <path>" << std::endl;
  std::cerr << "\t--ssl-https-client-certificate-type <string>" << std::endl;
  std::cerr << "\t--ssl-https-private-key-file <path>" << std::endl;
  std::cerr << "\t--ssl-https-private-key-type <string>" << std::endl;
  std::cerr << std::endl;
  std::cerr << "IV. OTHER OPTIONS: " << std::endl;
  std::cerr << "\t-f <filename for storing report in csv format>" << std::endl;
  std::cerr << "\t--profile-export-file <path>" << std::endl;
  std::cerr << "\t-H <HTTP header>" << std::endl;
  std::cerr << "\t--streaming" << std::endl;
  std::cerr << "\t--grpc-compression-algorithm <compression_algorithm>"
            << std::endl;
  std::cerr << "\t--trace-level" << std::endl;
  std::cerr << "\t--trace-rate" << std::endl;
  std::cerr << "\t--trace-count" << std::endl;
  std::cerr << "\t--log-frequency" << std::endl;
  std::cerr << "\t--collect-metrics" << std::endl;
  std::cerr << "\t--metrics-url" << std::endl;
  std::cerr << "\t--metrics-interval" << std::endl;
  std::cerr << std::endl;
  std::cerr << "==== OPTIONS ==== \n \n";

  std::cerr << FormatMessage(
                   " --version: print the current version of Perf Analyzer.",
                   18)
            << std::endl;

  std::cerr
      << std::setw(9) << std::left << " -m: "
      << FormatMessage(
             "This is a required argument and is used to specify the model"
             " against which to run perf_analyzer.",
             9)
      << std::endl;
  std::cerr << std::setw(9) << std::left << " -x: "
            << FormatMessage(
                   "The version of the above model to be used. If not specified"
                   " the most recent version (that is, the highest numbered"
                   " version) of the model will be used.",
                   9)
            << std::endl;
  std::cerr << FormatMessage(
                   " --model-signature-name: The signature name of the saved "
                   "model to use. Default value is \"serving_default\". This "
                   "option will be ignored if --service-kind is not "
                   "\"tfserving\".",
                   18)
            << std::endl;

  std::cerr
      << FormatMessage(
             " --service-kind: Describes the kind of service perf_analyzer to "
             "generate load for. The options are \"triton\", \"openai\", "
             "\"triton_c_api\", \"tfserving\" and \"torchserve\". Default "
             "value is \"triton\". Note in order to use \"openai\" you must "
             "specify an endpoint via --endpoint. "
             "Note in order to use \"torchserve\" backend --input-data option "
             "must point to a json file holding data in the following format "
             "{\"data\" : [{\"TORCHSERVE_INPUT\" : [\"<complete path to the "
             "content file>\"]}, {...}...]}. The type of file here will depend "
             "on the model. In order to use \"triton_c_api\" you must specify "
             "the Triton server install path and the model repository path via "
             "the --triton-server-directory and --model-repository flags",
             18)
      << std::endl;

  std::cerr
      << FormatMessage(
             " --endpoint: Describes what endpoint to send requests to on the "
             "server. This is required when using \"openai\" service-kind, and "
             "is ignored for all other cases. Currently only "
             "\"v1/chat/completions\" is confirmed to work.",
             18)
      << std::endl;

  std::cerr << std::setw(9) << std::left
            << " -v: " << FormatMessage("Enables verbose mode.", 9)
            << std::endl;
  std::cerr << std::setw(9) << std::left
            << " -v -v: " << FormatMessage("Enables extra verbose mode.", 9)
            << std::endl;
  std::cerr << std::endl;
  std::cerr << "I. MEASUREMENT PARAMETERS: " << std::endl;
  std::cerr
      << FormatMessage(
             " --async (-a): Enables asynchronous mode in perf_analyzer. "
             "By default, perf_analyzer will use synchronous API to "
             "request inference. However, if the model is sequential "
             "then default mode is asynchronous. Specify --sync to "
             "operate sequential models in synchronous mode. In synchronous "
             "mode, perf_analyzer will start threads equal to the concurrency "
             "level. Use asynchronous mode to limit the number of threads, yet "
             "maintain the concurrency.",
             18)
      << std::endl;
  std::cerr << FormatMessage(
                   " --sync: Force enables synchronous mode in perf_analyzer. "
                   "Can be used to operate perf_analyzer with sequential model "
                   "in synchronous mode.",
                   18)
            << std::endl;
  std::cerr
      << FormatMessage(
             " --measurement-interval (-p): Indicates the time interval used "
             "for each measurement in milliseconds. The perf analyzer will "
             "sample a time interval specified by -p and take measurement over "
             "the requests completed within that time interval. The default "
             "value is 5000 msec.",
             18)
      << std::endl;
  std::cerr << FormatMessage(
                   " --measurement-mode <\"time_windows\"|\"count_windows\">: "
                   "Indicates the mode used for stabilizing measurements."
                   " \"time_windows\" will create windows such that the length "
                   "of each window is equal to --measurement-interval. "
                   "\"count_windows\" will create "
                   "windows such that there are at least "
                   "--measurement-request-count requests in each window.",
                   18)
            << std::endl;
  std::cerr
      << FormatMessage(
             " --measurement-request-count: "
             "Indicates the minimum number of requests to be collected in each "
             "measurement window when \"count_windows\" mode is used. This "
             "mode can "
             "be enabled using the --measurement-mode flag.",
             18)
      << std::endl;
  std::cerr
      << FormatMessage(
             " --concurrency-range <start:end:step>: Determines the range of "
             "concurrency levels covered by the perf_analyzer. The "
             "perf_analyzer "
             "will start from the concurrency level of 'start' and go till "
             "'end' with a stride of 'step'. The default value of 'end' and "
             "'step' are 1. If 'end' is not specified then perf_analyzer will "
             "run for a single concurrency level determined by 'start'. If "
             "'end' is set as 0, then the concurrency limit will be "
             "incremented by 'step' till latency threshold is met. 'end' and "
             "--latency-threshold can not be both 0 simultaneously. 'end' can "
             "not be 0 for sequence models while using asynchronous mode.",
             18)
      << std::endl;
  std::cerr
      << FormatMessage(
             " --periodic-concurrency-range <start:end:step>: Determines the "
             "range of concurrency levels in the similar but slightly "
             "different manner as the --concurrency-range. Perf Analyzer will "
             "start from the concurrency level of 'start' and increase by "
             "'step' each time. Unlike --concurrency-range, the 'end' "
             "indicates the *total* number of concurrency since the 'start' "
             "(including) and will stop increasing once the cumulative number "
             "of concurrent requests has reached the 'end'. The user can "
             "specify *when* to periodically increase the concurrency level "
             "using the --request-period option. The concurrency level will "
             "periodically increase for every n-th response specified by "
             "--request-period. Since this disables stability check in Perf "
             "Analyzer and reports response timestamps only, the user must "
             "provide --profile-export-file to specify where to dump all the "
             "measured timestamps. The default values of 'start', 'end', and "
             "'step' are 1.",
             18)
      << std::endl;
  std::cerr
      << FormatMessage(
             " --request-period <n>: Indicates the number of responses that "
             "each request must receive before new, concurrent requests are "
             "sent when --periodic-concurrency-range is specified. Default "
             "value is 10.",
             18)
      << std::endl;
  std::cerr
      << FormatMessage(
             " --request-parameter <name:value:type>: Specifies a custom "
             "parameter that can be sent to a Triton backend as part of the "
             "request. For example, providing '--request-parameter "
             "max_tokens:256:int' to the command line will set an additional "
             "parameter 'max_tokens' of type 'int' to 256 as part of the "
             "request. The --request-parameter may be specified multiple times "
             "for different custom parameters.",
             18)
      << std::endl;
  std::cerr
      << FormatMessage(
             " --request-rate-range <start:end:step>: Determines the range of "
             "request rates for load generated by analyzer. This option can "
             "take floating-point values. The search along the request rate "
             "range is enabled only when using this option. If not specified, "
             "then analyzer will search along the concurrency-range. The "
             "perf_analyzer will start from the request rate of 'start' and go "
             "till 'end' with a stride of 'step'. The default values of "
             "'start', 'end' and 'step' are all 1.0. If 'end' is not specified "
             "then perf_analyzer will run for a single request rate as "
             "determined by 'start'. If 'end' is set as 0.0, then the request "
             "rate will be incremented by 'step' till latency threshold is "
             "met. 'end' and --latency-threshold can not be both 0 "
             "simultaneously.",
             18)
      << std::endl;
  std::cerr
      << FormatMessage(
             " --request-distribution <\"poisson\"|\"constant\">: Specifies "
             "the time interval distribution between dispatching inference "
             "requests to the server. Poisson distribution closely mimics the "
             "real-world work load on a server. This option is ignored if not "
             "using --request-rate-range. By default, this option is set to be "
             "constant.",
             18)
      << std::endl;
  std::cerr
      << FormatMessage(
             " --request-intervals: Specifies a path to a file containing time "
             "intervals in microseconds. Each time interval should be in a new "
             "line. The analyzer will try to maintain time intervals between "
             "successive generated requests to be as close as possible in this "
             "file. This option can be used to apply custom load to server "
             "with a certain pattern of interest. The analyzer will loop "
             "around the file if the duration of execution exceeds to that "
             "accounted for by the intervals. This option can not be used with "
             "--request-rate-range or --concurrency-range.",
             18)
      << std::endl;
  std::cerr
      << FormatMessage(
             " --binary-search: Enables the binary search on the specified "
             "search range. This option requires 'start' and 'end' to be "
             "expilicitly specified in the --concurrency-range or "
             "--request-rate-range. When using this option, 'step' is more "
             "like the precision. Lower the 'step', more the number of "
             "iterations along the search path to find suitable convergence. "
             "By default, linear search is used.",
             18)
      << std::endl;

  std::cerr << FormatMessage(
                   " --num-of-sequences: Sets the number of concurrent "
                   "sequences for sequence models. This option is ignored when "
                   "--request-rate-range is not specified. By default, its "
                   "value is 4.",
                   18)
            << std::endl;

  std::cerr
      << FormatMessage(
             " --latency-threshold (-l): Sets the limit on the observed "
             "latency. Analyzer will terminate the concurrency search once "
             "the measured latency exceeds this threshold. By default, "
             "latency threshold is set 0 and the perf_analyzer will run "
             "for entire --concurrency-range.",
             18)
      << std::endl;
  std::cerr
      << FormatMessage(
             " --max-threads: Sets the maximum number of threads that will be "
             "created for providing desired concurrency or request rate. "
             "However, when running"
             "in synchronous mode with concurrency-range having explicit 'end' "
             "specification,"
             "this value will be ignored. Default is 4 if --request-rate-range "
             "is specified otherwise default is 16.",
             18)
      << std::endl;
  std::cerr
      << FormatMessage(
             " --stability-percentage (-s): Indicates the allowed variation in "
             "latency measurements when determining if a result is stable. The "
             "measurement is considered as stable if the ratio of max / min "
             "from the recent 3 measurements is within (stability percentage)% "
             "in terms of both infer per second and latency. Default is "
             "10(%).",
             18)
      << std::endl;
  std::cerr << FormatMessage(
                   " --max-trials (-r): Indicates the maximum number of "
                   "measurements for each concurrency level visited during "
                   "search. The perf analyzer will take multiple measurements "
                   "and report the measurement until it is stable. The perf "
                   "analyzer will abort if the measurement is still unstable "
                   "after the maximum number of measurements. The default "
                   "value is 10.",
                   18)
            << std::endl;
  std::cerr
      << FormatMessage(
             " --percentile: Indicates the confidence value as a percentile "
             "that will be used to determine if a measurement is stable. For "
             "example, a value of 85 indicates that the 85th percentile "
             "latency will be used to determine stability. The percentile will "
             "also be reported in the results. The default is -1 indicating "
             "that the average latency is used to determine stability",
             18)
      << std::endl;
  std::cerr
      << FormatMessage(
             " --request-count: Specifies a total number of requests to "
             "use for measurement. The default is 0, which means that there is "
             "no request count and the measurement will proceed using windows "
             "until stabilization is detected.",
             18)
      << std::endl;
  std::cerr << FormatMessage(
                   " --serial-sequences: Enables serial sequence mode "
                   "where a maximum of one request is outstanding at a time "
                   "for any given sequence. The default is false.",
                   18)
            << std::endl;
  std::cerr << std::endl;
  std::cerr << "II. INPUT DATA OPTIONS: " << std::endl;
  std::cerr << std::setw(9) << std::left
            << " -b: " << FormatMessage("Batch size for each request sent.", 9)
            << std::endl;
  std::cerr
      << FormatMessage(
             " --input-data: Select the type of data that will be used "
             "for input in inference requests. The available options are "
             "\"zero\", \"random\", path to a directory or a json file. If the "
             "option is path to a directory then the directory must "
             "contain a binary/text file for each non-string/string input "
             "respectively, named the same as the input. Each "
             "file must contain the data required for that input for a batch-1 "
             "request. Each binary file should contain the raw binary "
             "representation of the input in row-major order for non-string "
             "inputs. The text file should contain all strings needed by "
             "batch-1, each in a new line, listed in row-major order. When "
             "pointing to a json file, user must adhere to the format "
             "described in the Performance Analyzer documentation. By "
             "specifying json data users can control data used with every "
             "request. Multiple data streams can be specified for a sequence "
             "model and the analyzer will select a data stream in a "
             "round-robin fashion for every new sequence. Multiple json files "
             "can also be provided (--input-data json_file1 --input-data "
             "json-file2 and so on) and the analyzer will append data streams "
             "from each file. When using --service-kind=torchserve make sure "
             "this option points to a json file. Default is \"random\".",
             18)
      << std::endl;
  std::cerr << FormatMessage(
                   " --shared-memory <\"system\"|\"cuda\"|\"none\">: Specifies "
                   "the type of the shared memory to use for input and output "
                   "data. Default is none.",
                   18)
            << std::endl;

  std::cerr
      << FormatMessage(
             " --output-shared-memory-size: The size in bytes of the shared "
             "memory region to allocate per output tensor. Only needed when "
             "one or more of the outputs are of string type and/or variable "
             "shape. The value should be larger than the size of the largest "
             "output tensor the model is expected to return. The analyzer will "
             "use the following formula to calculate the total shared memory "
             "to allocate: output_shared_memory_size * number_of_outputs * "
             "batch_size. Defaults to 100KB.",
             18)
      << std::endl;

  std::cerr << FormatMessage(
                   " --shape: The shape used for the specified input. The "
                   "argument must be specified as 'name:shape' where the shape "
                   "is a comma-separated list for dimension sizes, for example "
                   "'--shape input_name:1,2,3' indicate tensor shape [ 1, 2, 3 "
                   "]. --shape may be specified multiple times to specify "
                   "shapes for different inputs.",
                   18)
            << std::endl;
  std::cerr << FormatMessage(
                   " --sequence-length: Indicates the base length of a "
                   "sequence used for sequence models. A sequence with length "
                   "X will be composed of X requests to be sent as the "
                   "elements in the sequence. The actual length of the sequence"
                   "will be within +/- Y% of the base length, where Y defaults "
                   "to 20% and is customizable via "
                   "`--sequence-length-variation`. If sequence length is "
                   "unspecified and input data is provided, the sequence "
                   "length will be the number of inputs in the user-provided "
                   "input data. Default is 20.",
                   18)
            << std::endl;
  std::cerr << FormatMessage(
                   " --sequence-length-variation: The percentage variation in "
                   "length of sequences. This flag is only valid when "
                   "not using user-provided input data or when "
                   "`--sequence-length` is specified while using user-provided "
                   "input data. Default is 20.",
                   18)
            << std::endl;
  std::cerr
      << FormatMessage(
             " --sequence-id-range <start:end>: Determines the range of "
             "sequence id used by the perf_analyzer. The perf_analyzer "
             "will start from the sequence id of 'start' and go till "
             "'end' (excluded). If 'end' is not specified then perf_analyzer "
             "will use new sequence id without bounds. If 'end' is specified "
             "and the concurrency setting may result in maintaining a number "
             "of sequences more than the range of available sequence id, "
             "perf analyzer will exit with error due to possible sequence id "
             "collision. The default setting is start from sequence id 1 and "
             "without bounds",
             18)
      << std::endl;
  std::cerr << FormatMessage(
                   " --string-length: Specifies the length of the random "
                   "strings to be generated by the analyzer for string input. "
                   "This option is ignored if --input-data points to a "
                   "directory. Default is 128.",
                   18)
            << std::endl;
  std::cerr << FormatMessage(
                   " --string-data: If provided, analyzer will use this string "
                   "to initialize string input buffers. The perf analyzer will "
                   "replicate the given string to build tensors of required "
                   "shape. --string-length will not have any effect. This "
                   "option is ignored if --input-data points to a directory.",
                   18)
            << std::endl;
  std::cerr << FormatMessage(
                   " --input-tensor-format=[binary|json]: Specifies Triton "
                   "inference request input tensor format. Only valid when "
                   "HTTP protocol is used. Default is 'binary'.",
                   18)
            << std::endl;
  std::cerr << FormatMessage(
                   " --output-tensor-format=[binary|json]: Specifies Triton "
                   "inference response output tensor format. Only valid when "
                   "HTTP protocol is used. Default is 'binary'.",
                   18)
            << std::endl;
  std::cerr << std::endl;
  std::cerr << "III. SERVER DETAILS: " << std::endl;
  std::cerr << std::setw(38) << std::left << " -u: "
            << FormatMessage(
                   "Specify URL to the server. When using triton default is "
                   "\"localhost:8000\" if using HTTP and \"localhost:8001\" "
                   "if using gRPC. When using tfserving default is "
                   "\"localhost:8500\". ",
                   38)
            << std::endl;
  std::cerr << std::setw(38) << std::left << " -i: "
            << FormatMessage(
                   "The communication protocol to use. The available protocols "
                   "are gRPC and HTTP. Default is HTTP.",
                   38)
            << std::endl;
  std::cerr << std::setw(38) << std::left << " --ssl-grpc-use-ssl: "
            << FormatMessage(
                   "Bool (true|false) for whether "
                   "to use encrypted channel to the server. Default false.",
                   38)
            << std::endl;
  std::cerr << std::setw(38) << std::left
            << " --ssl-grpc-root-certifications-file: "
            << FormatMessage(
                   "Path to file containing the "
                   "PEM encoding of the server root certificates.",
                   38)
            << std::endl;
  std::cerr << std::setw(38) << std::left << " --ssl-grpc-private-key-file: "
            << FormatMessage(
                   "Path to file containing the "
                   "PEM encoding of the client's private key.",
                   38)
            << std::endl;
  std::cerr << std::setw(38) << std::left
            << " --ssl-grpc-certificate-chain-file: "
            << FormatMessage(
                   "Path to file containing the "
                   "PEM encoding of the client's certificate chain.",
                   38)
            << std::endl;
  std::cerr << std::setw(38) << std::left << " --ssl-https-verify-peer: "
            << FormatMessage(
                   "Number (0|1) to verify the "
                   "peer's SSL certificate. See "
                   "https://curl.se/libcurl/c/CURLOPT_SSL_VERIFYPEER.html for "
                   "the meaning of each value. Default is 1.",
                   38)
            << std::endl;
  std::cerr
      << std::setw(38) << std::left << " --ssl-https-verify-host: "
      << FormatMessage(
             "Number (0|1|2) to verify the "
             "certificate's name against host. "
             "See https://curl.se/libcurl/c/CURLOPT_SSL_VERIFYHOST.html for "
             "the meaning of each value. Default is 2.",
             38)
      << std::endl;
  std::cerr << std::setw(38) << std::left
            << " --ssl-https-ca-certificates-file: "
            << FormatMessage(
                   "Path to Certificate Authority "
                   "(CA) bundle.",
                   38)
            << std::endl;
  std::cerr << std::setw(38) << std::left
            << " --ssl-https-client-certificate-file: "
            << FormatMessage("Path to the SSL client certificate.", 38)
            << std::endl;
  std::cerr << std::setw(38) << std::left
            << " --ssl-https-client-certificate-type: "
            << FormatMessage(
                   "Type (PEM|DER) of the client "
                   "SSL certificate. Default is PEM.",
                   38)
            << std::endl;
  std::cerr << std::setw(38) << std::left << " --ssl-https-private-key-file: "
            << FormatMessage(
                   "Path to the private keyfile "
                   "for TLS and SSL client cert.",
                   38)
            << std::endl;
  std::cerr << std::setw(38) << std::left << " --ssl-https-private-key-type: "
            << FormatMessage(
                   "Type (PEM|DER) of the private "
                   "key file. Default is PEM.",
                   38)
            << std::endl;
  std::cerr << std::endl;
  std::cerr << "IV. OTHER OPTIONS: " << std::endl;
  std::cerr
      << std::setw(9) << std::left << " -f: "
      << FormatMessage(
             "The latency report will be stored in the file named by "
             "this option. By default, the result is not recorded in a file.",
             9)
      << std::endl;
  std::cerr << std::setw(9) << std::left << " --profile-export-file: "
            << FormatMessage(
                   "Specifies the path that the profile export will be "
                   "generated at. By default, the profile export will not be "
                   "generated.",
                   9)
            << std::endl;
  std::cerr
      << std::setw(9) << std::left << " -H: "
      << FormatMessage(
             "The header will be added to HTTP requests (ignored for GRPC "
             "requests). The header must be specified as 'Header:Value'. -H "
             "may be specified multiple times to add multiple headers.",
             9)
      << std::endl;
  std::cerr
      << FormatMessage(
             " --streaming: Enables the use of streaming API. This flag is "
             "only valid with gRPC protocol. By default, it is set false.",
             18)
      << std::endl;

  std::cerr << FormatMessage(
                   " --grpc-compression-algorithm: The compression algorithm "
                   "to be used by gRPC when sending request. Only supported "
                   "when grpc protocol is being used. The supported values are "
                   "none, gzip, and deflate. Default value is none.",
                   18)
            << std::endl;

  std::cerr
      << FormatMessage(
             " --trace-level: Specify a trace level. OFF to disable tracing, "
             "TIMESTAMPS to trace timestamps, TENSORS to trace tensors. It "
             "may be specified multiple times to trace multiple "
             "information. Default is OFF.",
             18)
      << std::endl;
  std::cerr
      << FormatMessage(
             " --trace-rate: Set the trace sampling rate. Default is 1000.", 18)
      << std::endl;
  std::cerr << FormatMessage(
                   " --trace-count: Set the number of traces to be sampled. "
                   "If the value is -1, the number of traces to be sampled "
                   "will not be limited. Default is -1.",
                   18)
            << std::endl;
  std::cerr
      << FormatMessage(
             " --log-frequency:  Set the trace log frequency. If the "
             "value is 0, Triton will only log the trace output to "
             "the trace file when shutting down. Otherwise, Triton will log "
             "the trace output to <trace-file>.<idx> when it collects the "
             "specified number of traces. For example, if the log frequency "
             "is 100, when Triton collects the 100-th trace, it logs the "
             "traces to file <trace-file>.0, and when it collects the 200-th "
             "trace, it logs the 101-th to the 200-th traces to file "
             "<trace-file>.1. Default is 0.",
             18)
      << std::endl;

  std::cerr << FormatMessage(
                   " --triton-server-directory: The Triton server install "
                   "path. Required by and only used when C API "
                   "is used (--service-kind=triton_c_api). "
                   "eg:--triton-server-directory=/opt/tritonserver.",
                   18)
            << std::endl;
  std::cerr
      << FormatMessage(
             " --model-repository: The model repository of which the model is "
             "loaded. Required by and only used when C API is used "
             "(--service-kind=triton_c_api). "
             "eg:--model-repository=/tmp/host/docker-data/model_unit_test.",
             18)
      << std::endl;
  std::cerr << FormatMessage(
                   " --verbose-csv: The csv files generated by perf analyzer "
                   "will include additional information.",
                   18)
            << std::endl;
  std::cerr << FormatMessage(
                   " --collect-metrics: Enables collection of server-side "
                   "inference server metrics. Outputs metrics in the csv file "
                   "generated with the -f option. Must enable `--verbose-csv` "
                   "option to use the `--collect-metrics`.",
                   18)
            << std::endl;
  std::cerr << FormatMessage(
                   " --metrics-url: The URL to query for server-side inference "
                   "server metrics. Default is 'localhost:8002/metrics'.",
                   18)
            << std::endl;
  std::cerr << FormatMessage(
                   " --metrics-interval: How often in milliseconds, within "
                   "each measurement window, to query for server-side "
                   "inference server metrics. Default is 1000.",
                   18)
            << std::endl;
  std::cerr << FormatMessage(
                   " --bls-composing-models: A comma separated list of all "
                   "BLS composing models (with optional model version number "
                   "after a colon for each) that may be called by the input "
                   "BLS model. For example, 'modelA:3,modelB' would specify "
                   "that modelA and modelB are composing models that may be "
                   "called by the input BLS model, and that modelA will use "
                   "version 3, while modelB's version is unspecified",
                   18)
            << std::endl;
  throw pa::PerfAnalyzerException(GENERIC_ERROR);
}

void
CLParser::PrintVersion()
{
  std::cerr << "Perf Analyzer Version " << VERSION << " (commit " << SHA << ")"
            << std::endl;
  exit(SUCCESS);
}

void
CLParser::ParseCommandLine(int argc, char** argv)
{
  argc_ = argc;
  argv_ = argv;

  // {name, has_arg, *flag, val}
  static struct option long_options[] = {
      {"streaming", no_argument, 0, 0},
      {"max-threads", required_argument, 0, 1},
      {"sequence-length", required_argument, 0, 2},
      {"percentile", required_argument, 0, 3},
      {"data-directory", required_argument, 0, 4},
      {"shape", required_argument, 0, 5},
      {"measurement-interval", required_argument, 0, 6},
      {"concurrency-range", required_argument, 0, 7},
      {"latency-threshold", required_argument, 0, 8},
      {"stability-percentage", required_argument, 0, 9},
      {"max-trials", required_argument, 0, 10},
      {"input-data", required_argument, 0, 11},
      {"string-length", required_argument, 0, 12},
      {"string-data", required_argument, 0, 13},
      {"async", no_argument, 0, 14},
      {"sync", no_argument, 0, 15},
      {"request-rate-range", required_argument, 0, 16},
      {"num-of-sequences", required_argument, 0, 17},
      {"binary-search", no_argument, 0, 18},
      {"request-distribution", required_argument, 0, 19},
      {"request-intervals", required_argument, 0, 20},
      {"shared-memory", required_argument, 0, 21},
      {"output-shared-memory-size", required_argument, 0, 22},
      {"service-kind", required_argument, 0, 23},
      {"model-signature-name", required_argument, 0, 24},
      {"grpc-compression-algorithm", required_argument, 0, 25},
      {"measurement-mode", required_argument, 0, 26},
      {"measurement-request-count", required_argument, 0, 27},
      {"triton-server-directory", required_argument, 0, 28},
      {"model-repository", required_argument, 0, 29},
      {"sequence-id-range", required_argument, 0, 30},
      {"ssl-grpc-use-ssl", no_argument, 0, 31},
      {"ssl-grpc-root-certifications-file", required_argument, 0, 32},
      {"ssl-grpc-private-key-file", required_argument, 0, 33},
      {"ssl-grpc-certificate-chain-file", required_argument, 0, 34},
      {"ssl-https-verify-peer", required_argument, 0, 35},
      {"ssl-https-verify-host", required_argument, 0, 36},
      {"ssl-https-ca-certificates-file", required_argument, 0, 37},
      {"ssl-https-client-certificate-file", required_argument, 0, 38},
      {"ssl-https-client-certificate-type", required_argument, 0, 39},
      {"ssl-https-private-key-file", required_argument, 0, 40},
      {"ssl-https-private-key-type", required_argument, 0, 41},
      {"verbose-csv", no_argument, 0, 42},
      {"enable-mpi", no_argument, 0, 43},
      {"trace-level", required_argument, 0, 44},
      {"trace-rate", required_argument, 0, 45},
      {"trace-count", required_argument, 0, 46},
      {"log-frequency", required_argument, 0, 47},
      {"collect-metrics", no_argument, 0, 48},
      {"metrics-url", required_argument, 0, 49},
      {"metrics-interval", required_argument, 0, 50},
      {"sequence-length-variation", required_argument, 0, 51},
      {"bls-composing-models", required_argument, 0, 52},
      {"serial-sequences", no_argument, 0, 53},
      {"input-tensor-format", required_argument, 0, 54},
      {"output-tensor-format", required_argument, 0, 55},
      {"version", no_argument, 0, 56},
      {"profile-export-file", required_argument, 0, 57},
      {"periodic-concurrency-range", required_argument, 0, 58},
      {"request-period", required_argument, 0, 59},
      {"request-parameter", required_argument, 0, 60},
      {"endpoint", required_argument, 0, 61},
      {"request-count", required_argument, 0, 62},
      {0, 0, 0, 0}};

  // Parse commandline...
  int opt;
  while ((opt = getopt_long(
              argc, argv, "vdazc:u:m:x:b:t:p:i:H:l:r:s:f:", long_options,
              NULL)) != -1) {
    try {
      switch (opt) {
        case 0:
          params_->streaming = true;
          break;
        case 1: {
          std::string max_threads{optarg};
          if (std::stoi(max_threads) > 0) {
            params_->max_threads = std::stoull(max_threads);
            params_->max_threads_specified = true;
          } else {
            Usage("Failed to parse --max-threads. The value must be > 0.");
          }
          break;
        }
        case 2: {
          std::string sequence_length{optarg};
          if (std::stoi(sequence_length) > 0) {
            params_->sequence_length = std::stoull(sequence_length);
          } else {
            std::cerr << "WARNING: The sequence length must be > 0. Perf "
                         "Analyzer will use default value if it is measuring "
                         "on sequence model."
                      << std::endl;
          }
          params_->sequence_length_specified = true;
          break;
        }
        case 3:
          params_->percentile = std::atoi(optarg);
          break;
        case 4:
          params_->user_data.push_back(optarg);
          break;
        case 5: {
          std::string arg = optarg;
          auto colon_pos = arg.rfind(":");
          if (colon_pos == std::string::npos) {
            Usage(
                "Failed to parse --shape. There must be a colon after input "
                "name.");
          }
          std::string name = arg.substr(0, colon_pos);
          std::string shape_str = arg.substr(name.size() + 1);
          size_t pos = 0;
          std::vector<int64_t> shape;
          while (pos != std::string::npos) {
            size_t comma_pos = shape_str.find(",", pos);
            int64_t dim;
            if (comma_pos == std::string::npos) {
              dim = std::stoll(shape_str.substr(pos, comma_pos));
              pos = comma_pos;
            } else {
              dim = std::stoll(shape_str.substr(pos, comma_pos - pos));
              pos = comma_pos + 1;
            }
            if (dim <= 0) {
              Usage(
                  "Failed to parse --shape. The dimensions of input tensor "
                  "must be > 0.");
            }
            shape.emplace_back(dim);
          }

          params_->input_shapes[name] = shape;
          break;
        }
        case 6:
        case 'p': {
          std::string measurement_window_ms{optarg};
          if (std::stoi(measurement_window_ms) > 0) {
            params_->measurement_window_ms = std::stoull(measurement_window_ms);
          } else {
            Usage(
                "Failed to parse --measurement-interval (-p). The value must "
                "be > 0 msec.");
          }
          break;
        }
        case 7: {
          params_->using_concurrency_range = true;
          std::string arg = optarg;
          std::vector<std::string> values{SplitString(arg)};
          if (values.size() > 3) {
            Usage(
                "Failed to parse --concurrency-range. The value does not match "
                "<start:end:step>.");
          }

          for (size_t i = 0; i < values.size(); ++i) {
            uint64_t val = std::stoull(values[i]);
            if (i == 0) {
              params_->concurrency_range.start = val;
            } else if (i == 1) {
              params_->concurrency_range.end = val;
            } else if (i == 2) {
              params_->concurrency_range.step = val;
            }
          }
          break;
        }
        case 8:
        case 'l': {
          std::string latency_threshold_ms{optarg};
          if (std::stoi(latency_threshold_ms) == 0) {
            params_->latency_threshold_ms = NO_LIMIT;
          } else if (std::stoi(latency_threshold_ms) > 0) {
            params_->latency_threshold_ms = std::stoull(latency_threshold_ms);
          } else {
            Usage(
                "Failed to parse --latency-threshold (-l). The value must be "
                ">= 0 msecs.");
          }
          break;
        }
        case 9:
        case 's': {
          std::string stability_threshold{optarg};
          if (std::stof(stability_threshold) >= 0.0) {
            params_->stability_threshold = std::stof(optarg) / 100;
          } else {
            Usage(
                "Failed to parse --stability-percentage (-s). The value must "
                "be >= 0.0.");
          }
          break;
        }
        case 10:
        case 'r': {
          std::string max_trials{optarg};
          if (std::stoi(max_trials) > 0) {
            params_->max_trials = std::stoull(max_trials);
          } else {
            Usage("Failed to parse --max-trials (-r). The value must be > 0.");
          }
          break;
        }
        case 11: {
          std::string arg = optarg;
          // Check whether the argument is a directory
          if (IsDirectory(arg) || IsFile(arg)) {
            params_->user_data.push_back(optarg);
          } else if (arg.compare("zero") == 0) {
            params_->zero_input = true;
          } else if (arg.compare("random") == 0) {
            break;
          } else {
            Usage(
                "Failed to parse --input-data. Unsupported type provided: '" +
                std::string(optarg) +
                "'. The available options are 'zero', 'random', path to a "
                "directory, or a json file.");
          }
          break;
        }
        case 12: {
          std::string string_length{optarg};
          if (std::stoi(string_length) > 0) {
            params_->string_length = std::stoull(string_length);
          } else {
            Usage("Failed to parse --string-length. The value must be > 0");
          }
          break;
        }
        case 13: {
          params_->string_data = optarg;
          break;
        }
        case 14:
        case 'a': {
          params_->async = true;
          break;
        }
        case 15: {
          params_->forced_sync = true;
          break;
        }
        case 16: {
          params_->using_request_rate_range = true;
          std::string arg = optarg;
          size_t pos = 0;
          int index = 0;
          while (pos != std::string::npos) {
            size_t colon_pos = arg.find(":", pos);
            if (index > 2) {
              Usage(
                  "Failed to parse --request-rate-range. The value does not "
                  "match <start:end:step>.");
            }
            if (colon_pos == std::string::npos) {
              params_->request_rate_range[index] =
                  std::stod(arg.substr(pos, colon_pos));
              pos = colon_pos;
            } else {
              params_->request_rate_range[index] =
                  std::stod(arg.substr(pos, colon_pos - pos));
              pos = colon_pos + 1;
              index++;
            }
          }

          break;
        }
        case 17: {
          std::string num_of_sequences{optarg};
          if (std::stoi(num_of_sequences) > 0) {
            params_->num_of_sequences = std::stoul(num_of_sequences);
          } else {
            Usage("Failed to parse --num-of-sequences. The value must be > 0.");
          }
          break;
        }
        case 18: {
          params_->search_mode = SearchMode::BINARY;
          break;
        }
        case 19: {
          std::string arg = optarg;
          if (arg.compare("poisson") == 0) {
            params_->request_distribution = Distribution::POISSON;
          } else if (arg.compare("constant") == 0) {
            params_->request_distribution = Distribution::CONSTANT;
          } else {
            Usage(
                "Failed to parse --request-distribution. Unsupported type "
                "provided: '" +
                std::string(optarg) + "'. Choices are 'posson' or 'constant'.");
          }
          break;
        }
        case 20: {
          std::string request_intervals_file{optarg};
          if (IsFile(request_intervals_file)) {
            params_->request_intervals_file = request_intervals_file;
            params_->using_custom_intervals = true;
          } else {
            Usage(
                "Failed to parse --request-intervals. The value must be a "
                "valid file path");
          }
          break;
        }
        case 21: {
          std::string arg = optarg;
          if (arg.compare("system") == 0) {
            params_->shared_memory_type =
                SharedMemoryType::SYSTEM_SHARED_MEMORY;
          } else if (arg.compare("cuda") == 0) {
#ifdef TRITON_ENABLE_GPU
            params_->shared_memory_type = SharedMemoryType::CUDA_SHARED_MEMORY;
#else
            Usage(
                "Cuda shared memory is not supported when "
                "TRITON_ENABLE_GPU=0.");
#endif  // TRITON_ENABLE_GPU
          } else if (arg.compare("none") == 0) {
            params_->shared_memory_type = SharedMemoryType::NO_SHARED_MEMORY;
          } else {
            Usage(
                "Failed to parse --shared-memory. Unsupported type provided: "
                "'" +
                std::string(optarg) +
                "'. The available options are 'system', 'cuda', or 'none'.");
          }
          break;
        }
        case 22: {
          std::string output_shm_size{optarg};
          if (std::stoi(output_shm_size) >= 0) {
            params_->output_shm_size = std::stoull(output_shm_size);
          } else {
            Usage(
                "Failed to parse --output-shared-memory-size. The value must "
                "be >= 0.");
          }
          break;
        }
        case 23: {
          std::string arg = optarg;
          if (arg.compare("triton") == 0) {
            params_->kind = cb::TRITON;
          } else if (arg.compare("tfserving") == 0) {
            params_->kind = cb::TENSORFLOW_SERVING;
          } else if (arg.compare("torchserve") == 0) {
            params_->kind = cb::TORCHSERVE;
          } else if (arg.compare("triton_c_api") == 0) {
            params_->kind = cb::TRITON_C_API;
          } else if (arg.compare("openai") == 0) {
            params_->kind = cb::OPENAI;
          } else {
            Usage(
                "Failed to parse --service-kind. Unsupported type provided: '" +
                std::string{optarg} +
                "'. The available options are 'triton', 'tfserving', "
                "'torchserve', or 'triton_c_api'.");
          }
          break;
        }
        case 24:
          params_->model_signature_name = optarg;
          break;
        case 25: {
          std::string arg = optarg;
          if (arg.compare("none") == 0) {
            params_->compression_algorithm = cb::COMPRESS_NONE;
          } else if (arg.compare("deflate") == 0) {
            params_->compression_algorithm = cb::COMPRESS_DEFLATE;
          } else if (arg.compare("gzip") == 0) {
            params_->compression_algorithm = cb::COMPRESS_GZIP;
          } else {
            Usage(
                "Failed to parse --grpc-compression-algorithm. Unsupported "
                "type provided: '" +
                arg +
                "'. The available options are 'gzip', 'deflate', or 'none'.");
          }
          params_->using_grpc_compression = true;
          break;
        }
        case 26: {
          std::string arg = optarg;
          if (arg.compare("time_windows") == 0) {
            params_->measurement_mode = MeasurementMode::TIME_WINDOWS;
          } else if (arg.compare("count_windows") == 0) {
            params_->measurement_mode = MeasurementMode::COUNT_WINDOWS;
          } else {
            Usage(
                "Failed to parse --measurement-mode. Unsupported type "
                "provided: '" +
                arg +
                "'. The available options are 'time_windows' or "
                "'count_windows'.");
          }
          break;
        }
        case 27: {
          std::string request_count{optarg};
          if (std::stoi(request_count) > 0) {
            params_->measurement_request_count = std::stoull(request_count);
          } else {
            Usage(
                "Failed to parse --measurement-request-count. The value must "
                "be > 0.");
          }
          break;
        }
        case 28: {
          params_->triton_server_path = optarg;
          break;
        }
        case 29: {
          params_->model_repository_path = optarg;
          break;
        }
        case 30: {
          std::string arg = optarg;
          int64_t start_id;
          int64_t end_id;
          size_t pos = 0;
          int index = 0;
          while (pos != std::string::npos) {
            size_t colon_pos = arg.find(":", pos);
            if (index > 1) {
              Usage(
                  "Failed to parse --sequence-id-range. The value does not "
                  "match <start:end>.");
            }
            if (colon_pos == std::string::npos) {
              std::string sequence_id{arg.substr(pos, colon_pos)};
              if (index == 0) {
                start_id = std::stoi(sequence_id);
              } else {
                end_id = std::stoi(sequence_id);
              }
              pos = colon_pos;
            } else {
              std::string sequence_id{arg.substr(pos, colon_pos - pos)};
              start_id = std::stoi(sequence_id);
              pos = colon_pos + 1;
              index++;
            }
          }

          // Check for invalid inputs
          if (start_id < 0 || end_id < 0) {
            Usage(
                "Failed to parse --sequence-id-range. The range values must be "
                ">= 0.");
          } else if (start_id > end_id) {
            Usage(
                "Failed to parse --sequence-id-range. The 'end' value must be "
                "greater than 'start' value.");
          }

          if (index == 0) {  // Only start ID is given
            params_->start_sequence_id = start_id;
          } else {
            params_->start_sequence_id = start_id;
            params_->sequence_id_range = end_id - start_id;
          }
          break;
        }
        case 31: {
          params_->ssl_options.ssl_grpc_use_ssl = true;
          break;
        }
        case 32: {
          if (IsFile(optarg)) {
            params_->ssl_options.ssl_grpc_root_certifications_file = optarg;
          } else {
            Usage(
                "Failed to parse --ssl-grpc-root-certifications-file. The "
                "value must be a valid file path.");
          }
          break;
        }
        case 33: {
          if (IsFile(optarg)) {
            params_->ssl_options.ssl_grpc_private_key_file = optarg;
          } else {
            Usage(
                "Failed to parse --ssl-grpc-private-key-file. The value must "
                "be a valid file path.");
          }
          break;
        }
        case 34: {
          if (IsFile(optarg)) {
            params_->ssl_options.ssl_grpc_certificate_chain_file = optarg;
          } else {
            Usage(
                "Failed to parse --ssl-grpc-certificate-chain-file. The value "
                "must be a valid file path.");
          }
          break;
        }
        case 35: {
          if (std::atol(optarg) == 0 || std::atol(optarg) == 1) {
            params_->ssl_options.ssl_https_verify_peer = std::atol(optarg);
          } else {
            Usage(
                "Failed to parse --ssl-https-verify-peer. The value must be "
                "either 0 or 1.");
          }
          break;
        }
        case 36: {
          if (std::atol(optarg) == 0 || std::atol(optarg) == 1 ||
              std::atol(optarg) == 2) {
            params_->ssl_options.ssl_https_verify_host = std::atol(optarg);
          } else {
            Usage(
                "Failed to parse --ssl-https-verify-host. The value must be "
                "either 0, 1, or 2.");
          }
          break;
        }
        case 37: {
          if (IsFile(optarg)) {
            params_->ssl_options.ssl_https_ca_certificates_file = optarg;
          } else {
            Usage(
                "Failed to parse --ssl-https-ca-certificates-file. The value "
                "must be a valid file path.");
          }
          break;
        }
        case 38: {
          if (IsFile(optarg)) {
            params_->ssl_options.ssl_https_client_certificate_file = optarg;
          } else {
            Usage(
                "Failed to parse --ssl-https-client-certificate-file. The "
                "value must be a valid file path.");
          }
          break;
        }
        case 39: {
          if (std::string(optarg) == "PEM" || std::string(optarg) == "DER") {
            params_->ssl_options.ssl_https_client_certificate_type = optarg;
          } else {
            Usage(
                "Failed to parse --ssl-https-client-certificate-type. "
                "Unsupported type provided: '" +
                std::string{optarg} +
                "'. The available options are 'PEM' or 'DER'.");
          }
          break;
        }
        case 40: {
          if (IsFile(optarg)) {
            params_->ssl_options.ssl_https_private_key_file = optarg;
          } else {
            Usage(
                "Failed to parse --ssl-https-private-key-file. The value must "
                "be a valid file path.");
          }
          break;
        }
        case 41: {
          if (std::string(optarg) == "PEM" || std::string(optarg) == "DER") {
            params_->ssl_options.ssl_https_private_key_type = optarg;
          } else {
            Usage(
                "Failed to parse --ssl-https-private-key-type. Unsupported "
                "type provided: '" +
                std::string{optarg} +
                "'. The available options are 'PEM' or 'DER'.");
          }
          break;
        }
        case 42: {
          params_->verbose_csv = true;
          break;
        }
        case 43: {
          params_->enable_mpi = true;
          break;
        }
        case 44: {
          std::string trace_level{optarg};
          if (trace_level == "OFF" || trace_level == "TIMESTAMPS" ||
              trace_level == "TENSORS") {
            params_->trace_options["trace_level"] = {trace_level};
          } else {
            Usage(
                "Failed to parse --trace-level. Unsupported type provided: '" +
                trace_level +
                "'. The available options are 'OFF', 'TIMESTAMPS', or "
                "'TENSORS'.");
          }
          break;
        }
        case 45: {
          params_->trace_options["trace_rate"] = {optarg};
          break;
        }
        case 46: {
          std::string trace_count{optarg};
          if (std::stoi(trace_count) >= -1) {
            params_->trace_options["trace_count"] = {trace_count};
          } else {
            Usage(
                "Failed to parse --trace-count. The value must be >= 0 or set "
                "to -1 (default).");
          }
          break;
        }
        case 47: {
          std::string log_frequency{optarg};
          if (std::stoi(log_frequency) >= 0) {
            params_->trace_options["log_frequency"] = {log_frequency};
          } else {
            Usage("Failed to parse --log-frequency. The value must be >= 0.");
          }
          break;
        }
        case 48: {
          params_->should_collect_metrics = true;
          break;
        }
        case 49: {
          params_->metrics_url = optarg;
          params_->metrics_url_specified = true;
          break;
        }
        case 50: {
          std::string metrics_interval_ms{optarg};
          if (std::stoi(metrics_interval_ms) > 0) {
            params_->metrics_interval_ms = std::stoull(metrics_interval_ms);
            params_->metrics_interval_ms_specified = true;
          } else {
            Usage(
                "Failed to parse --metrics-interval. The value must be > 0 "
                "msecs.");
          }
          break;
        }
        case 51: {
          params_->sequence_length_variation = std::stod(optarg);
          break;
        }
        case 52: {
          std::string arg = optarg;

          // Remove all spaces in the string
          arg.erase(
              std::remove_if(arg.begin(), arg.end(), ::isspace), arg.end());

          std::stringstream ss(arg);
          while (ss.good()) {
            std::string model_name;
            std::string model_version{""};
            std::string tmp_model_name;

            getline(ss, tmp_model_name, ',');

            size_t colon_pos = tmp_model_name.find(":");

            if (colon_pos == std::string::npos) {
              model_name = tmp_model_name;
            } else {
              model_name = tmp_model_name.substr(0, colon_pos);
              model_version = tmp_model_name.substr(colon_pos + 1);
            }

            params_->bls_composing_models.push_back(
                {model_name, model_version});
          }
          break;
        }
        case 53: {
          params_->serial_sequences = true;
          break;
        }
        case 54: {
          cb::TensorFormat input_tensor_format{ParseTensorFormat(optarg)};
          if (input_tensor_format == cb::TensorFormat::UNKNOWN) {
            Usage(
                "Failed to parse --input-tensor-format. Unsupported type "
                "provided: '" +
                std::string{optarg} +
                "'. The available options are 'binary' or 'json'.");
          }
          params_->input_tensor_format = input_tensor_format;
          break;
        }
        case 55: {
          cb::TensorFormat output_tensor_format{ParseTensorFormat(optarg)};
          if (output_tensor_format == cb::TensorFormat::UNKNOWN) {
            Usage(
                "Failed to parse --output-tensor-format. Unsupported type "
                "provided: '" +
                std::string{optarg} +
                "'. The available options are 'binary' or 'json'.");
          }
          params_->output_tensor_format = output_tensor_format;
          break;
        }
        case 56: {
          PrintVersion();
          break;
        }
        case 57: {
          std::string profile_export_file{optarg};
          if (IsFile(profile_export_file) || IsDirectory(profile_export_file)) {
            Usage(
                "Failed to parse --profile-export-file. Path must not already "
                "exist.");
          }
          params_->profile_export_file = profile_export_file;
          break;
        }
        case 58: {
          params_->is_using_periodic_concurrency_mode = true;
          std::string arg = optarg;
          std::vector<std::string> values{SplitString(arg)};
          if (values.size() < 2) {
            Usage(
                "Failed to parse --periodic-concurrency-range. Both <start> "
                "and <end> values must be provided.");
          } else if (values.size() > 3) {
            Usage(
                "Failed to parse --periodic-concurrency-range. The value does "
                "not match <start:end:step>.");
          }

          for (size_t i = 0; i < values.size(); ++i) {
            uint64_t val = std::stoull(values[i]);
            if (i == 0) {
              params_->periodic_concurrency_range.start = val;
            } else if (i == 1) {
              params_->periodic_concurrency_range.end = val;
            } else if (i == 2) {
              params_->periodic_concurrency_range.step = val;
            }
          }

          Range<uint64_t> range{params_->periodic_concurrency_range};
          if (range.step == 0) {
            Usage(
                "Failed to parse --periodic-concurrency-range. The <step> "
                "value must be > 0.");
          } else if (range.start > range.end) {
            Usage(
                "Failed to parse --periodic-concurrency-range. The <start> "
                "must be <= <end>.");
          } else if ((range.end - range.start) % range.step != 0) {
            Usage(
                "Failed to parse --periodic-concurrency-range. The <step> "
                "value must be a factor of the range size (<end> - <start>).");
          }
          break;
        }
        case 59: {
          std::string request_period{optarg};
          if (std::stoi(request_period) > 0) {
            params_->request_period = std::stoull(request_period);
          } else {
            Usage("Failed to parse --request-period. The value must be > 0");
          }
          break;
        }
        case 60: {
          std::string arg = optarg;
          std::vector<std::string> values{SplitString(arg)};
          if (values.size() != 3) {
            Usage(
                "Failed to parse --request-parameter. The value does not match "
                "<name:value:type>.");
          }

          std::for_each(values.begin(), values.end(), ToLowerCase);
          std::string name{values[0]};
          std::string value{values[1]};
          std::string type{values[2]};

          cb::RequestParameter param;
          param.name = name;
          param.value = value;
          param.type = type;
          params_->request_parameters[name] = param;
          break;
        }
        case 61: {
          params_->endpoint = optarg;
          break;
        }
        case 62: {
          if (std::stoi(optarg) < 0) {
            Usage("Failed to parse --request-count. The value must be > 0.");
          }
          params_->request_count = std::stoi(optarg);
          break;
        }
        case 'v':
          params_->extra_verbose = params_->verbose;
          params_->verbose = true;
          break;
        case 'z':
          params_->zero_input = true;
          break;
        case 'd':
          params_->using_old_options = true;
          params_->dynamic_concurrency_mode = true;
          break;
        case 'u':
          params_->url_specified = true;
          params_->url = optarg;
          break;
        case 'm':
          params_->model_name = optarg;
          break;
        case 'x':
          params_->model_version = optarg;
          break;
        case 'b': {
          std::string batch_size{optarg};
          if (std::stoi(batch_size) > 0) {
            params_->batch_size = std::stoull(batch_size);
            params_->using_batch_size = true;
          } else {
            Usage("Failed to parse -b (batch size). The value must be > 0.");
          }
          break;
        }
        case 't':
          params_->using_old_options = true;
          params_->concurrent_request_count = std::atoi(optarg);
          break;
        case 'i':
          params_->protocol = ParseProtocol(optarg);
          break;
        case 'H': {
          std::string arg = optarg;
          std::string header = arg.substr(0, arg.find(":"));
          (*params_->http_headers)[header] = arg.substr(header.size() + 1);
          break;
        }
        case 'c':
          params_->using_old_options = true;
          params_->max_concurrency = std::atoi(optarg);
          break;
        case 'f':
          params_->filename = optarg;
          break;
        case '?':
          Usage();
          break;
      }
    }
    catch (const std::invalid_argument& ia) {
      if (opt >= 'A') {  // short options
        Usage(
            "Failed to parse -" + std::string{(char)opt} +
            ". Invalid value provided: " + std::string{optarg});
      } else {
        Usage(
            "Failed to parse --" + std::string{long_options[opt].name} +
            ". Invalid value provided: " + std::string{optarg});
      }
    }
  }

  params_->mpi_driver = std::shared_ptr<triton::perfanalyzer::MPIDriver>{
      std::make_shared<triton::perfanalyzer::MPIDriver>(params_->enable_mpi)};
  params_->mpi_driver->MPIInit(&argc, &argv);

  if (!params_->url_specified &&
      (params_->protocol == cb::ProtocolType::GRPC)) {
    if (params_->kind == cb::BackendKind::TRITON) {
      params_->url = "localhost:8001";
    } else if (params_->kind == cb::BackendKind::TENSORFLOW_SERVING) {
      params_->url = "localhost:8500";
    }
  }

  // Overriding the max_threads default for request_rate search
  if (!params_->max_threads_specified && params_->targeting_concurrency()) {
    params_->max_threads = 16;
    params_->max_threads =
        std::max(params_->max_threads, params_->concurrency_range.end);
  }

  if (params_->using_custom_intervals) {
    // Will be using user-provided time intervals, hence no control variable.
    params_->search_mode = SearchMode::NONE;
  }

  // When the request-count feature is enabled, override the measurement mode to
  // be count windows with a window size of the requested count
  if (params_->request_count) {
    params_->measurement_mode = MeasurementMode::COUNT_WINDOWS;
    params_->measurement_request_count = params_->request_count;
  }
}

void
CLParser::VerifyOptions()
{
  if (params_->model_name.empty()) {
    Usage("Failed to parse -m (model name). The value must be specified.");
  }
  if (params_->concurrency_range.start <= 0 ||
      params_->concurrent_request_count < 0) {
    Usage("The start of the search range must be > 0");
  }
  if (params_->request_rate_range[SEARCH_RANGE::kSTART] <= 0) {
    Usage(
        "Failed to parse --request-rate-range. The start of the search range "
        "must be > 0.");
  }
  if (params_->protocol == cb::ProtocolType::UNKNOWN) {
    Usage(
        "Failed to parse -i (protocol). The value should be either HTTP or "
        "gRPC.");
  }
  if (params_->streaming && (params_->protocol != cb::ProtocolType::GRPC)) {
    Usage("Streaming is only allowed with gRPC protocol.");
  }
  if (params_->using_grpc_compression &&
      (params_->protocol != cb::ProtocolType::GRPC)) {
    Usage("Using compression algorithm is only allowed with gRPC protocol.");
  }
  if (params_->sequence_length_variation < 0.0) {
    Usage(
        "Failed to parse --sequence-length-variation. The value must be >= "
        "0.0.");
  }
  if (params_->start_sequence_id == 0) {
    params_->start_sequence_id = 1;
    std::cerr << "WARNING: using an invalid start sequence id. Perf Analyzer"
              << " will use default value if it is measuring on sequence model."
              << std::endl;
  }
  if (params_->percentile != -1 &&
      (params_->percentile > 99 || params_->percentile < 1)) {
    Usage(
        "Failed to parse --percentile. The value must be -1 for not reporting "
        "or in range (0, 100).");
  }
  if (params_->zero_input && !params_->user_data.empty()) {
    Usage("The -z flag cannot be set when --data-directory is provided.");
  }
  if (params_->async && params_->forced_sync) {
    Usage("Cannot specify --async and --sync simultaneously.");
  }

  if (params_->using_concurrency_range && params_->using_old_options) {
    Usage("Cannot use deprecated options with --concurrency-range.");
  } else if (params_->using_old_options) {
    if (params_->dynamic_concurrency_mode) {
      params_->concurrency_range.end = params_->max_concurrency;
    }
    params_->concurrency_range.start = params_->concurrent_request_count;
  }

  if (params_->using_request_rate_range && params_->using_old_options) {
    Usage("Cannot use concurrency options with --request-rate-range.");
  }

  std::vector<bool> load_modes{
      params_->is_using_periodic_concurrency_mode,
      params_->using_concurrency_range, params_->using_request_rate_range,
      params_->using_custom_intervals};
  if (std::count(load_modes.begin(), load_modes.end(), true) > 1) {
    Usage(
        "Cannot specify more then one inference load mode. Please choose only "
        "one of the following modes: --concurrency-range, "
        "--periodic-concurrency-range, --request-rate-range, or "
        "--request-intervals.");
  }

  if (params_->is_using_periodic_concurrency_mode && !params_->streaming) {
    Usage(
        "The --periodic-concurrency-range option requires bi-directional gRPC "
        "streaming.");
  }

  if (params_->is_using_periodic_concurrency_mode &&
      (params_->profile_export_file == "")) {
    Usage(
        "Must provide --profile-export-file when using the "
        "--periodic-concurrency-range option.");
  }

  if (params_->is_using_periodic_concurrency_mode) {
    if (params_->periodic_concurrency_range.end == pa::NO_LIMIT) {
      std::cerr
          << "WARNING: The maximum attainable concurrency will be limited by "
             "max_threads specification."
          << std::endl;
      params_->periodic_concurrency_range.end = params_->max_threads;
    } else {
      if (params_->max_threads_specified) {
        std::cerr << "WARNING: Overriding max_threads specification to ensure "
                     "requested concurrency range."
                  << std::endl;
      }
      params_->max_threads = std::max(
          params_->max_threads, params_->periodic_concurrency_range.end);
    }
  }

  if (params_->request_parameters.size() > 0 &&
      params_->protocol != cb::ProtocolType::GRPC) {
    Usage(
        "The --request-parameter option is currently only supported by gRPC "
        "protocol.");
  }

  if (params_->using_request_rate_range && params_->mpi_driver->IsMPIRun() &&
      (params_->request_rate_range[SEARCH_RANGE::kEND] != 1.0 ||
       params_->request_rate_range[SEARCH_RANGE::kSTEP] != 1.0)) {
    Usage("Cannot specify --request-rate-range when in multi-model mode.");
  }

  if (params_->using_custom_intervals && params_->using_old_options) {
    Usage("Cannot use deprecated options with --request-intervals.");
  }

  if ((params_->using_custom_intervals) &&
      (params_->using_request_rate_range || params_->using_concurrency_range)) {
    Usage(
        "Cannot use --concurrency-range or --request-rate-range "
        "along with --request-intervals.");
  }

  if (params_->using_concurrency_range && params_->mpi_driver->IsMPIRun() &&
      (params_->concurrency_range.end != 1 ||
       params_->concurrency_range.step != 1)) {
    Usage("Cannot specify --concurrency-range when in multi-model mode.");
  }

  if (((params_->concurrency_range.end == NO_LIMIT) ||
       (params_->request_rate_range[SEARCH_RANGE::kEND] ==
        static_cast<double>(NO_LIMIT))) &&
      (params_->latency_threshold_ms == NO_LIMIT)) {
    Usage(
        "The end of the search range and the latency limit can not be both 0 "
        "(or 0.0) simultaneously");
  }

  if (((params_->concurrency_range.end == NO_LIMIT) ||
       (params_->request_rate_range[SEARCH_RANGE::kEND] ==
        static_cast<double>(NO_LIMIT))) &&
      (params_->search_mode == SearchMode::BINARY)) {
    Usage("The end of the range can not be 0 (or 0.0) for binary search mode.");
  }

  if ((params_->search_mode == SearchMode::BINARY) &&
      (params_->latency_threshold_ms == NO_LIMIT)) {
    Usage("The --latency-threshold cannot be 0 for binary search mode.");
  }

  if (((params_->concurrency_range.end < params_->concurrency_range.start) ||
       (params_->request_rate_range[SEARCH_RANGE::kEND] <
        params_->request_rate_range[SEARCH_RANGE::kSTART])) &&
      (params_->search_mode == SearchMode::BINARY)) {
    Usage(
        "The end of the range can not be less than start of the range for "
        "binary search mode.");
  }

  if (params_->request_count != 0) {
    if (params_->using_concurrency_range) {
      if (params_->request_count < params_->concurrency_range.start) {
        Usage("request-count can not be less than concurrency");
      }
      if (params_->concurrency_range.start < params_->concurrency_range.end) {
        Usage(
            "request-count not supported with multiple concurrency values in "
            "one run");
      }
    }
    if (params_->using_request_rate_range) {
      if (params_->request_count <
          static_cast<int>(params_->request_rate_range[0])) {
        Usage("request-count can not be less than request-rate");
      }
      if (params_->request_rate_range[SEARCH_RANGE::kSTART] <
          params_->request_rate_range[SEARCH_RANGE::kEND]) {
        Usage(
            "request-count not supported with multiple request-rate values in "
            "one run");
      }
    }
  }

  if (params_->kind == cb::TENSORFLOW_SERVING) {
    if (params_->protocol != cb::ProtocolType::GRPC) {
      Usage(
          "perf_analyzer supports only grpc protocol for TensorFlow Serving.");
    } else if (params_->streaming) {
      Usage("perf_analyzer does not support streaming for TensorFlow Serving.");
    } else if (params_->async) {
      Usage("perf_analyzer does not support async API for TensorFlow Serving.");
    } else if (!params_->using_batch_size) {
      params_->batch_size = 0;
    }
  } else if (params_->kind == cb::TORCHSERVE) {
    if (params_->user_data.empty()) {
      Usage(
          "--input-data should be provided with a json file with "
          "input data for torchserve.");
    }
  }

  if (params_->kind == cb::BackendKind::TRITON_C_API) {
    if (params_->triton_server_path.empty()) {
      Usage(
          "--triton-server-path should not be empty when using "
          "service-kind=triton_c_api.");
    }

    if (params_->model_repository_path.empty()) {
      Usage(
          "--model-repository should not be empty when using "
          "service-kind=triton_c_api.");
    }

    if (params_->async) {
      Usage(
          "Async mode is not supported by triton_c_api service "
          "kind.");
    }

    params_->protocol = cb::ProtocolType::UNKNOWN;
  }

  if (params_->kind == cb::BackendKind::OPENAI) {
    if (params_->user_data.empty()) {
      Usage("Must supply --input-data for OpenAI service kind.");
    }
    if (params_->endpoint.empty()) {
      Usage(
          "Must supply --endpoint for OpenAI service kind. For example, "
          "\"v1/chat/completions\".");
    }
    if (!params_->async) {
      Usage("Only async mode is currently supported for OpenAI service-kind");
    }
    if (params_->batch_size != 1) {
      Usage("Batching is not currently supported with OpenAI service-kind");
    }
  }

  if (params_->should_collect_metrics &&
      params_->kind != cb::BackendKind::TRITON) {
    Usage(
        "Server-side metric collection is only supported with Triton client "
        "backend.");
  }

  if (params_->metrics_url_specified &&
      params_->should_collect_metrics == false) {
    Usage(
        "Must specify --collect-metrics when using the --metrics-url option.");
  }

  if (params_->metrics_interval_ms_specified &&
      params_->should_collect_metrics == false) {
    Usage(
        "Must specify --collect-metrics when using the --metrics-interval "
        "option.");
  }

  if (params_->should_collect_metrics && !params_->metrics_url_specified) {
    // Update the default metrics URL to be associated with the input URL
    // instead of localhost
    //
    size_t colon_pos = params_->url.find(':');
    if (colon_pos != std::string::npos) {
      params_->metrics_url =
          params_->url.substr(0, colon_pos) + ":8002/metrics";
    }
  }
}

}}  // namespace triton::perfanalyzer
