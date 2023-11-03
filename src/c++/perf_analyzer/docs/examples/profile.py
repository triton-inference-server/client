# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import csv
import json
import subprocess
from dataclasses import asdict, dataclass, fields
from itertools import pairwise
from pathlib import Path
from typing import Optional

import numpy as np

INPUT_FILENAME = "generated_input_data.json"
METRIC_FIELDS = {
    "max_first_token_latency": ("Max first token latency", "ms"),
    "min_first_token_latency": ("Min first token latency", "ms"),
    "avg_first_token_latency": ("Avg first token latency", "ms"),
    "p50_first_token_latency": ("p50 first token latency", "ms"),
    "p90_first_token_latency": ("p90 first token latency", "ms"),
    "p95_first_token_latency": ("p95 first token latency", "ms"),
    "p99_first_token_latency": ("p99 first token latency", "ms"),
    "max_gen_latency": ("Max generation latency", "ms"),
    "min_gen_latency": ("Min generation latency", "ms"),
    "avg_gen_latency": ("Avg generation latency", "ms"),
    "p50_gen_latency": ("p50 generation latency", "ms"),
    "p90_gen_latency": ("p90 generation latency", "ms"),
    "p95_gen_latency": ("p95 generation latency", "ms"),
    "p99_gen_latency": ("p99 generation latency", "ms"),
    "avg_token_latency": ("Avg token latency", "ms/token"),
    "avg_total_t2t_latency": ("Avg total token-to-token latency", "ms"),
    "max_e2e_latency": ("Max end-to-end latency", "ms"),
    "min_e2e_latency": ("Min end-to-end latency", "ms"),
    "avg_e2e_latency": ("Avg end-to-end latency", "ms"),
    "max_token_throughput": ("Max token throughput", "tokens/s"),
    "min_token_throughput": ("Min token throughput", "tokens/s"),
    "avg_token_throughput": ("Avg token throughput", "tokens/s"),
    "p50_token_throughput": ("p50 token throughput", "tokens/s"),
    "p90_token_throughput": ("p90 token throughput", "tokens/s"),
    "p95_token_throughput": ("p95 token throughput", "tokens/s"),
    "p99_token_throughput": ("p99 token throughput", "tokens/s"),
}


@dataclass
class ProfileResults:
    prompt_size: int
    max_first_token_latency: Optional[float] = None
    min_first_token_latency: Optional[float] = None
    avg_first_token_latency: Optional[float] = None
    p50_first_token_latency: Optional[float] = None
    p90_first_token_latency: Optional[float] = None
    p95_first_token_latency: Optional[float] = None
    p99_first_token_latency: Optional[float] = None
    max_gen_latency: Optional[float] = None
    min_gen_latency: Optional[float] = None
    avg_gen_latency: Optional[float] = None
    p50_gen_latency: Optional[float] = None
    p90_gen_latency: Optional[float] = None
    p95_gen_latency: Optional[float] = None
    p99_gen_latency: Optional[float] = None
    avg_token_latency: Optional[float] = None
    avg_total_t2t_latency: Optional[float] = None
    avg_periodic_t2t_latencies: Optional[list[float]] = None
    max_e2e_latency: Optional[float] = None
    min_e2e_latency: Optional[float] = None
    avg_e2e_latency: Optional[float] = None
    max_token_throughput: Optional[float] = None
    min_token_throughput: Optional[float] = None
    avg_token_throughput: Optional[float] = None
    p50_token_throughput: Optional[float] = None
    p90_token_throughput: Optional[float] = None
    p95_token_throughput: Optional[float] = None
    p99_token_throughput: Optional[float] = None


def load_json_data(filename):
    with open(filename) as f:
        return json.load(f)


def save_json_data(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f)


def get_postfix(args, prompt_size):
    """Generate postfix for profile export filename and plot.

    e.g.
      - trtllm-prompt100-maxtokens256
      - trtllm-prompt100-periodic1_100_1-period32-maxtokens1024
    """
    stream_type = "online" if args.stream else "offline"
    postfix = f"{args.model}-{stream_type}-prompt{prompt_size}-"
    if args.periodic_concurrency_range:
        start, end, step = args.periodic_concurrency_range
        postfix += f"periodic{start}_{end}_{step}-period{args.request_period}-"
    postfix += f"maxtokens{args.max_tokens}"
    return postfix


def get_export_filename(args, prompt_size):
    postfix = get_postfix(args, prompt_size)
    filename = f"profile_export-{postfix}.json"
    return filename


def get_plot_filename(args, prompt_size):
    postfix = get_postfix(args, prompt_size)
    filename = f"inflight_batching_benchmark-{postfix}.png"
    return filename


def save_benchmark_results(args, profile_results):
    for pr in profile_results:
        postfix = get_postfix(args, pr.prompt_size)
        results_csv = f"results-{postfix}.csv"
        with open(results_csv, "w") as f:
            fieldnames = [f.name for f in fields(pr)]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(asdict(pr))
        print(f"Saved benchmark results @ '{results_csv}'")


def print_benchmark_summary(profile_results):
    print("[ BENCHMARK SUMMARY ]")
    for pr in profile_results:
        print(f"Prompt size: {pr.prompt_size}")
        for metric, (name, unit) in METRIC_FIELDS.items():
            if getattr(pr, metric):
                print(f"  * {name}: {getattr(pr, metric):.4f} {unit}")
        print("")


def plot_results(latencies, filename="inflight_batching_benchmark.png"):
    """Plot in-flight batching LLM bencharmark results."""
    import matplotlib.pyplot as plt  # Lazy import

    periods = np.arange(1, len(latencies) + 1)
    fig, ax = plt.subplots()
    ax.plot(periods, latencies)

    # Set pyplot parameters
    ax.grid(linestyle="--")
    ax.set_xlabel("i-th Request Period", fontsize=12)
    ax.set_ylabel("Avg Token-to-Token Latency (ms)", fontsize=12)
    ax.set_title("In-Flight Batching Benchmark Summary", fontsize=14)
    ax.set_ylim(bottom=0.0)

    fig.savefig(filename, dpi=300)


def add_latencies_to_bins(bins, pos, responses, request_period):
    """Add token-to-token latencies into the corresponding bin.

    Given the responses of a single request, calculate token-to-token
    latency and add it into bin. Update the bin position to the next
    for every request period.
    """
    for response_id, (prev_res, res) in enumerate(pairwise(responses)):
        bins[pos].append(res - prev_res)
        if (response_id + 1) % request_period == 0:
            pos += 1


def update_start_position(request_id, start_pos, initial_requests, step):
    """Shift the start position of the bin.

    Once we iterate through the entire <start> requests, we shift
    the start position. Then, we shift the start position for every
    <step> requests.
    """
    if (request_id + 1) >= initial_requests:
        num_requests_after_start = request_id + 1 - initial_requests
        if num_requests_after_start % step == 0:
            start_pos += 1
    return start_pos


def collect_periodic_latencies(args, filename):
    """Split the entire benchmark results into segments with size
    of request period and collect latencies for each segment.
    """
    start, end, step = args.periodic_concurrency_range

    num_bins = args.max_tokens // args.request_period + (end - start) // step
    if args.max_tokens % args.request_period != 0:
        num_bins += 1  # extra bin

    bins = [[] for _ in range(num_bins)]
    bin_start_position = 0

    data = load_json_data(filename)
    requests = data["experiments"][0]["requests"]

    for i, r in enumerate(requests):
        add_latencies_to_bins(
            bins=bins,
            pos=bin_start_position,
            responses=r["response_timestamps"],
            request_period=args.request_period,
        )
        bin_start_position = update_start_position(
            request_id=i,
            start_pos=bin_start_position,
            initial_requests=start,
            step=step,
        )
    return bins


def calculate_avg_periodic_latencies(args, profile_result, filename):
    """Calculate average token-to-token latency for each request period."""
    bins = collect_periodic_latencies(args, filename)

    latencies = []
    for bin in bins:
        latencies.append(np.mean(bin) / 1_000_000)

    profile_result.avg_periodic_t2t_latencies = latencies


def collect_latencies(requests):
    # Example json demonstrating format:
    #   see client/src/c++/perf_analyzer/docs/examples/decoupled_output_file.json
    first_token_latencies = []
    generation_latencies = []
    token_to_token_latencies = []
    requests = requests["experiments"][0]["requests"]
    for r in requests:
        init_request, responses = r["timestamp"], r["response_timestamps"]
        first_token_latencies.append((responses[0] - init_request) / 1_000_000)
        generation_latencies.append((responses[-1] - responses[0]) / 1_000_000)
        token_to_token_latencies = []
        for prev_res, res in pairwise(responses):
            token_to_token_latencies.append((res - prev_res) / 1_000_000)
    return first_token_latencies, generation_latencies, token_to_token_latencies


def calculate_online_metrics(args, profile_result, filename):
    """Calculate online metrics for more fine-grained performance information."""
    if not args.stream:
        return  # skip if offline

    requests = load_json_data(filename)
    latencies = collect_latencies(requests)
    first_token_latencies, generation_latencies, token_to_token_latencies = latencies

    profile_result.avg_first_token_latency = np.mean(first_token_latencies)
    profile_result.avg_total_t2t_latency = np.mean(token_to_token_latencies)

    profile_result.max_first_token_latency = max(first_token_latencies)
    profile_result.min_first_token_latency = min(first_token_latencies)
    profile_result.avg_first_token_latency = np.mean(first_token_latencies)
    profile_result.p50_first_token_latency = np.percentile(
        first_token_latencies, 50, method="lower"
    )
    profile_result.p90_first_token_latency = np.percentile(
        first_token_latencies, 90, method="lower"
    )
    profile_result.p95_first_token_latency = np.percentile(
        first_token_latencies, 95, method="lower"
    )
    profile_result.p99_first_token_latency = np.percentile(
        first_token_latencies, 99, method="lower"
    )

    profile_result.max_gen_latency = max(generation_latencies)
    profile_result.min_gen_latency = min(generation_latencies)
    profile_result.avg_gen_latency = np.mean(generation_latencies)
    profile_result.p50_gen_latency = np.percentile(
        generation_latencies, 50, method="lower"
    )
    profile_result.p90_gen_latency = np.percentile(
        generation_latencies, 90, method="lower"
    )
    profile_result.p95_gen_latency = np.percentile(
        generation_latencies, 95, method="lower"
    )
    profile_result.p99_gen_latency = np.percentile(
        generation_latencies, 99, method="lower"
    )

    token_latencies = [t / args.max_tokens for t in generation_latencies]
    profile_result.avg_token_latency = np.mean(token_latencies)


def collect_offline_metrics(requests, sequence_len):
    end_to_end_latencies = []
    throughputs = []
    requests = requests["experiments"][0]["requests"]

    for request in requests:
        total_time = request["response_timestamps"][-1] - request["timestamp"]
        time_s = total_time / 1_000_000_000  # sec
        time_ms = total_time / 1_000_000  # msec
        end_to_end_latencies.append(time_ms)
        throughputs.append(sequence_len / time_s)
    return throughputs, end_to_end_latencies


def calculate_offline_metrics(args, profile_result, filename):
    """Calculate offline metrics that show end-to-end performance."""
    requests = load_json_data(filename)
    throughputs, end_to_end_latencies = collect_offline_metrics(
        requests=requests, sequence_len=profile_result.prompt_size + args.max_tokens
    )

    profile_result.max_e2e_latency = max(end_to_end_latencies)
    profile_result.min_e2e_latency = min(end_to_end_latencies)
    profile_result.avg_e2e_latency = np.mean(end_to_end_latencies)
    profile_result.max_token_throughput = max(throughputs)
    profile_result.min_token_throughput = min(throughputs)
    profile_result.avg_token_throughput = np.mean(throughputs)
    profile_result.p50_token_throughput = np.percentile(throughputs, 50, method="lower")
    profile_result.p90_token_throughput = np.percentile(throughputs, 90, method="lower")
    profile_result.p95_token_throughput = np.percentile(throughputs, 95, method="lower")
    profile_result.p99_token_throughput = np.percentile(throughputs, 99, method="lower")


def summarize_profile_results(args, prompts):
    results = []
    for prompt in prompts:
        prompt_size = len(prompt.split())
        export_file = get_export_filename(args, prompt_size)

        profile_result = ProfileResults(prompt_size=prompt_size)
        calculate_offline_metrics(args, profile_result, export_file)
        calculate_online_metrics(args, profile_result, export_file)

        if args.periodic_concurrency_range:
            calculate_avg_periodic_latencies(args, profile_result, export_file)
            plot_results(
                latencies=profile_result.avg_periodic_t2t_latencies,
                filename=get_plot_filename(args, prompt_size),
            )
        results.append(profile_result)

    print_benchmark_summary(results)
    save_benchmark_results(args, results)

    if args.periodic_concurrency_range:
        print(
            "Saved in-flight batching benchmark plots "
            "@ 'inflight_batching_benchmark-*.png'."
        )


def profile(args, export_file):
    command = (
        f"perf_analyzer -m {args.model} -i grpc --async --streaming "
        f"--input-data={INPUT_FILENAME} "
        f"--profile-export-file={export_file} "
    )
    if args.periodic_concurrency_range:
        start, end, step = args.periodic_concurrency_range
        command += (
            f"--periodic-concurrency-range={start}:{end}:{step} "
            f"--request-period={args.request_period}"
        )
    else:
        command += (
            "--measurement-mode=count_windows "
            "--measurement-request-count=10 "
            "--stability-percentage=999"
        )

    print("Running Perf Analyzer...")
    subprocess.run(args=[command], shell=True)


def prepare_export_file(args, prompt):
    prompt_size = len(prompt.split())
    filename = get_export_filename(args, prompt_size)

    # If exists, clean up
    export_file = Path(filename)
    export_file.unlink(missing_ok=True)
    return export_file


def prepare_input_data(input_data, prompt):
    """Insert the prompt to send into input JSON data."""
    input_data["data"][0]["PROMPT"] = [prompt]
    save_json_data(input_data, INPUT_FILENAME)


def generate_prompts(args, input_data):
    """Generate dummy prompts if not specified by input JSON file."""
    prompt = input_data["data"][0]["PROMPT"][0]

    if not prompt:  # Generate dummy prompt
        assert args.prompt_size_range, "Must specify --prompt-size-range."
        start, end, step = args.prompt_size_range
        return [" ".join(["hi"] * size) for size in range(start, end + 1, step)]
    return [prompt]


def construct_input_data(args):
    """Construct input data that contains input tensors and parameters.

    Parse the input JSON file (if exists) to construct the input data.
    When user sets parameters through command line, overwrite the
    parameters set by input JSON file.
    """
    prompt = ""
    stream = False
    sampling_params = {}

    if args.input_data:
        data = load_json_data(filename=args.input_data)["data"][0]
        stream = data["STREAM"][0] if "STREAM" in data else stream
        prompt = data["PROMPT"][0] if "PROMPT" in data else prompt
        if "SAMPLING_PARAMETERS" in data:
            sampling_params = json.loads(data["SAMPLING_PARAMETERS"][0])

    # If command line option is specified, overwrite
    if args.stream:
        stream = args.stream
    else:
        args.stream = stream

    if args.max_tokens:
        sampling_params["max_tokens"] = args.max_tokens
    elif "max_tokens" in sampling_params:
        args.max_tokens = sampling_params["max_tokens"]
    else:
        args.max_tokens = 256  # default
        sampling_params["max_tokens"] = args.max_tokens

    if "ignore_eos" not in sampling_params:
        sampling_params["ignore_eos"] = args.ignore_eos
    elif args.ignore_eos:
        sampling_params["ignore_eos"] = True

    input_data = {"data": [{}]}
    input_data["data"][0]["PROMPT"] = [prompt]
    input_data["data"][0]["STREAM"] = [stream]
    input_data["data"][0]["SAMPLING_PARAMETERS"] = [json.dumps(sampling_params)]
    return input_data


def main(args):
    input_data = construct_input_data(args)
    prompts = generate_prompts(args, input_data)

    for prompt in prompts:
        prepare_input_data(input_data, prompt)
        export_file = prepare_export_file(args, prompt)

        # Run Perf Analyzer
        profile(args, export_file)

    summarize_profile_results(args, prompts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="vllm",
        help="The name of the model to profile.",
    )
    parser.add_argument(
        "--prompt-size-range",
        type=int,
        nargs=3,
        metavar=("START", "END", "STEP"),
        help="The range of prompt sizes '<[START, END], STEP>' where END is inclusive.",
    )
    parser.add_argument(
        "--periodic-concurrency-range",
        type=int,
        nargs=3,
        metavar=("START", "END", "STEP"),
        help="The range of concurrency level that periodically increases until it reaches END.",
    )
    parser.add_argument(
        "--request-period",
        type=int,
        default=10,
        help="The number of responses that each request must receive before launching new requests.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="The maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Whether to ignore end-of-sequence token.",
    )
    parser.add_argument(
        "--input-data",
        type=str,
        help="The input data file to be used for inference request.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Whether to stream the model outputs.",
    )
    args = parser.parse_args()
    main(args)
