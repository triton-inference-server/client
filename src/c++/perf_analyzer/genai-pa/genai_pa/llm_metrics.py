#!/usr/bin/env python3

# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import contextlib
import io
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from genai_pa.utils import load_json
from rich.console import Console
from rich.table import Table

# Silence tokenizer warning on import
with contextlib.redirect_stdout(io.StringIO()) as stdout, contextlib.redirect_stderr(
    io.StringIO()
) as stderr:
    from transformers import AutoTokenizer


class Metrics:
    """A base class for all the metrics class that contains common metrics."""

    def __init__(
        self,
        request_throughputs: list[float] = [],
        request_latencies: list[int] = [],
    ) -> None:
        self.request_throughputs = request_throughputs
        self.request_latencies = request_latencies
        self._base_names = {
            "request_throughputs": "request_throughput",
            "request_latencies": "request_latency",
        }

    @property
    def data(self) -> dict:
        """Returns all the metrics."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def get_base_name(self, metric_name: str) -> str:
        """Returns singular name of a given metric."""
        if metric_name in self._base_names:
            return self._base_names[metric_name]
        else:
            raise KeyError(f"No metric named '{metric_name}' exists.")


class LLMMetrics(Metrics):
    """A simple dataclass that holds core LLM performance metrics."""

    def __init__(
        self,
        request_throughputs: list[float] = [],
        request_latencies: list[int] = [],
        time_to_first_tokens: list[int] = [],
        inter_token_latencies: list[int] = [],
        output_token_throughputs: list[int] = [],
    ) -> None:
        super().__init__(request_throughputs, request_latencies)
        self.time_to_first_tokens = time_to_first_tokens
        self.inter_token_latencies = inter_token_latencies
        self.output_token_throughputs = output_token_throughputs

        # add base name mapping
        self._base_names["time_to_first_tokens"] = "time_to_first_token"
        self._base_names["inter_token_latencies"] = "inter_token_latency"
        self._base_names["output_token_throughputs"] = "output_token_throughput"


class Statistics:
    """A class that aggregates various statistics from given metrics class.

    The Statistics class goes through each metric in the metrics class and
    calculates several statistics such as:
      - average (arithmetic mean)
      - percentiles (p25, p50, p75, p90, p95, p99)
      - minimum & maximum
      - standard deviation
    The class will store each calculated statistics as part of its attribute.

    Example:

      >>> metrics = LLMMetrics(request_throughputs=[2, 4])
      >>> stats = Statistics(metrics)
      >>> print(stats.avg_request_throughput)  # output: 3
    """

    def __init__(self, metrics: Metrics):
        # iterate through Metrics to calculate statistics and set attributes
        for attr, data in metrics.data.items():
            if data:
                attr = metrics.get_base_name(attr)
                self._calculate_mean(data, attr)
                self._calculate_percentiles(data, attr)
                self._calculate_minmax(data, attr)
                self._calculate_std(data, attr)

    def _calculate_mean(self, data: list[int], attr: str):
        avg = np.mean(data)
        setattr(self, "avg_" + attr, avg)

    def _calculate_percentiles(self, data: list[int], attr: str):
        p25, p50, p75 = np.percentile(data, [25, 50, 75])
        p90, p95, p99 = np.percentile(data, [90, 95, 99])
        setattr(self, "p25_" + attr, p25)
        setattr(self, "p50_" + attr, p50)
        setattr(self, "p75_" + attr, p75)
        setattr(self, "p90_" + attr, p90)
        setattr(self, "p95_" + attr, p95)
        setattr(self, "p99_" + attr, p99)

    def _calculate_minmax(self, data: list[int], attr: str):
        min, max = np.min(data), np.max(data)
        setattr(self, "min_" + attr, min)
        setattr(self, "max_" + attr, max)

    def _calculate_std(self, data: list[int], attr: str):
        std = np.std(data)
        setattr(self, "std_" + attr, std)

    def __repr__(self):
        attr_strs = ",".join([f"{k}={v}" for k, v in self.__dict__.items()])
        return f"Statistics({attr_strs})"

    def _is_time_field(self, field: str):
        time_fields = [
            "inter_token_latency",
            "time_to_first_token",
            "end_to_end_latency",
        ]
        return field in time_fields

    def pretty_print(self):
        table = Table(title="PA LLM Metrics")

        table.add_column("Statistic", justify="right", style="cyan", no_wrap=True)
        stats = ["avg", "min", "max", "p99", "p95", "p90", "p75", "p50", "p25"]
        for stat in stats:
            table.add_column(stat, justify="right", style="green")

        metrics = ["inter_token_latency", "time_to_first_token"]
        for metric in metrics:
            formatted_metric = metric.replace("_", " ").capitalize()
            is_time_field = self._is_time_field(metric)
            if is_time_field:
                formatted_metric += " (ns)"
            row_values = [formatted_metric]

            for stat in stats:
                value = self.__dict__.get(f"{stat}_{metric}", -1)
                row_values.append("{:,.0f}".format(value))
            table.add_row(*row_values)

        console = Console()
        console.print(table)


class LLMProfileData:
    """A class that calculates and aggregates all the LLM performance statistics
    across the Perf Analyzer profile results.

    The LLMProfileData class parses profile export JSON file, collects the core
    LLM performance metrics, and calculates summary statistics for each different
    Perf Analyzer runs/experiments.

    Example:

      >>> ... # run Perf Analyzer with concurrency level 10
      >>>
      >>> from transformers import AutoTokenizer
      >>>
      >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
      >>> pd = LLMProfileData(filename="profile_export.json", tokenizer)
      >>> stats = pd.get_statistics(infer_mode="concurrency", level=10)
      >>>
      >>> print(stats)  # output: Statistics(avg_time_to_first_token=...)
      >>> stats.pretty_print()  # Output: time_to_first_token_s: ...
    """

    def __init__(self, filename: str, tokenizer: AutoTokenizer) -> None:
        data = load_json(filename)
        self._profile_results = {}

        for experiment in data["experiments"]:
            infer_mode = experiment["experiment"]["mode"]
            load_level = experiment["experiment"]["value"]
            requests = experiment["requests"]

            metrics = self._collect_llm_metrics(requests, tokenizer)

            # aggregate and calculate statistics
            statistics = Statistics(metrics)
            self._profile_results[(infer_mode, load_level)] = statistics

    def _collect_llm_metrics(
        self, requests: dict, tokenizer: AutoTokenizer
    ) -> LLMMetrics:
        min_req_timestamp, max_res_timestamp = float("inf"), 0
        request_latencies = []
        time_to_first_tokens = []
        inter_token_latencies = []
        output_token_throughputs = []
        for request in requests:
            req_timestamp = request["timestamp"]
            res_timestamps = request["response_timestamps"]
            res_outputs = request["response_outputs"]

            # track entire benchmark duration
            min_req_timestamp = min(min_req_timestamp, req_timestamp)
            max_res_timestamp = max(max_res_timestamp, res_timestamps[-1])

            # request latencies
            req_latency = res_timestamps[-1] - req_timestamp
            request_latencies.append(req_latency)

            # time to first token
            time_to_first_tokens.append(res_timestamps[0] - req_timestamp)

            # output token throughput
            output_tokens = tokenizer(res_outputs)["input_ids"]
            num_output_tokens = list(map(len, output_tokens))
            total_output_tokens = np.sum(num_output_tokens)
            output_token_throughputs.append(total_output_tokens / req_latency)

            # inter token latency
            for (t1, _), (t2, n2) in pairwise(zip(res_timestamps, num_output_tokens)):
                inter_token_latencies.append(round((t2 - t1) / n2))

        # request throughput
        benchmark_duration = max_res_timestamp - min_req_timestamp
        request_throughputs = [len(requests) / benchmark_duration]

        return LLMMetrics(
            request_throughputs,
            request_latencies,
            time_to_first_tokens,
            inter_token_latencies,
            output_token_throughputs,
        )

    def get_statistics(self, infer_mode: str, load_level: int | float) -> Statistics:
        if (infer_mode, load_level) not in self._profile_results:
            raise KeyError(f"Profile with {infer_mode}={load_level} does not exist.")
        return self._profile_results[(infer_mode, load_level)]
