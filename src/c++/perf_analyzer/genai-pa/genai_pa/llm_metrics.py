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
import csv
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


@dataclass
class LLMMetrics:
    """A simple dataclass that holds core LLM performance metrics."""

    time_to_first_tokens: list[int]
    inter_token_latencies: list[int]
    output_token_throughputs: list[int]

    metric_labels = [
        "time_to_first_token",
        "inter_token_latency",
    ]

    time_fields = [
        "inter_token_latency",
        "time_to_first_token",
        "end_to_end_latency",
    ]

    def get_base_name(self, attr_name: str) -> str:
        # Attempted to extract and store the mapping as a dataclass member as a
        # dictionary but encountered two issues: (1) Python does not allow
        # dataclass member to be mutable and (2) if we set it as member of
        # normal class, the dict member will be parsed by Statistics class,
        # which is not what we want since it's not one of the LLM metrics.
        # Leaving it as conditional statements for now.
        if attr_name == "time_to_first_tokens":
            return "time_to_first_token"
        elif attr_name == "inter_token_latencies":
            return "inter_token_latency"
        elif attr_name == "output_token_throughputs":
            return "output_token_throughput"
        else:
            raise ValueError(f"No attribute named '{attr_name}' exists.")


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

      >>> metrics = LLMMetrics([3, 4, 5], [], [])
      >>> stats = Statistics(metrics)
      >>> print(stats.avg_time_to_first_token)  # output: 4
    """

    def __init__(self, metrics: LLMMetrics):
        # iterate through LLMMetrics to calculate statistics and set attributes
        for attr, data in metrics.__dict__.items():
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
        return field in LLMMetrics.time_fields

    def pretty_print(self):
        table = Table(title="PA LLM Metrics")

        table.add_column("Statistic", justify="right", style="cyan", no_wrap=True)
        stats = ["avg", "min", "max", "p99", "p90", "p75"]
        for stat in stats:
            table.add_column(stat, justify="right", style="green")

        for metric in LLMMetrics.metric_labels:
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

    def export_to_csv(self, csv_filename: str):
        header = [
            "Statistic",
            "avg",
            "min",
            "max",
            "p99",
            "p95",
            "p90",
            "p75",
            "p50",
            "p25",
        ]

        with open(csv_filename, mode="w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(header)

            for metric in LLMMetrics.metric_labels:
                formatted_metric = metric
                is_time_field = self._is_time_field(metric)
                if is_time_field:
                    formatted_metric += "(ns)"

                row_values = [formatted_metric]

                for stat in header[1:]:
                    value = self.__dict__.get(f"{stat}_{metric}", -1)
                    row_values.append(f"{value:.0f}")

                csv_writer.writerow(row_values)


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
        time_to_first_tokens = []
        inter_token_latencies = []
        output_token_throughputs = []
        for request in requests:
            req_timestamp = request["timestamp"]
            res_timestamps = request["response_timestamps"]
            # res_outputs = request["response_outputs"]

            # time to first token
            time_to_first_tokens.append(res_timestamps[0] - req_timestamp)

            # output token throughput
            # output_tokens = tokenizer(res_outputs)["input_ids"]
            # total_output_tokens = np.sum(list(map(len, output_tokens)))
            # req_latency = res_timestamps[-1] - req_timestamp
            # output_token_throughputs.append(total_output_tokens / req_latency)

            # inter token latency
            for t1, t2 in pairwise(res_timestamps):
                inter_token_latencies.append(t2 - t1)

        return LLMMetrics(
            time_to_first_tokens,
            inter_token_latencies,
            output_token_throughputs,
        )

    def get_statistics(self, infer_mode: str, load_level: int | float) -> Statistics:
        if (infer_mode, load_level) not in self._profile_results:
            raise KeyError(f"Profile with {infer_mode}={load_level} does not exist.")
        return self._profile_results[(infer_mode, load_level)]
