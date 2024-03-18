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
import json
from itertools import pairwise

import numpy as np
from genai_perf.utils import load_json, remove_sse_prefix
from rich.console import Console
from rich.table import Table

# Silence tokenizer warning on import
with contextlib.redirect_stdout(io.StringIO()) as stdout, contextlib.redirect_stderr(
    io.StringIO()
) as stderr:
    from transformers import AutoTokenizer


class Metrics:
    """A base class for all the metrics class that contains common metrics."""

    metric_labels = [
        "time_to_first_token",
        "inter_token_latency",
        "request_latency",
        "output_token_throughput",
        "output_token_throughput_per_request",
        "request_throughput",
        "num_output_token",
    ]

    time_fields = [
        "inter_token_latency",
        "time_to_first_token",
        "request_latency",
    ]

    # TODO (TMA-1678): output_token_throughput_per_request is not on this list
    # since the current code treats all the throughput metrics to be displayed
    # outside of the statistics table.
    throughput_fields = [
        "request_throughput",
        "output_token_throughput",
    ]

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
        output_token_throughputs: list[float] = [],
        output_token_throughputs_per_request: list[int] = [],
        num_output_tokens: list[int] = [],
    ) -> None:
        super().__init__(request_throughputs, request_latencies)
        self.time_to_first_tokens = time_to_first_tokens
        self.inter_token_latencies = inter_token_latencies
        self.output_token_throughputs = output_token_throughputs
        self.output_token_throughputs_per_request = output_token_throughputs_per_request
        self.num_output_tokens = num_output_tokens

        # add base name mapping
        self._base_names["time_to_first_tokens"] = "time_to_first_token"
        self._base_names["inter_token_latencies"] = "inter_token_latency"
        self._base_names["output_token_throughputs"] = "output_token_throughput"
        self._base_names[
            "output_token_throughputs_per_request"
        ] = "output_token_throughput_per_request"
        self._base_names["num_output_tokens"] = "num_output_token"


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

    def _calculate_mean(self, data: list[int | float], attr: str):
        avg = np.mean(data)
        setattr(self, "avg_" + attr, avg)

    def _calculate_percentiles(self, data: list[int | float], attr: str):
        p25, p50, p75 = np.percentile(data, [25, 50, 75])
        p90, p95, p99 = np.percentile(data, [90, 95, 99])
        setattr(self, "p25_" + attr, p25)
        setattr(self, "p50_" + attr, p50)
        setattr(self, "p75_" + attr, p75)
        setattr(self, "p90_" + attr, p90)
        setattr(self, "p95_" + attr, p95)
        setattr(self, "p99_" + attr, p99)

    def _calculate_minmax(self, data: list[int | float], attr: str):
        min, max = np.min(data), np.max(data)
        setattr(self, "min_" + attr, min)
        setattr(self, "max_" + attr, max)

    def _calculate_std(self, data: list[int | float], attr: str):
        std = np.std(data)
        setattr(self, "std_" + attr, std)

    def __repr__(self):
        attr_strs = ",".join([f"{k}={v}" for k, v in self.__dict__.items()])
        return f"Statistics({attr_strs})"

    def _is_throughput_field(self, field: str):
        return field in Metrics.throughput_fields

    def _is_time_field(self, field: str):
        return field in Metrics.time_fields

    def pretty_print(self):
        """Prints the statistics in a tabular format."""

        singular_metric_rows = []
        table = Table(title="PA LLM Metrics")

        table.add_column("Statistic", justify="right", style="cyan", no_wrap=True)
        stats = ["avg", "min", "max", "p99", "p90", "p75"]
        for stat in stats:
            table.add_column(stat, justify="right", style="green")

        for metric in Metrics.metric_labels:
            formatted_metric = metric.replace("_", " ").capitalize()

            # Throughput fields are printed after the table
            is_throughput_field = self._is_throughput_field(metric)
            if is_throughput_field:
                value = self.__dict__.get(f"{stats[0]}_{metric}", -1)
                formatted_metric += f" (per sec): {value:.2f}"
                singular_metric_rows.append(formatted_metric)
                continue

            # TODO (TMA-1712): need to decide if we need this metric. Remove
            # from statistics display for now.
            # TODO (TMA-1678): output_token_throughput_per_request is treated
            # separately since the current code treats all throughput metrics to
            # be displayed outside of the statistics table.
            if metric == "output_token_throughput_per_request":
                formatted_metric += f" (per sec)"
                continue

            is_time_field = self._is_time_field(metric)
            if is_time_field:
                formatted_metric += " (ns)"

            row_values = [formatted_metric]

            for stat in stats:
                value = self.__dict__.get(f"{stat}_{metric}", -1)
                row_values.append(f"{value:,.0f}")

            # Without streaming, there is no inter-token latency available, so do not print it.
            if metric == "inter_token_latency":
                if all(value == "-1" for value in row_values[1:]):
                    continue
            # Without streaming, TTFT and request latency are the same, so do not print TTFT.
            elif metric == "time_to_first_token":
                unique_values = False
                for stat in stats:
                    value_ttft = self.__dict__.get(f"{stat}_{metric}", -1)
                    value_req_latency = self.__dict__.get(f"{stat}_request_latency", -1)
                    if value_ttft != value_req_latency:
                        unique_values = True
                        break
                if not unique_values:
                    continue

            table.add_row(*row_values)

        console = Console()
        console.print(table)

        for row in singular_metric_rows:
            print(row)

    def export_to_csv(self, csv_filename: str):
        """Exports the statistics to a CSV file."""

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
            singular_metric_rows = []

            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(header)

            for metric in Metrics.metric_labels:
                formatted_metric = metric

                is_throughput_field = self._is_throughput_field(metric)
                is_time_field = self._is_time_field(metric)

                if is_time_field:
                    formatted_metric += "(ns)"
                elif is_throughput_field:
                    formatted_metric += "(per sec)"
                # TODO (TMA-1712): need to decide if we need this metric. Do not
                # include in the csv for now.
                # TODO (TMA-1678): output_token_throughput_per_request is treated
                # separately since the current code treats all throughput metrics
                # to be displayed outside of the statistics table.
                elif metric == "output_token_throughput_per_request":
                    formatted_metric += "(per sec)"
                    continue

                row_values = [formatted_metric]

                if is_throughput_field:
                    value = self.__dict__.get(f"{header[1]}_{metric}", -1)
                    row_values.append(f"{value:.2f}")
                    singular_metric_rows.append(row_values)
                    continue

                for stat in header[1:]:
                    value = self.__dict__.get(f"{stat}_{metric}", -1)
                    row_values.append(f"{value:.0f}")

                # Without streaming, there is no inter-token latency available, so do not print it.
                if metric == "inter_token_latency":
                    if all(value == "-1" for value in row_values[1:]):
                        continue
                # Without streaming, TTFT and request latency are the same, so do not print TTFT.
                elif metric == "time_to_first_token":
                    unique_values = False
                    for stat in header[1:]:
                        value_ttft = self.__dict__.get(f"{stat}_{metric}", -1)
                        value_req_latency = self.__dict__.get(
                            f"{stat}_request_latency", -1
                        )
                        if value_ttft != value_req_latency:
                            unique_values = True
                            break
                    if not unique_values:
                        continue

                csv_writer.writerow(row_values)

            for row in singular_metric_rows:
                csv_writer.writerow(row)


class ProfileDataParser:
    """Base profile data parser class that reads the profile data JSON file to
    extract core metrics and calculate various performance statistics.
    """

    def __init__(self, filename: str) -> None:
        data = load_json(filename)
        self._parse_profile_data(data)

    def _parse_profile_data(self, data: dict) -> None:
        """Parse through the entire profile data to collect statistics."""
        self._profile_results = {}
        for experiment in data["experiments"]:
            infer_mode = experiment["experiment"]["mode"]
            load_level = experiment["experiment"]["value"]
            requests = experiment["requests"]

            metrics = self._parse_requests(requests)

            # aggregate and calculate statistics
            statistics = Statistics(metrics)
            self._profile_results[(infer_mode, str(load_level))] = statistics

    def _parse_requests(self, requests: dict) -> LLMMetrics:
        """Parse each request in profile data to extract core metrics."""
        raise NotImplementedError

    def get_statistics(self, infer_mode: str, load_level: str) -> Statistics:
        """Return profile statistics if it exists."""
        if (infer_mode, load_level) not in self._profile_results:
            raise KeyError(f"Profile with {infer_mode}={load_level} does not exist.")
        return self._profile_results[(infer_mode, load_level)]


class LLMProfileDataParser(ProfileDataParser):
    """A class that calculates and aggregates all the LLM performance statistics
    across the Perf Analyzer profile results.

    The LLMProfileDataParser class parses profile export JSON file, collects the
    core LLM performance metrics, and calculates summary statistics for each
    different Perf Analyzer runs/experiments.

    Example:

      >>> ... # run Perf Analyzer with concurrency level 10
      >>>
      >>> from transformers import AutoTokenizer
      >>>
      >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
      >>> pd = LLMProfileDataParser(
      >>>     filename="profile_export.json",
      >>>     service_kind="triton",
      >>>     tokenizer=tokenizer,
      >>> )
      >>> stats = pd.get_statistics(infer_mode="concurrency", level=10)
      >>>
      >>> print(stats)  # output: Statistics(avg_time_to_first_token=...)
      >>> stats.pretty_print()  # Output: time_to_first_token_s: ...
    """

    def __init__(
        self, filename: str, service_kind: str, tokenizer: AutoTokenizer
    ) -> None:
        self._tokenizer = tokenizer
        self._service_kind = service_kind
        super().__init__(filename)

    def _parse_requests(self, requests: dict) -> LLMMetrics:
        """Parse each requests in profile export data to extract key metrics."""
        min_req_timestamp, max_res_timestamp = float("inf"), 0
        request_latencies = []
        time_to_first_tokens = []
        inter_token_latencies = []
        output_token_throughputs_per_request = []
        num_generated_tokens = []
        for request in requests:
            req_timestamp = request["timestamp"]
            res_timestamps = request["response_timestamps"]
            res_outputs = request["response_outputs"]

            self._preprocess_response(res_timestamps, res_outputs)

            # track entire benchmark duration
            min_req_timestamp = min(min_req_timestamp, req_timestamp)
            max_res_timestamp = max(max_res_timestamp, res_timestamps[-1])

            # request latencies
            req_latency = res_timestamps[-1] - req_timestamp
            request_latencies.append(req_latency)  # nanosec
            req_latency = req_latency / 1e9  # sec

            # time to first token
            time_to_first_tokens.append(res_timestamps[0] - req_timestamp)

            # output token throughput per request
            output_tokens = self._tokenize_response_outputs(res_outputs)
            num_output_tokens = list(map(len, output_tokens))
            total_output_tokens = np.sum(num_output_tokens)
            output_token_throughputs_per_request.append(
                total_output_tokens / req_latency
            )
            num_generated_tokens.append(total_output_tokens)

            # inter token latency
            for (t1, _), (t2, n2) in pairwise(zip(res_timestamps, num_output_tokens)):
                # TMA-1676: handle empty first/last responses
                # if the latter response has zero token (e.g. empty string),
                # then set it default to one for the sake of inter token latency
                # calculation and to avoid divide by zero.
                num_token = 1 if n2 == 0 else n2
                inter_token_latencies.append(round((t2 - t1) / num_token))

        # request & output token throughput
        benchmark_duration = (max_res_timestamp - min_req_timestamp) / 1e9  # nanosec
        request_throughputs = [len(requests) / benchmark_duration]
        output_token_throughputs = [sum(num_generated_tokens) / benchmark_duration]

        return LLMMetrics(
            request_throughputs,
            request_latencies,
            time_to_first_tokens,
            inter_token_latencies,
            output_token_throughputs,
            output_token_throughputs_per_request,
            num_generated_tokens,
        )

    def _remove_leading_invalid_chars(self, text: str):
        if len(text) < 4:
            return text

        for i, char in enumerate(text):
            # There will be 3 or 4 chars
            # (but sometimes the first char looks valid, so don't stop until we've seen at least 3)
            if char.isprintable() and i > 2:
                break

        return text[i:]

    def _preprocess_response(
        self, res_timestamps: list[int], res_outputs: list[dict[str, str]]
    ) -> None:
        """Helper function to preprocess responses of a request."""
        # FIXME -- remove this triton code once it is properly fixed in PA
        # (PA/triton will add junk to the start of the BYTES array. Remove it here)
        if self._service_kind == "triton":
            for d in res_outputs:
                d["text_output"] = self._remove_leading_invalid_chars(d["text_output"])
        elif self._service_kind == "openai":
            # remove the null final response in streaming mode
            last_response = res_outputs[-1]["response"]
            last_response = remove_sse_prefix(last_response)
            if last_response == "[DONE]":
                res_timestamps.pop()
                res_outputs.pop()

            # after removing the final null response, check if the last response
            # of the remaining responses is missing text/content and remove it
            # if it is an empty response.
            last_response = res_outputs[-1]["response"]
            text_output = self._extract_openai_text_output(last_response)
            if text_output == "":
                res_timestamps.pop()
                res_outputs.pop()

    def _tokenize_response_outputs(self, res_outputs: dict) -> list[list[int]]:
        """Deserialize the response output and return tokenized outputs."""
        if self._service_kind == "triton":
            return self._tokenize_triton_response_output(res_outputs)
        elif self._service_kind == "openai":
            return self._tokenize_openai_response_output(res_outputs)
        else:
            raise ValueError(f"Unknown service kind: '{self._service_kind}'.")

    def _tokenize_triton_response_output(self, res_outputs: dict) -> list[list[int]]:
        """Tokenize the Triton response output texts."""
        output_texts = []
        for output in res_outputs:
            output_texts.append(output["text_output"])
        return self._tokenizer(output_texts)["input_ids"]

    def _tokenize_openai_response_output(self, res_outputs: dict) -> list[list[int]]:
        """Tokenize the OpenAI response output texts."""
        output_texts = []
        for output in res_outputs:
            text = self._extract_openai_text_output(output["response"])
            output_texts.append(text)
        return self._tokenizer(output_texts)["input_ids"]

    def _extract_openai_text_output(self, response: str) -> str:
        """Extracts text/content of the OpenAI response object."""
        response = remove_sse_prefix(response)
        data = json.loads(response)
        completions = data["choices"][0]

        text_output = ""
        if data["object"] == "text_completion":  # legacy
            text_output = completions.get("text", "")
        elif data["object"] == "chat.completion":  # non-streaming
            text_output = completions["message"]["content"]
        elif data["object"] == "chat.completion.chunk":  # streaming
            text_output = completions["delta"].get("content", "")
        else:
            obj_type = data["object"]
            raise ValueError(f"Unknown OpenAI response object type '{obj_type}'.")
        return text_output
