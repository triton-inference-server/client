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

import csv
import json
from collections import defaultdict
from enum import Enum, auto
from itertools import tee
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from genai_perf.tokenizer import Tokenizer
from genai_perf.utils import load_json, remove_sse_prefix
from rich.console import Console
from rich.table import Table


class ResponseFormat(Enum):
    OPENAI_CHAT_COMPLETIONS = auto()
    OPENAI_COMPLETIONS = auto()
    TRITON = auto()


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
        "num_input_token",
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
        request_throughputs: List[float] = [],
        request_latencies: List[int] = [],
    ) -> None:
        self.request_throughputs = request_throughputs
        self.request_latencies = request_latencies
        self._base_names = {
            "request_throughputs": "request_throughput",
            "request_latencies": "request_latency",
        }

    def __repr__(self):
        attr_strs = []
        for k, v in self.__dict__.items():
            if not k.startswith("_"):
                attr_strs.append(f"{k}={v}")
        return f"Metrics({','.join(attr_strs)})"

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
        request_throughputs: List[float] = [],
        request_latencies: List[int] = [],
        time_to_first_tokens: List[int] = [],
        inter_token_latencies: List[int] = [],
        output_token_throughputs: List[float] = [],
        output_token_throughputs_per_request: List[int] = [],
        num_output_tokens: List[int] = [],
        num_input_tokens: List[int] = [],
        chunked_inter_token_latencies: List[List[int]] = [[]],
    ) -> None:
        super().__init__(request_throughputs, request_latencies)
        self.time_to_first_tokens = time_to_first_tokens
        self.inter_token_latencies = inter_token_latencies
        self.output_token_throughputs = output_token_throughputs
        self.output_token_throughputs_per_request = output_token_throughputs_per_request
        self.num_output_tokens = num_output_tokens
        self.num_input_tokens = num_input_tokens

        # Keeping chunked ITL (old) as a WAR to preserve visualization.
        # Excluded from data.
        self._chunked_inter_token_latencies = chunked_inter_token_latencies

        # add base name mapping
        self._base_names["time_to_first_tokens"] = "time_to_first_token"
        self._base_names["inter_token_latencies"] = "inter_token_latency"
        self._base_names["output_token_throughputs"] = "output_token_throughput"
        self._base_names["output_token_throughputs_per_request"] = (
            "output_token_throughput_per_request"
        )
        self._base_names["num_output_tokens"] = "num_output_token"
        self._base_names["num_input_tokens"] = "num_input_token"


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
        self._metrics = metrics
        self._stats_dict: Dict = defaultdict(dict)
        for attr, data in metrics.data.items():
            if self._should_skip(data, attr):
                continue

            attr = metrics.get_base_name(attr)
            self._add_units(attr)
            self._calculate_mean(data, attr)
            if not self._is_throughput_field(attr):
                self._calculate_percentiles(data, attr)
                self._calculate_minmax(data, attr)
                self._calculate_std(data, attr)

    def _should_skip(self, data: List[Union[int, float]], attr: str) -> bool:
        """Checks if some metrics should be skipped."""
        # No data points
        if len(data) == 0:
            return True
        # Skip ITL when non-streaming (all zero)
        elif attr == "inter_token_latencies" and sum(data) == 0:
            return True
        return False

    def _calculate_mean(self, data: List[Union[int, float]], attr: str) -> None:
        avg = np.mean(data)
        if self._is_time_field(attr):
            avg = self._scale(float(avg))
        self._stats_dict[attr]["avg"] = float(avg)
        setattr(self, "avg_" + attr, avg)

    def _calculate_percentiles(self, data: List[Union[int, float]], attr: str) -> None:
        p25, p50, p75 = np.percentile(data, [25, 50, 75])
        p90, p95, p99 = np.percentile(data, [90, 95, 99])
        if self._is_time_field(attr):
            p25 = self._scale(float(p25))
            p50 = self._scale(float(p50))
            p75 = self._scale(float(p75))
            p90 = self._scale(float(p90))
            p95 = self._scale(float(p95))
            p99 = self._scale(float(p99))
        self._stats_dict[attr]["p99"] = float(p99)
        self._stats_dict[attr]["p95"] = float(p95)
        self._stats_dict[attr]["p90"] = float(p90)
        self._stats_dict[attr]["p75"] = float(p75)
        self._stats_dict[attr]["p50"] = float(p50)
        self._stats_dict[attr]["p25"] = float(p25)
        setattr(self, "p25_" + attr, p25)
        setattr(self, "p50_" + attr, p50)
        setattr(self, "p75_" + attr, p75)
        setattr(self, "p90_" + attr, p90)
        setattr(self, "p95_" + attr, p95)
        setattr(self, "p99_" + attr, p99)

    def _calculate_minmax(self, data: List[Union[int, float]], attr: str) -> None:
        min, max = np.min(data), np.max(data)
        if self._is_time_field(attr):
            min = self._scale(float(min))
            max = self._scale(float(max))
        self._stats_dict[attr]["max"] = float(max)
        self._stats_dict[attr]["min"] = float(min)
        setattr(self, "min_" + attr, min)
        setattr(self, "max_" + attr, max)

    def _calculate_std(self, data: List[Union[int, float]], attr: str) -> None:
        std = np.std(data)
        if self._is_time_field(attr):
            std = self._scale(float(std))
        self._stats_dict[attr]["std"] = float(std)
        setattr(self, "std_" + attr, std)

    def _scale(self, metric: float, factor: float = 1 / 1e6) -> float:
        """
        Scale metrics from nanoseconds to milliseconds
        """
        return metric * factor

    def _add_units(self, key) -> None:
        if self._is_time_field(key):
            self._stats_dict[key]["unit"] = "ms"
        if key == "request_throughput":
            self._stats_dict[key]["unit"] = "requests/sec"
        if key.startswith("output_token_throughput"):
            self._stats_dict[key]["unit"] = "tokens/sec"
        if key == "num_input_token" or key == "num_output_token":
            self._stats_dict[key]["unit"] = "tokens"

    def __repr__(self) -> str:
        attr_strs = []
        for k, v in self.__dict__.items():
            if not k.startswith("_"):
                attr_strs.append(f"{k}={v}")
        return f"Statistics({','.join(attr_strs)})"

    @property
    def data(self) -> dict:
        """Return all the aggregated statistics."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    @property
    def metrics(self) -> Metrics:
        """Return the underlying metrics used to calculate the statistics."""
        return self._metrics

    @property
    def stats_dict(self) -> Dict:
        return self._stats_dict

    def _is_throughput_field(self, field: str) -> bool:
        return field in Metrics.throughput_fields

    def _is_time_field(self, field: str) -> bool:
        return field in Metrics.time_fields

    def export_parquet(self, artifact_dir: Path, filename: str) -> None:
        max_length = -1
        col_index = 0
        filler_list = []
        df = pd.DataFrame()

        # Data frames require all columns of the same length
        # find the max length column
        for key, value in self._metrics.data.items():
            max_length = max(max_length, len(value))

        # Insert None for shorter columns to match longest column
        for key, value in self._metrics.data.items():
            if len(value) < max_length:
                diff = max_length - len(value)
                filler_list = [None] * diff
            df.insert(col_index, key, value + filler_list)
            diff = 0
            filler_list = []
            col_index = col_index + 1

        filepath = artifact_dir / f"{filename}.gzip"
        df.to_parquet(filepath, compression="gzip")


class ProfileDataParser:
    """Base profile data parser class that reads the profile data JSON file to
    extract core metrics and calculate various performance statistics.
    """

    def __init__(self, filename: Path) -> None:
        data = load_json(filename)
        self._get_profile_metadata(data)
        self._parse_profile_data(data)

    def _get_profile_metadata(self, data: dict) -> None:
        self._service_kind = data["service_kind"]
        if self._service_kind == "openai":
            if data["endpoint"] == "v1/chat/completions":
                self._response_format = ResponseFormat.OPENAI_CHAT_COMPLETIONS
            elif data["endpoint"] == "v1/completions":
                self._response_format = ResponseFormat.OPENAI_COMPLETIONS
            else:
                # TPA-66: add PA metadata to handle this case
                # When endpoint field is either empty or custom endpoint, fall
                # back to parsing the response to extract the response format.
                request = data["experiments"][0]["requests"][0]
                response = request["response_outputs"][0]["response"]
                if "chat.completion" in response:
                    self._response_format = ResponseFormat.OPENAI_CHAT_COMPLETIONS
                elif "text_completion" in response:
                    self._response_format = ResponseFormat.OPENAI_COMPLETIONS
                else:
                    raise RuntimeError("Unknown OpenAI response format.")

        elif self._service_kind == "triton":
            self._response_format = ResponseFormat.TRITON
        else:
            raise ValueError(f"Unknown service kind: {self._service_kind}")

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

    def get_profile_load_info(self) -> List[Tuple[str, str]]:
        """Return available (infer_mode, load_level) tuple keys."""
        return [k for k, _ in self._profile_results.items()]


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
      >>>     tokenizer=tokenizer,
      >>> )
      >>> stats = pd.get_statistics(infer_mode="concurrency", level=10)
      >>>
      >>> print(stats)  # output: Statistics(avg_time_to_first_token=...)
      >>> stats.pretty_print()  # Output: time_to_first_token_s: ...
    """

    def __init__(
        self,
        filename: Path,
        tokenizer: Tokenizer,
    ) -> None:
        self._tokenizer = tokenizer
        super().__init__(filename)

    def _parse_requests(self, requests: dict) -> LLMMetrics:
        """Parse each requests in profile export data to extract key metrics."""
        min_req_timestamp, max_res_timestamp = float("inf"), 0
        request_latencies = []
        time_to_first_tokens = []
        inter_token_latencies = []
        output_token_throughputs_per_request = []
        num_input_tokens = []
        num_output_tokens = []
        chunked_inter_token_latencies = []

        for request in requests:
            req_timestamp = request["timestamp"]
            req_inputs = request["request_inputs"]
            res_timestamps = request["response_timestamps"]
            res_outputs = request["response_outputs"]

            self._preprocess_response(res_timestamps, res_outputs)

            # Skip requests with empty response. This happens sometimes when the
            # model returns a single response with empty string.
            if not res_timestamps:
                continue

            # track entire benchmark duration
            min_req_timestamp = min(min_req_timestamp, req_timestamp)
            max_res_timestamp = max(max_res_timestamp, res_timestamps[-1])

            # request latencies
            req_latency_ns = res_timestamps[-1] - req_timestamp
            request_latencies.append(req_latency_ns)  # nanosec
            req_latency_s = req_latency_ns / 1e9  # sec

            # time to first token
            ttft = res_timestamps[0] - req_timestamp
            time_to_first_tokens.append(ttft)

            # number of input tokens
            input_token_count = self._get_input_token_count(req_inputs)
            num_input_tokens.append(input_token_count)

            # output token throughput per request
            output_token_counts = self._get_output_token_counts(res_outputs)
            total_output_token = sum(output_token_counts)
            output_token_throughputs_per_request.append(
                total_output_token / req_latency_s
            )
            num_output_tokens.append(total_output_token)

            # inter token latencies
            if total_output_token > 1:
                inter_token_latency = (req_latency_ns - ttft) / (total_output_token - 1)
                inter_token_latencies.append(round(inter_token_latency))

            # The new ITL calculation above loses all token-level ITL information
            # and as a result breaks ITL vs token position visualization. Keep
            # the old version of inter token latency as a WAR to preserve the
            # visualization.
            chunked_inter_token_latency = []
            for (t1, _), (t2, n2) in self._pairwise(
                zip(res_timestamps, output_token_counts)
            ):
                # TMA-1676: handle empty first/last responses
                # if the latter response has zero token (e.g. empty string),
                # then set it default to one for the sake of inter token latency
                # calculation and to avoid divide by zero.
                num_token = 1 if n2 == 0 else n2
                chunked_inter_token_latency.append(round((t2 - t1) / num_token))
            chunked_inter_token_latencies.append(chunked_inter_token_latency)

        # request & output token throughput
        benchmark_duration = (max_res_timestamp - min_req_timestamp) / 1e9  # nanosec
        request_throughputs = [len(requests) / benchmark_duration]
        output_token_throughputs = [sum(num_output_tokens) / benchmark_duration]

        return LLMMetrics(
            request_throughputs,
            request_latencies,
            time_to_first_tokens,
            inter_token_latencies,
            output_token_throughputs,
            output_token_throughputs_per_request,
            num_output_tokens,
            num_input_tokens,
            chunked_inter_token_latencies,
        )

    def _pairwise(self, iterable):
        """Generate pairs of consecutive elements from the given iterable."""
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)

    def _preprocess_response(
        self, res_timestamps: List[int], res_outputs: List[Dict[str, str]]
    ) -> None:
        """Helper function to preprocess responses of a request."""
        if self._service_kind == "openai":
            # PA sometimes receives multiple SSE responses at once (as a single
            # response). Handle these responses by merging into a single response.
            for i in range(len(res_outputs)):
                response = res_outputs[i]["response"]
                responses = response.strip().split("\n\n")
                if len(responses) > 1:
                    merged_response = json.loads(remove_sse_prefix(responses[0]))
                    if (
                        merged_response["choices"][0]["delta"].get("content", None)
                        is None
                    ):
                        merged_response["choices"][0]["delta"]["content"] = ""
                    for r in responses[1:]:
                        text = self._extract_openai_text_output(r)
                        merged_response["choices"][0]["delta"]["content"] += text

                    res_outputs[i] = {"response": json.dumps(merged_response)}

            # Remove responses without any content
            indices_to_remove = []
            for idx, out in enumerate(res_outputs):
                if self._is_openai_empty_response(out["response"]):
                    indices_to_remove.append(idx)
            indices_to_remove.sort(reverse=True)
            for index in indices_to_remove:
                res_timestamps.pop(index)
                res_outputs.pop(index)

    def _get_input_token_count(self, req_inputs: dict) -> int:
        """Deserialize the request input and return tokenized inputs."""
        if self._service_kind == "triton":
            input_text = req_inputs["text_input"]
        elif self._service_kind == "openai":
            input_text = self._get_openai_input_text(req_inputs)
        else:
            raise ValueError(f"Unknown service kind: '{self._service_kind}'.")

        return len(self._tokenizer.encode(input_text))

    def _get_openai_input_text(self, req_inputs: dict) -> str:
        """Tokenize the OpenAI request input texts."""
        payload = json.loads(req_inputs["payload"])
        if self._response_format == ResponseFormat.OPENAI_CHAT_COMPLETIONS:
            return payload["messages"][0]["content"]
        elif self._response_format == ResponseFormat.OPENAI_COMPLETIONS:
            return payload["prompt"]
        else:
            raise ValueError(
                "Failed to parse OpenAI request input in profile export file."
            )

    def _get_output_token_counts(self, res_outputs: dict) -> List[int]:
        """Deserialize the response output and return tokenized outputs."""
        if self._service_kind == "triton":
            output_tokens = self._get_triton_output_tokens(res_outputs)
        elif self._service_kind == "openai":
            output_tokens = self._get_openai_output_tokens(res_outputs)
        else:
            raise ValueError(f"Unknown service kind: '{self._service_kind}'.")

        return list(map(len, output_tokens))

    def _get_triton_output_tokens(self, res_outputs: dict) -> List[List[int]]:
        """Return a list of Triton response output tokens."""
        output_texts = [r["text_output"] for r in res_outputs]
        return self._run_tokenizer(output_texts)

    def _get_openai_output_tokens(self, res_outputs: dict) -> List[List[int]]:
        """Return a list of OpenAI response output tokens."""
        output_texts = []
        for output in res_outputs:
            text = self._extract_openai_text_output(output["response"])
            output_texts.append(text)
        return self._run_tokenizer(output_texts)

    def _run_tokenizer(self, output_texts: List[str]) -> List[List[int]]:
        # exclamation mark trick forces the llama tokenization to consistently
        # start each output with a specific token which allows us to safely skip
        # the first token of every tokenized output and get only the ones that
        # are returned by the model
        output_texts = ["!" + txt for txt in output_texts]
        encodings = self._tokenizer(output_texts)
        return [out[1:] for out in encodings.data["input_ids"]]

    def _extract_openai_text_output(self, response: str) -> str:
        """Extracts text/content of the OpenAI response object."""
        response = remove_sse_prefix(response)

        if response == "[DONE]":
            return ""

        data = json.loads(response)
        completions = data["choices"][0]

        text_output = ""
        if "object" not in data:
            # FIXME: TPA-47 workaround for vLLM not following OpenAI Completions
            # API specification when streaming, missing 'object' field:
            # https://platform.openai.com/docs/api-reference/completions
            text_output = completions.get("text", "")
        elif data["object"] == "text_completion":  # legacy
            text_output = completions.get("text", "")
        elif data["object"] == "chat.completion":  # non-streaming
            text_output = completions["message"].get("content", "")
        elif data["object"] == "chat.completion.chunk":  # streaming
            text_output = completions["delta"].get("content", "")
        else:
            obj_type = data["object"]
            raise ValueError(f"Unknown OpenAI response object type '{obj_type}'.")
        return text_output

    def _is_openai_empty_response(self, response: str) -> bool:
        """Returns true if the response is an openai response with no content (or empty content)"""
        text = self._extract_openai_text_output(response)
        if text:
            return False
        return True
