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

from pathlib import Path
from typing import List, Tuple

from genai_perf.metrics import Metrics, ResponseFormat, Statistics
from genai_perf.utils import load_json


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
            elif data["endpoint"] == "v1/embeddings":
                self._response_format = ResponseFormat.OPENAI_EMBEDDINGS
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
                elif "embedding" in response:
                    self._response_format = ResponseFormat.OPENAI_EMBEDDINGS
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

    def _parse_requests(self, requests: dict) -> Metrics:
        """Parse each request in profile data to extract core metrics."""
        min_req_timestamp, max_res_timestamp = float("inf"), 0
        request_latencies = []

        for request in requests:
            req_timestamp = request["timestamp"]
            res_timestamps = request["response_timestamps"]

            # track entire benchmark duration
            min_req_timestamp = min(min_req_timestamp, req_timestamp)
            max_res_timestamp = max(max_res_timestamp, res_timestamps[-1])

            # request latencies
            req_latency = res_timestamps[-1] - req_timestamp
            request_latencies.append(req_latency)

        # request throughput
        benchmark_duration = (max_res_timestamp - min_req_timestamp) / 1e9  # to seconds
        request_throughputs = [len(requests) / benchmark_duration]

        return Metrics(
            request_throughputs,
            request_latencies,
        )

    def get_statistics(self, infer_mode: str, load_level: str) -> Statistics:
        """Return profile statistics if it exists."""
        if (infer_mode, load_level) not in self._profile_results:
            raise KeyError(f"Profile with {infer_mode}={load_level} does not exist.")
        return self._profile_results[(infer_mode, load_level)]

    def get_profile_load_info(self) -> List[Tuple[str, str]]:
        """Return available (infer_mode, load_level) tuple keys."""
        return [k for k, _ in self._profile_results.items()]
