#!/usr/bin/env python3

# Copyright 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met: * Redistributions of source code must retain the above copyright
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

import json
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from transformers import AutoTokenizer


@dataclass
class LLMMetrics:
    time_to_first_tokens: list[int]
    inter_token_latencies: list[int]
    output_token_throughputs: list[int]

    def get_base_name(self, attr_name: str) -> str:
        if attr_name == "time_to_first_tokens":
            return "time_to_first_token"
        elif attr_name == "inter_token_latencies":
            return "inter_token_latency"
        elif attr_name == "output_token_throughputs":
            return "output_token_throughput"
        else:
            raise ValueError(f"No attribute named '{attr_name}' exists.")


class Statistics:
    # TODO: make this parameter LLM agnostic
    def __init__(self, metrics: LLMMetrics):
        # iterate through LLMMetrics to calculate statistics and set attributes
        for attr, data in metrics.__dict__.items():
            attr = metrics.get_base_name(attr)
            self._calculate_mean(data, attr)
            self._calculate_percentiles(data, attr)
            self._calculate_minmax(data, attr)
            self._calculate_std(data, attr)

    def _calculate_mean(self, data: list[int], attr: str):
        avg = np.mean(data)
        setattr(self, "avg_" + attr, avg)

    def _calculate_percentiles(self, data: list[int], attr: str):
        p50, p90, p95, p99 = np.percentile(data, [50, 90, 95, 99])
        setattr(self, "p50_" + attr, p50)
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
        attr_str = ""
        for k, v in self.__dict__.items():
            attr_str += f"{k}={v},"
        return f"Statistics({attr_str})"


class LLMProfileData:
    def __init__(self, filename: str, tokenizer_model: str = "gpt2") -> None:
        # load profile export data
        with open(filename) as f:
            data = json.load(f)

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)

        self._profile_results = {}
        for experiment in data["experiments"]:
            infer_mode = experiment["experiment"]["mode"]
            level = experiment["experiment"]["value"]
            requests = experiment["requests"]

            metrics = self._collect_llm_metrics(requests, tokenizer)

            # aggregate and calculate statistics
            statistics = Statistics(metrics)
            self._profile_results[(infer_mode, level)] = statistics

    # TODO: handle single response case
    def _collect_llm_metrics(
        self, requests: dict, tokenizer: AutoTokenizer
    ) -> LLMMetrics:
        time_to_first_tokens = []
        inter_token_latencies = []
        output_token_throughputs = []
        for request in requests:
            req_timestamp = request["timestamp"]
            res_timestamps = request["response_timestamps"]
            res_outputs = request["response_outputs"]

            # time to first token
            time_to_first_tokens.append(res_timestamps[0] - req_timestamp)

            # output token throughput
            output_tokens = tokenizer(res_outputs)["input_ids"]
            num_output_tokens = list(map(lambda x: len(x), output_tokens))
            total_output_tokens = np.sum(num_output_tokens)
            output_latency = res_timestamps[-1] - res_timestamps[0]
            output_token_throughputs.append(total_output_tokens / output_latency)

            # inter token latency
            for t1, t2 in pairwise(res_timestamps):
                inter_token_latencies.append(t2 - t1)

        return LLMMetrics(
            time_to_first_tokens,
            inter_token_latencies,
            output_token_throughputs,
        )

    def get_statistics(self, infer_mode: str, level: int | float) -> Statistics:
        if (infer_mode, level) not in self._profile_results:
            raise KeyError(f"Profile with {infer_mode}={level} does not exist.")
        return self._profile_results[(infer_mode, level)]
