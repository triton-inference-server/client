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

from enum import Enum, auto
from typing import List


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
        "output_sequence_length",
        "input_sequence_length",
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
