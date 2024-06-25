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

from typing import List

from genai_perf.metrics.metrics import Metrics


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
        output_sequence_lengths: List[int] = [],
        input_sequence_lengths: List[int] = [],
        chunked_inter_token_latencies: List[List[int]] = [[]],
    ) -> None:
        super().__init__(request_throughputs, request_latencies)
        self.time_to_first_tokens = time_to_first_tokens
        self.inter_token_latencies = inter_token_latencies
        self.output_token_throughputs = output_token_throughputs
        self.output_token_throughputs_per_request = output_token_throughputs_per_request
        self.output_sequence_lengths = output_sequence_lengths
        self.input_sequence_lengths = input_sequence_lengths

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
        self._base_names["output_sequence_lengths"] = "output_sequence_length"
        self._base_names["input_sequence_lengths"] = "input_sequence_length"
