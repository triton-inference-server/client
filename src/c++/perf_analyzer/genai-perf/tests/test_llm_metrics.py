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

import pytest
from genai_perf.metrics import LLMMetrics


class TestLLMMetrics:

    def test_llm_metric_request_metrics(self) -> None:
        """Test request_metrics property."""
        m = LLMMetrics(
            request_throughputs=[10.12, 11.33],
            request_latencies=[3, 44],
            time_to_first_tokens=[1, 2, 3],
            inter_token_latencies=[4, 5],
            output_token_throughputs=[22.13, 9423.02],
            output_token_throughputs_per_request=[7, 8, 9],
            output_sequence_lengths=[3, 4],
            input_sequence_lengths=[12, 34],
        )
        req_metrics = m.request_metrics
        assert len(req_metrics) == 6
        assert req_metrics[0].name == "time_to_first_token"
        assert req_metrics[0].unit == "ms"
        assert req_metrics[1].name == "inter_token_latency"
        assert req_metrics[1].unit == "ms"
        assert req_metrics[2].name == "request_latency"
        assert req_metrics[2].unit == "ms"
        assert req_metrics[3].name == "output_token_throughput_per_request"
        assert req_metrics[3].unit == "tokens/sec"
        assert req_metrics[4].name == "output_sequence_length"
        assert req_metrics[4].unit == "tokens"
        assert req_metrics[5].name == "input_sequence_length"
        assert req_metrics[5].unit == "tokens"

    def test_llm_metric_system_metrics(self) -> None:
        """Test system_metrics property."""
        m = LLMMetrics(
            request_throughputs=[10.12, 11.33],
            request_latencies=[3, 44],
            time_to_first_tokens=[1, 2, 3],
            inter_token_latencies=[4, 5],
            output_token_throughputs=[22.13, 9423.02],
            output_token_throughputs_per_request=[7, 8, 9],
            output_sequence_lengths=[3, 4],
            input_sequence_lengths=[12, 34],
        )

        sys_metrics = m.system_metrics
        assert len(sys_metrics) == 2
        assert sys_metrics[0].name == "output_token_throughput"
        assert sys_metrics[0].unit == "per sec"
        assert sys_metrics[1].name == "request_throughput"
        assert sys_metrics[1].unit == "per sec"

    def test_llm_metrics_get_base_name(self) -> None:
        """Test get_base_name method in LLMMetrics class."""
        # initialize with dummy values
        metrics = LLMMetrics(
            request_throughputs=[10.12, 11.33],
            request_latencies=[3, 44],
            time_to_first_tokens=[1, 2, 3],
            inter_token_latencies=[4, 5],
            output_token_throughputs=[22.13, 9423.02],
            output_token_throughputs_per_request=[7, 8, 9],
            output_sequence_lengths=[3, 4],
            input_sequence_lengths=[12, 34],
        )
        assert metrics.get_base_name("time_to_first_tokens") == "time_to_first_token"
        assert metrics.get_base_name("inter_token_latencies") == "inter_token_latency"
        assert (
            metrics.get_base_name("output_token_throughputs_per_request")
            == "output_token_throughput_per_request"
        )
        assert (
            metrics.get_base_name("output_sequence_lengths") == "output_sequence_length"
        )
        assert (
            metrics.get_base_name("input_sequence_lengths") == "input_sequence_length"
        )
        with pytest.raises(KeyError):
            metrics.get_base_name("hello1234")
