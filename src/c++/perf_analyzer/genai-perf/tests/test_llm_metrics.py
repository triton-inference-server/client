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

import json
from io import StringIO
from pathlib import Path

import numpy as np
import pytest
from genai_perf.llm_metrics import LLMMetrics, LLMProfileDataParser
from genai_perf.utils import remove_file
from transformers import AutoTokenizer


def ns_to_sec(ns: int) -> int | float:
    """Convert from nanosecond to second."""
    return ns / 1e9


class TestLLMProfileDataParser:
    openai_profile_data = {
        "experiments": [
            {
                "experiment": {
                    "mode": "concurrency",
                    "value": 10,
                },
                "requests": [
                    {
                        "timestamp": 1,
                        # last two empty/null responses will be ignored
                        "response_timestamps": [3, 5, 8, 12, 13, 14],
                        "response_outputs": [
                            {
                                "response": 'data: {"id":"abc","object":"chat.completion.chunk","created":123,"model":"llama-2-7b","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}\n\n'
                            },
                            {
                                "response": 'data: {"id":"abc","object":"chat.completion.chunk","created":123,"model":"llama-2-7b","choices":[{"index":0,"delta":{"content":"dogs"},"finish_reason":null}]}\n\n'
                            },
                            {
                                "response": 'data: {"id":"abc","object":"chat.completion.chunk","created":123,"model":"llama-2-7b","choices":[{"index":0,"delta":{"content":"are"},"finish_reason":null}]}\n\n'
                            },
                            {
                                "response": 'data: {"id":"abc","object":"chat.completion.chunk","created":123,"model":"llama-2-7b","choices":[{"index":0,"delta":{"content":"cool"},"finish_reason":null}]}\n\n'
                            },
                            {
                                "response": 'data: {"id":"abc","object":"chat.completion.chunk","created":123,"model":"llama-2-7b","choices":[{"index":0,"delta":{},"finish_reason":null}]}\n\n'
                            },
                            {"response": "data: [DONE]\n\n"},
                        ],
                    },
                    {
                        "timestamp": 2,
                        # last two empty/null responses will be ignored
                        "response_timestamps": [4, 7, 11, 15, 18, 19],
                        "response_outputs": [
                            {
                                "response": 'data: {"id":"abc","object":"chat.completion.chunk","created":123,"model":"llama-2-7b","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}\n\n'
                            },
                            {
                                "response": 'data: {"id":"abc","object":"chat.completion.chunk","created":123,"model":"llama-2-7b","choices":[{"index":0,"delta":{"content":"I"},"finish_reason":null}]}\n\n'
                            },
                            {
                                "response": 'data: {"id":"abc","object":"chat.completion.chunk","created":123,"model":"llama-2-7b","choices":[{"index":0,"delta":{"content":"don\'t"},"finish_reason":null}]}\n\n'
                            },
                            {
                                "response": 'data: {"id":"abc","object":"chat.completion.chunk","created":123,"model":"llama-2-7b","choices":[{"index":0,"delta":{"content":"cook food"},"finish_reason":null}]}\n\n'
                            },
                            {
                                "response": 'data: {"id":"abc","object":"chat.completion.chunk","created":123,"model":"llama-2-7b","choices":[{"index":0,"delta":{},"finish_reason":null}]}\n\n'
                            },
                            {"response": "data: [DONE]\n\n"},
                        ],
                    },
                ],
            },
        ],
    }

    triton_profile_data = {
        "experiments": [
            {
                "experiment": {
                    "mode": "concurrency",
                    "value": 10,
                },
                "requests": [
                    {
                        "timestamp": 1,
                        "response_timestamps": [3, 5, 8],
                        # FIXME - remove the whitespace once PA handles it.
                        # LLMProfileDataParser preprocessse the responses
                        # from triton server and removes first few chars.
                        # Add whitespace to avoid valid chars being removed.
                        "response_outputs": [
                            {"text_output": "   dogs"},
                            {"text_output": "   are"},
                            {"text_output": "   cool"},
                        ],
                    },
                    {
                        "timestamp": 2,
                        "response_timestamps": [4, 7, 11],
                        "response_outputs": [
                            {"text_output": "   I"},
                            {"text_output": "   don't"},
                            {"text_output": "   cook food"},
                        ],
                    },
                ],
            },
            {
                "experiment": {
                    "mode": "request_rate",
                    "value": 2.0,
                },
                "requests": [
                    {
                        "timestamp": 5,
                        "response_timestamps": [7, 8, 13, 18],
                        "response_outputs": [
                            {"text_output": "   cats"},
                            {"text_output": "   are"},
                            {"text_output": "   cool"},
                            {"text_output": "   too"},
                        ],
                    },
                    {
                        "timestamp": 3,
                        "response_timestamps": [6, 8, 11],
                        "response_outputs": [
                            {"text_output": "   it's"},
                            {"text_output": "   very"},
                            {"text_output": "   simple work"},
                        ],
                    },
                ],
            },
        ],
    }

    @pytest.fixture
    def mock_read_write(self, monkeypatch) -> None:
        written_data = []

        original_open = open

        def custom_open(filename, *args, **kwargs):
            def write(content):
                written_data.append(content)

            if filename == "triton_profile_export.json":
                tmp_file = StringIO(json.dumps(self.triton_profile_data))
                return tmp_file
            elif filename == "openai_profile_export.json":
                tmp_file = StringIO(json.dumps(self.openai_profile_data))
                return tmp_file
            elif filename == "profile_export.csv":
                tmp_file = StringIO()
                tmp_file.write = write
                return tmp_file
            else:
                return original_open(filename, *args, **kwargs)

        monkeypatch.setattr("builtins.open", custom_open)

        return written_data

    def test_csv_output(self, mock_read_write) -> None:
        """
        Collect LLM metrics from profile export data and confirm correct values are printed in csv
        """

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        pd = LLMProfileDataParser(
            filename="triton_profile_export.json",
            service_kind="triton",
            tokenizer=tokenizer,
        )
        stat = pd.get_statistics(infer_mode="concurrency", load_level="10")

        expected_content = [
            "Metric,avg,min,max,p99,p95,p90,p75,p50,p25\r\n",
            "Time To First Token (ns),2,2,2,2,2,2,2,2,2\r\n",
            "Inter Token Latency (ns),2,2,3,3,3,3,2,2,2\r\n",
            "Request Latency (ns),8,7,9,9,9,9,8,8,8\r\n",
            "Num Output Token,4,3,5,5,5,5,4,4,4\r\n",
            "\r\n",
            "Metric,Value\r\n",
            "Output Token Throughput (per sec),800000000.00\r\n",
            "Request Throughput (per sec),200000000.00\r\n",
        ]

        stat.export_to_csv("profile_export.csv")

        returned_data = mock_read_write

        assert returned_data == expected_content

    def test_triton_llm_profile_data(self, mock_read_write) -> None:
        """Collect LLM metrics from profile export data and check values.

        Metrics
        * time to first tokens
            - experiment 1: [3 - 1, 4 - 2] = [2, 2]
            - experiment 2: [7 - 5, 6 - 3] = [2, 3]
        * inter token latencies
            - experiment 1: [(5 - 3)/1, (8 - 5)/1, (7 - 4)/2, (11 - 7)/2]
                          : [2, 3, 3/2, 2]
                          : [2, 3, 2, 2]
            - experiment 2: [(8 - 7)/1, (13 - 8)/1, (18 - 13)/1, (8 - 6)/1, (11 - 8)/2]
                          : [1, 5, 5, 2, 3/2]
                          : [1, 5, 5, 2, 2]
        * output token throughputs per request
            - experiment 1: [3/(8 - 1), 5/(11 - 2)] = [3/7, 5/9]
            - experiment 2: [4/(18 - 5), 5/(11 - 3)] = [4/13, 5/8]
        * output token throughputs
            - experiment 1: [(3 + 5)/(11 - 1)] = [8/10]
            - experiment 2: [(4 + 5)/(18 - 3)] = [3/5]
        * num output tokens
            - experiment 1: [3, 5]
            - experiment 2: [4, 5]
        """
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        pd = LLMProfileDataParser(
            filename="triton_profile_export.json",
            service_kind="triton",
            tokenizer=tokenizer,
        )

        # experiment 1 statistics
        stat = pd.get_statistics(infer_mode="concurrency", load_level="10")

        assert stat.avg_time_to_first_token == 2
        assert stat.avg_inter_token_latency == 2.25
        avg_ottpr = 31 / ns_to_sec(63)
        assert stat.avg_output_token_throughput_per_request == pytest.approx(avg_ottpr)
        assert stat.avg_num_output_token == 4

        assert stat.p50_time_to_first_token == 2
        assert stat.p50_inter_token_latency == 2
        p50_ottpr = 31 / ns_to_sec(63)
        assert stat.p50_output_token_throughput_per_request == pytest.approx(p50_ottpr)
        assert stat.p50_num_output_token == 4

        assert stat.min_time_to_first_token == 2
        assert stat.min_inter_token_latency == 2
        min_ottpr = 3 / ns_to_sec(7)
        assert stat.min_output_token_throughput_per_request == pytest.approx(min_ottpr)
        assert stat.min_num_output_token == 3

        assert stat.max_time_to_first_token == 2
        assert stat.max_inter_token_latency == 3
        max_ottpr = 5 / ns_to_sec(9)
        assert stat.max_output_token_throughput_per_request == pytest.approx(max_ottpr)
        assert stat.max_num_output_token == 5

        assert stat.std_time_to_first_token == np.std([2, 2])
        assert stat.std_inter_token_latency == np.std([2, 3, 2, 2])
        ottpr1 = 3 / ns_to_sec(7)
        ottpr2 = 5 / ns_to_sec(9)
        assert stat.std_output_token_throughput_per_request == pytest.approx(
            np.std([ottpr1, ottpr2])
        )
        assert stat.std_num_output_token == np.std([3, 5])

        oott = 8 / ns_to_sec(10)
        assert stat.avg_output_token_throughput == pytest.approx(oott)

        # experiment 2 statistics
        stat = pd.get_statistics(infer_mode="request_rate", load_level="2.0")

        assert stat.avg_time_to_first_token == 2.5
        assert stat.avg_inter_token_latency == 3
        avg_ottpr = 97 / ns_to_sec(208)
        assert stat.avg_output_token_throughput_per_request == pytest.approx(avg_ottpr)
        assert stat.avg_num_output_token == 4.5

        assert stat.p50_time_to_first_token == 2.5
        assert stat.p50_inter_token_latency == 2
        p50_ottpr = 97 / ns_to_sec(208)
        assert stat.p50_output_token_throughput_per_request == pytest.approx(p50_ottpr)
        assert stat.p50_num_output_token == 4.5

        assert stat.min_time_to_first_token == 2
        assert stat.min_inter_token_latency == 1
        min_ottpr = 4 / ns_to_sec(13)
        assert stat.min_output_token_throughput_per_request == pytest.approx(min_ottpr)
        assert stat.min_num_output_token == 4

        assert stat.max_time_to_first_token == 3
        assert stat.max_inter_token_latency == 5
        max_ottpr = 5 / ns_to_sec(8)
        assert stat.max_output_token_throughput_per_request == pytest.approx(max_ottpr)
        assert stat.max_num_output_token == 5

        assert stat.std_time_to_first_token == np.std([2, 3])
        assert stat.std_inter_token_latency == np.std([1, 5, 5, 2, 2])
        ottpr1 = 4 / ns_to_sec(13)
        ottpr2 = 5 / ns_to_sec(8)
        assert stat.std_output_token_throughput_per_request == pytest.approx(
            np.std([ottpr1, ottpr2])
        )
        assert stat.std_num_output_token == np.std([4, 5])

        oott = 6 / ns_to_sec(10)
        assert stat.avg_output_token_throughput == pytest.approx(oott)

        # check non-existing profile data
        with pytest.raises(KeyError):
            pd.get_statistics(infer_mode="concurrency", load_level="30")

    def test_openai_llm_profile_data(self, mock_read_write) -> None:
        """Collect LLM metrics from profile export data and check values.

        Metrics
        * time to first tokens
            - experiment 1: [3 - 1, 4 - 2] = [2, 2]
        * inter token latencies
            - experiment 1: [(5 - 3)/1, (8 - 5)/1, (12 - 8)/1, (7 - 4)/1, (11 - 7)/2, (15 - 11)/2]
                          : [2, 3, 4, 3, 2, 2]
        * output token throughputs per request
            - experiment 1: [3/(12 - 1), 5/(15 - 2)] = [3/11, 5/13]
        * output token throughputs
            - experiment 1: [(3 + 5)/(15 - 1)] = [4/7]
        * num output tokens
            - experiment 1: [3, 5]
        """
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        pd = LLMProfileDataParser(
            filename="openai_profile_export.json",
            service_kind="openai",
            tokenizer=tokenizer,
        )

        # experiment 1 statistics
        stat = pd.get_statistics(infer_mode="concurrency", load_level="10")

        assert stat.avg_time_to_first_token == 2
        assert stat.avg_inter_token_latency == 8 / 3
        avg_ottpr = 47 / ns_to_sec(143)
        assert stat.avg_output_token_throughput_per_request == pytest.approx(avg_ottpr)
        assert stat.avg_num_output_token == 4

        assert stat.p50_time_to_first_token == 2
        assert stat.p50_inter_token_latency == 2.5
        p50_ottpr = 47 / ns_to_sec(143)
        assert stat.p50_output_token_throughput_per_request == pytest.approx(p50_ottpr)
        assert stat.p50_num_output_token == 4

        assert stat.min_time_to_first_token == 2
        assert stat.min_inter_token_latency == 2
        min_ottpr = 3 / ns_to_sec(11)
        assert stat.min_output_token_throughput_per_request == pytest.approx(min_ottpr)
        assert stat.min_num_output_token == 3

        assert stat.max_time_to_first_token == 2
        assert stat.max_inter_token_latency == 4
        max_ottpr = 5 / ns_to_sec(13)
        assert stat.max_output_token_throughput_per_request == pytest.approx(max_ottpr)
        assert stat.max_num_output_token == 5

        assert stat.std_time_to_first_token == np.std([2, 2])
        assert stat.std_inter_token_latency == np.std([2, 3, 4, 3, 2, 2])
        ottpr1 = 3 / ns_to_sec(11)
        ottpr2 = 5 / ns_to_sec(13)
        assert stat.std_output_token_throughput_per_request == pytest.approx(
            np.std([ottpr1, ottpr2])
        )
        assert stat.std_num_output_token == np.std([3, 5])

        oott = 4 / ns_to_sec(7)
        assert stat.avg_output_token_throughput == pytest.approx(oott)

        # check non-existing profile data
        with pytest.raises(KeyError):
            pd.get_statistics(infer_mode="concurrency", load_level="40")

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
            num_output_tokens=[3, 4],
        )
        assert metrics.get_base_name("time_to_first_tokens") == "time_to_first_token"
        assert metrics.get_base_name("inter_token_latencies") == "inter_token_latency"
        assert (
            metrics.get_base_name("output_token_throughputs_per_request")
            == "output_token_throughput_per_request"
        )
        assert metrics.get_base_name("num_output_tokens") == "num_output_token"
        with pytest.raises(KeyError):
            metrics.get_base_name("hello1234")
