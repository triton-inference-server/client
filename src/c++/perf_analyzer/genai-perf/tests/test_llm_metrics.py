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
from typing import Any, List, Union

import numpy as np
import pytest
from genai_perf.llm_metrics import LLMMetrics, LLMProfileDataParser, ResponseFormat
from genai_perf.tokenizer import DEFAULT_TOKENIZER, get_tokenizer


def ns_to_sec(ns: int) -> Union[int, float]:
    """Convert from nanosecond to second."""
    return ns / 1e9


class TestLLMProfileDataParser:
    @pytest.fixture
    def mock_read_write(self, monkeypatch: pytest.MonkeyPatch) -> List[str]:
        """
        This function will mock the open function for specific files:

        - For "triton_profile_export.json", it will read and return the
          contents of self.triton_profile_data
        - For "openai_profile_export.json", it will read and return the
          contents of self.openai_profile_data
        - For "profile_export.csv", it will capture all data written to
          the file, and return it as the return value of this function
        - For all other files, it will behave like the normal open function
        """

        written_data = []

        original_open = open

        def custom_open(filename, *args, **kwargs):
            def write(self: Any, content: str) -> int:
                written_data.append(content)
                return len(content)

            if filename == "triton_profile_export.json":
                tmp_file = StringIO(json.dumps(self.triton_profile_data))
                return tmp_file
            elif filename == "openai_profile_export.json":
                tmp_file = StringIO(json.dumps(self.openai_profile_data))
                return tmp_file
            elif filename == "empty_profile_export.json":
                tmp_file = StringIO(json.dumps(self.empty_profile_data))
                return tmp_file
            elif filename == "profile_export.csv":
                tmp_file = StringIO()
                tmp_file.write = write.__get__(tmp_file)
                return tmp_file
            else:
                return original_open(filename, *args, **kwargs)

        monkeypatch.setattr("builtins.open", custom_open)

        return written_data

    def test_triton_llm_profile_data(self, mock_read_write: pytest.MonkeyPatch) -> None:
        """Collect LLM metrics from profile export data and check values.

        Metrics
        * time to first tokens
            - experiment 1: [3 - 1, 4 - 2] = [2, 2]
            - experiment 2: [7 - 5, 6 - 3] = [2, 3]
        * inter token latencies
            - experiment 1: [((8 - 1) - 2)/(3 - 1), ((11 - 2) - 2)/(6 - 1)]
                          : [2.5, 1.4]
                          : [2, 1]  # rounded
            - experiment 2: [((18 - 5) - 2)/(4 - 1), ((11 - 3) - 3)/(6 - 1)]
                          : [11/3, 1]
                          : [4, 1]  # rounded
        * output token throughputs per request
            - experiment 1: [3/(8 - 1), 6/(11 - 2)] = [3/7, 6/9]
            - experiment 2: [4/(18 - 5), 6/(11 - 3)] = [4/13, 6/8]
        * output token throughputs
            - experiment 1: [(3 + 6)/(11 - 1)] = [9/10]
            - experiment 2: [(4 + 6)/(18 - 3)] = [2/3]
        * num output tokens
            - experiment 1: [3, 6]
            - experiment 2: [4, 6]
        * num input tokens
            - experiment 1: [3, 4]
            - experiment 2: [3, 4]
        """
        tokenizer = get_tokenizer(DEFAULT_TOKENIZER)
        pd = LLMProfileDataParser(
            filename=Path("triton_profile_export.json"),
            tokenizer=tokenizer,
        )

        # experiment 1 metrics & statistics
        stat = pd.get_statistics(infer_mode="concurrency", load_level="10")
        metrics = stat.metrics

        assert isinstance(metrics, LLMMetrics)

        assert metrics.time_to_first_tokens == [2, 2]
        assert metrics.inter_token_latencies == [2, 1]
        ottpr = [3 / ns_to_sec(7), 6 / ns_to_sec(9)]
        assert metrics.output_token_throughputs_per_request == pytest.approx(ottpr)
        ott = [9 / ns_to_sec(10)]
        assert metrics.output_token_throughputs == pytest.approx(ott)
        assert metrics.num_output_tokens == [3, 6]
        assert metrics.num_input_tokens == [3, 4]

        # Disable Pylance warnings for dynamically set attributes due to Statistics
        # not having strict attributes listed.
        assert stat.avg_time_to_first_token == 2e-6  # type: ignore
        assert stat.avg_inter_token_latency == 1.5e-6  # type: ignore
        assert stat.avg_output_token_throughput_per_request == pytest.approx(  # type: ignore
            np.mean(ottpr)
        )
        assert stat.avg_num_output_token == 4.5  # type: ignore
        assert stat.avg_num_input_token == 3.5  # type: ignore

        assert stat.p50_time_to_first_token == 2e-6  # type: ignore
        assert stat.p50_inter_token_latency == 1.5e-6  # type: ignore
        assert stat.p50_output_token_throughput_per_request == pytest.approx(  # type: ignore
            np.percentile(ottpr, 50)
        )
        assert stat.p50_num_output_token == 4.5  # type: ignore
        assert stat.p50_num_input_token == 3.5  # type: ignore

        assert stat.min_time_to_first_token == 2e-6  # type: ignore
        assert stat.min_inter_token_latency == 1e-6  # type: ignore
        min_ottpr = 3 / ns_to_sec(7)
        assert stat.min_output_token_throughput_per_request == pytest.approx(min_ottpr)  # type: ignore
        assert stat.min_num_output_token == 3  # type: ignore
        assert stat.min_num_input_token == 3  # type: ignore

        assert stat.max_time_to_first_token == 2e-6  # type: ignore
        assert stat.max_inter_token_latency == 2e-6  # type: ignore
        max_ottpr = 6 / ns_to_sec(9)
        assert stat.max_output_token_throughput_per_request == pytest.approx(max_ottpr)  # type: ignore
        assert stat.max_num_output_token == 6  # type: ignore
        assert stat.max_num_input_token == 4  # type: ignore

        assert stat.std_time_to_first_token == np.std([2e-6, 2e-6])  # type: ignore
        assert stat.std_inter_token_latency == np.std([2e-6, 1e-6])  # type: ignore
        assert stat.std_output_token_throughput_per_request == pytest.approx(  # type: ignore
            np.std(ottpr)
        )
        assert stat.std_num_output_token == np.std([3, 6])  # type: ignore
        assert stat.std_num_input_token == np.std([3, 4])  # type: ignore

        oott = 9 / ns_to_sec(10)
        assert stat.avg_output_token_throughput == pytest.approx(oott)  # type: ignore

        # experiment 2 statistics
        stat = pd.get_statistics(infer_mode="request_rate", load_level="2.0")
        metrics = stat.metrics
        assert isinstance(metrics, LLMMetrics)

        assert metrics.time_to_first_tokens == [2, 3]
        assert metrics.inter_token_latencies == [4, 1]
        ottpr = [4 / ns_to_sec(13), 6 / ns_to_sec(8)]
        assert metrics.output_token_throughputs_per_request == pytest.approx(ottpr)
        ott = [2 / ns_to_sec(3)]
        assert metrics.output_token_throughputs == pytest.approx(ott)
        assert metrics.num_output_tokens == [4, 6]
        assert metrics.num_input_tokens == [3, 4]

        assert stat.avg_time_to_first_token == pytest.approx(2.5e-6)  # type: ignore
        assert stat.avg_inter_token_latency == pytest.approx(2.5e-6)  # type: ignore
        assert stat.avg_output_token_throughput_per_request == pytest.approx(  # type: ignore
            np.mean(ottpr)
        )
        assert stat.avg_num_output_token == 5  # type: ignore
        assert stat.avg_num_input_token == 3.5  # type: ignore

        assert stat.p50_time_to_first_token == pytest.approx(2.5e-6)  # type: ignore
        assert stat.p50_inter_token_latency == pytest.approx(2.5e-6)  # type: ignore
        assert stat.p50_output_token_throughput_per_request == pytest.approx(  # type: ignore
            np.percentile(ottpr, 50)
        )
        assert stat.p50_num_output_token == 5  # type: ignore
        assert stat.p50_num_input_token == 3.5  # type: ignore

        assert stat.min_time_to_first_token == pytest.approx(2e-6)  # type: ignore
        assert stat.min_inter_token_latency == pytest.approx(1e-6)  # type: ignore
        min_ottpr = 4 / ns_to_sec(13)
        assert stat.min_output_token_throughput_per_request == pytest.approx(min_ottpr)  # type: ignore
        assert stat.min_num_output_token == 4  # type: ignore
        assert stat.min_num_input_token == 3  # type: ignore

        assert stat.max_time_to_first_token == pytest.approx(3e-6)  # type: ignore
        assert stat.max_inter_token_latency == pytest.approx(4e-6)  # type: ignore
        max_ottpr = 6 / ns_to_sec(8)
        assert stat.max_output_token_throughput_per_request == pytest.approx(max_ottpr)  # type: ignore
        assert stat.max_num_output_token == 6  # type: ignore
        assert stat.max_num_input_token == 4  # type: ignore

        assert stat.std_time_to_first_token == np.std([2, 3]) * (1e-6)  # type: ignore
        assert stat.std_inter_token_latency == np.std([4, 1]) * (1e-6)  # type: ignore
        assert stat.std_output_token_throughput_per_request == pytest.approx(  # type: ignore
            np.std(ottpr)
        )
        assert stat.std_num_output_token == np.std([4, 6])  # type: ignore
        assert stat.std_num_input_token == np.std([3, 4])  # type: ignore

        oott = 2 / ns_to_sec(3)
        assert stat.avg_output_token_throughput == pytest.approx(oott)  # type: ignore

        # check non-existing profile data
        with pytest.raises(KeyError):
            pd.get_statistics(infer_mode="concurrency", load_level="30")

    def test_openai_llm_profile_data(self, mock_read_write: pytest.MonkeyPatch) -> None:
        """Collect LLM metrics from profile export data and check values.

        Metrics
        * time to first tokens
            - experiment 1: [5 - 1, 7 - 2] = [4, 5]
        * inter token latencies
            - experiment 1: [((12 - 1) - 4)/(3 - 1), ((15 - 2) - 5)/(6 - 1)]
                          : [3.5, 1.6]
                          : [4, 2]  # rounded
        * output token throughputs per request
            - experiment 1: [3/(12 - 1), 6/(15 - 2)] = [3/11, 6/13]
        * output token throughputs
            - experiment 1: [(3 + 6)/(15 - 1)] = [9/14]
        * num output tokens
            - experiment 1: [3, 6]
        * num input tokens
            - experiment 1: [3, 4]
        """
        tokenizer = get_tokenizer(DEFAULT_TOKENIZER)
        pd = LLMProfileDataParser(
            filename=Path("openai_profile_export.json"),
            tokenizer=tokenizer,
        )

        # experiment 1 statistics
        stat = pd.get_statistics(infer_mode="concurrency", load_level="10")
        metrics = stat.metrics
        assert isinstance(metrics, LLMMetrics)

        assert metrics.time_to_first_tokens == [4, 5]
        assert metrics.inter_token_latencies == [4, 2]
        ottpr = [3 / ns_to_sec(11), 6 / ns_to_sec(13)]
        assert metrics.output_token_throughputs_per_request == pytest.approx(ottpr)
        ott = [9 / ns_to_sec(14)]
        assert metrics.output_token_throughputs == pytest.approx(ott)
        assert metrics.num_output_tokens == [3, 6]
        assert metrics.num_input_tokens == [3, 4]

        assert stat.avg_time_to_first_token == pytest.approx(4.5e-6)  # type: ignore
        assert stat.avg_inter_token_latency == pytest.approx(3e-6)  # type: ignore
        assert stat.avg_output_token_throughput_per_request == pytest.approx(  # type: ignore
            np.mean(ottpr)
        )
        assert stat.avg_num_output_token == 4.5  # type: ignore
        assert stat.avg_num_input_token == 3.5  # type: ignore

        assert stat.p50_time_to_first_token == pytest.approx(4.5e-6)  # type: ignore
        assert stat.p50_inter_token_latency == pytest.approx(3e-6)  # type: ignore
        assert stat.p50_output_token_throughput_per_request == pytest.approx(  # type: ignore
            np.percentile(ottpr, 50)
        )
        assert stat.p50_num_output_token == 4.5  # type: ignore
        assert stat.p50_num_input_token == 3.5  # type: ignore

        assert stat.min_time_to_first_token == pytest.approx(4e-6)  # type: ignore
        assert stat.min_inter_token_latency == pytest.approx(2e-6)  # type: ignore
        min_ottpr = 3 / ns_to_sec(11)
        assert stat.min_output_token_throughput_per_request == pytest.approx(min_ottpr)  # type: ignore
        assert stat.min_num_output_token == 3  # type: ignore
        assert stat.min_num_input_token == 3  # type: ignore

        assert stat.max_time_to_first_token == pytest.approx(5e-6)  # type: ignore
        assert stat.max_inter_token_latency == pytest.approx(4e-6)  # type: ignore
        max_ottpr = 6 / ns_to_sec(13)
        assert stat.max_output_token_throughput_per_request == pytest.approx(max_ottpr)  # type: ignore
        assert stat.max_num_output_token == 6  # type: ignore
        assert stat.max_num_input_token == 4  # type: ignore

        assert stat.std_time_to_first_token == np.std([4, 5]) * (1e-6)  # type: ignore
        assert stat.std_inter_token_latency == np.std([4, 2]) * (1e-6)  # type: ignore
        assert stat.std_output_token_throughput_per_request == pytest.approx(  # type: ignore
            np.std(ottpr)
        )
        assert stat.std_num_output_token == np.std([3, 6])  # type: ignore
        assert stat.std_num_input_token == np.std([3, 4])  # type: ignore

        oott = 9 / ns_to_sec(14)
        assert stat.avg_output_token_throughput == pytest.approx(oott)  # type: ignore

        # check non-existing profile data
        with pytest.raises(KeyError):
            pd.get_statistics(infer_mode="concurrency", load_level="40")

    def test_merged_sse_response(self, mock_read_write: pytest.MonkeyPatch) -> None:
        """Test merging the multiple sse response."""
        res_timestamps = [0, 1, 2, 3]
        res_outputs = [
            {
                "response": 'data: {"choices":[{"delta":{"content":"aaa"}}],"object":"chat.completion.chunk"}\n\n'
            },
            {
                "response": (
                    'data: {"choices":[{"delta":{"content":"abc"}}],"object":"chat.completion.chunk"}\n\n'
                    'data: {"choices":[{"delta":{"content":"1234"}}],"object":"chat.completion.chunk"}\n\n'
                    'data: {"choices":[{"delta":{"content":"helloworld"}}],"object":"chat.completion.chunk"}\n\n'
                )
            },
            {"response": "data: [DONE]\n\n"},
        ]
        expected_response = '{"choices": [{"delta": {"content": "abc1234helloworld"}}], "object": "chat.completion.chunk"}'

        tokenizer = get_tokenizer(DEFAULT_TOKENIZER)
        pd = LLMProfileDataParser(
            filename=Path("openai_profile_export.json"),
            tokenizer=tokenizer,
        )

        pd._preprocess_response(res_timestamps, res_outputs)
        assert res_outputs[1]["response"] == expected_response

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
            num_input_tokens=[12, 34],
        )
        assert metrics.get_base_name("time_to_first_tokens") == "time_to_first_token"
        assert metrics.get_base_name("inter_token_latencies") == "inter_token_latency"
        assert (
            metrics.get_base_name("output_token_throughputs_per_request")
            == "output_token_throughput_per_request"
        )
        assert metrics.get_base_name("num_output_tokens") == "num_output_token"
        assert metrics.get_base_name("num_input_tokens") == "num_input_token"
        with pytest.raises(KeyError):
            metrics.get_base_name("hello1234")

    def test_empty_response(self, mock_read_write: pytest.MonkeyPatch) -> None:
        """Check if it handles all empty responses."""
        tokenizer = get_tokenizer(DEFAULT_TOKENIZER)

        # Should not throw error
        _ = LLMProfileDataParser(
            filename=Path("empty_profile_export.json"),
            tokenizer=tokenizer,
        )

    empty_profile_data = {
        "service_kind": "openai",
        "endpoint": "v1/chat/completions",
        "experiments": [
            {
                "experiment": {
                    "mode": "concurrency",
                    "value": 10,
                },
                "requests": [
                    {
                        "timestamp": 1,
                        "request_inputs": {
                            "payload": '{"messages":[{"role":"user","content":"This is test"}],"model":"llama-2-7b","stream":true}',
                        },
                        "response_timestamps": [3, 5, 8],
                        "response_outputs": [
                            {
                                "response": 'data: {"id":"abc","object":"chat.completion.chunk","created":123,"model":"llama-2-7b","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}\n\n'
                            },
                            {
                                "response": 'data: {"id":"abc","object":"chat.completion.chunk","created":123,"model":"llama-2-7b","choices":[{"index":0,"delta":{"content":""},"finish_reason":null}]}\n\n'
                            },
                            {"response": "data: [DONE]\n\n"},
                        ],
                    },
                ],
            },
        ],
    }

    openai_profile_data = {
        "service_kind": "openai",
        "endpoint": "v1/chat/completions",
        "experiments": [
            {
                "experiment": {
                    "mode": "concurrency",
                    "value": 10,
                },
                "requests": [
                    {
                        "timestamp": 1,
                        "request_inputs": {
                            "payload": '{"messages":[{"role":"user","content":"This is test"}],"model":"llama-2-7b","stream":true}',
                        },
                        # the first, and the last two responses will be ignored because they have no "content"
                        "response_timestamps": [3, 5, 8, 12, 13, 14],
                        "response_outputs": [
                            {
                                "response": 'data: {"id":"abc","object":"chat.completion.chunk","created":123,"model":"llama-2-7b","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}\n\n'
                            },
                            {
                                "response": 'data: {"id":"abc","object":"chat.completion.chunk","created":123,"model":"llama-2-7b","choices":[{"index":0,"delta":{"content":"I"},"finish_reason":null}]}\n\n'
                            },
                            {
                                "response": 'data: {"id":"abc","object":"chat.completion.chunk","created":123,"model":"llama-2-7b","choices":[{"index":0,"delta":{"content":" like"},"finish_reason":null}]}\n\n'
                            },
                            {
                                "response": 'data: {"id":"abc","object":"chat.completion.chunk","created":123,"model":"llama-2-7b","choices":[{"index":0,"delta":{"content":" dogs"},"finish_reason":null}]}\n\n'
                            },
                            {
                                "response": 'data: {"id":"abc","object":"chat.completion.chunk","created":123,"model":"llama-2-7b","choices":[{"index":0,"delta":{},"finish_reason":null}]}\n\n'
                            },
                            {"response": "data: [DONE]\n\n"},
                        ],
                    },
                    {
                        "timestamp": 2,
                        "request_inputs": {
                            "payload": '{"messages":[{"role":"user","content":"This is test too"}],"model":"llama-2-7b","stream":true}',
                        },
                        # the first, and the last two responses will be ignored because they have no "content"
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
        "service_kind": "triton",
        "endpoint": "",
        "experiments": [
            {
                "experiment": {
                    "mode": "concurrency",
                    "value": 10,
                },
                "requests": [
                    {
                        "timestamp": 1,
                        "request_inputs": {"text_input": "This is test"},
                        "response_timestamps": [3, 5, 8],
                        "response_outputs": [
                            {"text_output": "I"},
                            {"text_output": " like"},
                            {"text_output": " dogs"},
                        ],
                    },
                    {
                        "timestamp": 2,
                        "request_inputs": {"text_input": "This is test too"},
                        "response_timestamps": [4, 7, 11],
                        "response_outputs": [
                            {"text_output": "I"},
                            {"text_output": " don't"},
                            {"text_output": " cook food"},
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
                        "request_inputs": {"text_input": "This is test"},
                        "response_timestamps": [7, 8, 13, 18],
                        "response_outputs": [
                            {"text_output": "cat"},
                            {"text_output": " is"},
                            {"text_output": " cool"},
                            {"text_output": " too"},
                        ],
                    },
                    {
                        "timestamp": 3,
                        "request_inputs": {"text_input": "This is test too"},
                        "response_timestamps": [6, 8, 11],
                        "response_outputs": [
                            {"text_output": "it's"},
                            {"text_output": " very"},
                            {"text_output": " simple work"},
                        ],
                    },
                ],
            },
        ],
    }
