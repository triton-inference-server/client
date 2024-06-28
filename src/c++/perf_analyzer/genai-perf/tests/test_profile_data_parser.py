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
from genai_perf.metrics import Metrics
from genai_perf.profile_data_parser import ProfileDataParser


def ns_to_sec(ns: int) -> Union[int, float]:
    """Convert from nanosecond to second."""
    return ns / 1e9


class TestProfileDataParser:
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

            if filename == "embedding_profile_export.json":
                tmp_file = StringIO(json.dumps(self.embedding_profile_data))
                return tmp_file
            if filename == "ranking_profile_export.json":
                tmp_file = StringIO(json.dumps(self.ranking_profile_data))
                return tmp_file
            elif filename == "profile_export.csv":
                tmp_file = StringIO()
                tmp_file.write = write.__get__(tmp_file)
                return tmp_file
            else:
                return original_open(filename, *args, **kwargs)

        monkeypatch.setattr("builtins.open", custom_open)

        return written_data

    embedding_profile_data = {
        "service_kind": "openai",
        "endpoint": "v1/embeddings",
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
                            "payload": '{"input":"This is test","model":"NV-Embed-QA","input_type":"passage","encoding_format":"float","truncate":"NONE"}',
                        },
                        "response_timestamps": [3],
                        "response_outputs": [
                            {
                                "response": '{"object":"list","data":[{"index":0,"embedding":[1, 2, 3],"object":"embedding"}],"model":"NV-Embed-QA","usage":{"prompt_tokens":7,"total_tokens":7}}'
                            },
                        ],
                    },
                    {
                        "timestamp": 2,
                        "request_inputs": {
                            "payload": '{"input":"This is test too","model":"NV-Embed-QA","input_type":"passage","encoding_format":"float","truncate":"NONE"}',
                        },
                        "response_timestamps": [5],
                        "response_outputs": [
                            {
                                "response": '{"object":"list","data":[{"index":0,"embedding":[1, 2, 3, 4],"object":"embedding"}],"model":"NV-Embed-QA","usage":{"prompt_tokens":8,"total_tokens":8}}'
                            },
                        ],
                    },
                ],
            },
        ],
    }

    def test_embedding_profile_data(self, mock_read_write: pytest.MonkeyPatch) -> None:
        """Collect base metrics from profile export data and check values.

        Metrics
        * request latencies
            - [3 - 1, 5 - 2] = [2, 3]
        * request throughputs
            - [2 / (5e-9 - 1e-9)] = [5e8]
        """
        pd = ProfileDataParser(filename=Path("embedding_profile_export.json"))

        # experiment 1 statistics
        stats = pd.get_statistics(infer_mode="concurrency", load_level="10")
        metrics = stats.metrics
        stats_dict = stats.stats_dict
        assert isinstance(metrics, Metrics)

        assert metrics.request_latencies == [2, 3]
        assert metrics.request_throughputs == [pytest.approx(5e8)]

        assert stats_dict["request_latency"]["avg"] == pytest.approx(2.5)  # type: ignore
        assert stats_dict["request_latency"]["p50"] == pytest.approx(2.5)  # type: ignore
        assert stats_dict["request_latency"]["min"] == pytest.approx(2)  # type: ignore
        assert stats_dict["request_latency"]["max"] == pytest.approx(3)  # type: ignore
        assert stats_dict["request_latency"]["std"] == np.std([2, 3])  # type: ignore

        assert stats_dict["request_throughput"]["avg"] == pytest.approx(5e8)  # type: ignore

    ranking_profile_data = {
        "service_kind": "openai",
        "endpoint": "v1/ranking",
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
                            "payload": '{"query":{"text":"This is a test."},"passages":[{"text":"test output one"},{"text":"test output two"},{"text":"test output three"}],"model":"nv-rerank-qa-mistral-4b:1","truncate":"END"}',
                        },
                        "response_timestamps": [3],
                        "response_outputs": [
                            {
                                "response": '{"rankings":[{"index":0,"logit":-5.98828125},{"index":1,"logit":-6.828125},{"index":2,"logit":-7.60546875}]}'
                            },
                        ],
                    },
                    {
                        "timestamp": 2,
                        "request_inputs": {
                            "payload": '{"query":{"text":"This is a test."},"passages":[{"text":"test output one"},{"text":"test output two"},{"text":"test output three"}],"model":"nv-rerank-qa-mistral-4b:1","truncate":"END"}',
                        },
                        "response_timestamps": [5],
                        "response_outputs": [
                            {
                                "response": '{"rankings":[{"index":2,"logit":-6.15625},{"index":1,"logit":-7.83984375},{"index":0,"logit":-7.84765625}]}'
                            },
                        ],
                    },
                ],
            },
        ],
    }

    def test_ranking_profile_data(self, mock_read_write: pytest.MonkeyPatch) -> None:
        """Collect base metrics from profile export data and check values.

        Metrics
        * request latencies
            - [3 - 1, 5 - 2] = [2, 3]
        * request throughputs
            - [2 / (5e-9 - 1e-9)] = [5e8]
        """
        pd = ProfileDataParser(filename=Path("ranking_profile_export.json"))

        # experiment 1 statistics
        stats = pd.get_statistics(infer_mode="concurrency", load_level="10")
        metrics = stats.metrics
        stats_dict = stats.stats_dict
        assert isinstance(metrics, Metrics)

        assert metrics.request_latencies == [2, 3]
        assert metrics.request_throughputs == [pytest.approx(5e8)]

        assert stats_dict["request_latency"]["avg"] == pytest.approx(2.5)  # type: ignore
        assert stats_dict["request_latency"]["p50"] == pytest.approx(2.5)  # type: ignore
        assert stats_dict["request_latency"]["min"] == pytest.approx(2)  # type: ignore
        assert stats_dict["request_latency"]["max"] == pytest.approx(3)  # type: ignore
        assert stats_dict["request_latency"]["std"] == np.std([2, 3])  # type: ignore

        assert stats_dict["request_throughput"]["avg"] == pytest.approx(5e8)  # type: ignore
