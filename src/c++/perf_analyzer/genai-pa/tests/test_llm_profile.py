#!/usr/bin/env python3

# Copyright 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from pathlib import Path

import numpy as np
import pytest
from genai_pa.llm_profile import LLMMetrics, LLMProfileData
from genai_pa.utils import remove_file
from transformers import AutoTokenizer


class TestLLMProfileData:
    @pytest.fixture
    def prepare_profile_data(self) -> None:
        self.path = Path("temp_profile_export.json")
        self.profile_data = {
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
                            "response_outputs": ["dogs", "are", "cool"],
                        },
                        {
                            "timestamp": 2,
                            "response_timestamps": [4, 7, 11],
                            "response_outputs": ["I", "don't", "cook food"],
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
                            "response_outputs": ["cats", "are", "cool", "too"],
                        },
                        {
                            "timestamp": 3,
                            "response_timestamps": [6, 8, 11],
                            "response_outputs": ["it's", "very", "simple work"],
                        },
                    ],
                },
            ],
        }

        with open(self.path, "w") as f:
            json.dump(self.profile_data, f)

        yield None

        # clean up
        remove_file(self.path)

    def test_llm_profile_data(self, prepare_profile_data) -> None:
        """Collect LLM metrics from profile export data and check values.

        Metrics
        * time to first tokens
            - experiment 1: [3 - 1, 4 - 2] = [2, 2]
            - experiment 2: [7 - 5, 6 - 3] = [2, 3]
        * inter token latencies
            - experiment 1: [5 - 3, 8 - 5, 7 - 4, 10 - 7] = [2, 3, 3, 4]
            - experiment 2: [8 - 7, 13 - 8, 18 - 13, 8 - 6, 11 - 8] = [1, 5, 5, 2, 3]
        * output token throughputs
            - experiment 1: [3/(8 - 1), 5/(11 - 2)] = [3/7, 5/9]
            - experiment 2: [4/(18 - 5), 5/(11 - 3)] = [4/13, 5/8]
        """
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        pd = LLMProfileData("temp_profile_export.json", tokenizer)

        # experiment 1 statistics
        stat = pd.get_statistics(infer_mode="concurrency", load_level=10)
        assert stat.avg_time_to_first_token == 2
        assert stat.avg_inter_token_latency == 3
        assert stat.avg_output_token_throughput == pytest.approx(31 / 63)
        assert stat.p50_time_to_first_token == 2
        assert stat.p50_inter_token_latency == 3
        assert stat.p50_output_token_throughput == pytest.approx(31 / 63)
        assert stat.min_time_to_first_token == 2
        assert stat.min_inter_token_latency == 2
        assert stat.min_output_token_throughput == pytest.approx(3 / 7)
        assert stat.max_time_to_first_token == 2
        assert stat.max_inter_token_latency == 4
        assert stat.max_output_token_throughput == pytest.approx(5 / 9)
        assert stat.std_time_to_first_token == np.std([2, 2])
        assert stat.std_inter_token_latency == np.std([2, 3, 3, 4])
        assert stat.std_output_token_throughput == np.std([3 / 7, 5 / 9])

        # experiment 2 statistics
        stat = pd.get_statistics(infer_mode="request_rate", load_level=2.0)
        assert stat.avg_time_to_first_token == 2.5
        assert stat.avg_inter_token_latency == 3.2
        assert stat.avg_output_token_throughput == pytest.approx(97 / 208)
        assert stat.p50_time_to_first_token == 2.5
        assert stat.p50_inter_token_latency == 3
        assert stat.p50_output_token_throughput == pytest.approx(97 / 208)
        assert stat.min_time_to_first_token == 2
        assert stat.min_inter_token_latency == 1
        assert stat.min_output_token_throughput == pytest.approx(4 / 13)
        assert stat.max_time_to_first_token == 3
        assert stat.max_inter_token_latency == 5
        assert stat.max_output_token_throughput == pytest.approx(5 / 8)
        assert stat.std_time_to_first_token == np.std([2, 3])
        assert stat.std_inter_token_latency == np.std([1, 5, 5, 2, 3])
        assert stat.std_output_token_throughput == np.std([4 / 13, 5 / 8])

        # check non-existing profile data
        with pytest.raises(KeyError):
            pd.get_statistics(infer_mode="concurrency", load_level=30)

    def test_llm_metrics_get_base_name(self) -> None:
        """Test get_base_name method in LLMMetrics class."""
        metrics = LLMMetrics(
            time_to_first_tokens=[1, 2, 3],
            inter_token_latencies=[4, 5],
            output_token_throughputs=[7, 8, 9],
        )
        assert metrics.get_base_name("time_to_first_tokens") == "time_to_first_token"
        assert metrics.get_base_name("inter_token_latencies") == "inter_token_latency"
        assert (
            metrics.get_base_name("output_token_throughputs")
            == "output_token_throughput"
        )
        with pytest.raises(ValueError):
            metrics.get_base_name("hello1234")
