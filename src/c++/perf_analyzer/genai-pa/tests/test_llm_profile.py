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
import unittest
from pathlib import Path

import numpy as np
from genai_pa.llm_profile import LLMProfileData


class TestLLMProfileData(unittest.TestCase):
    def setUp(self) -> None:
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

    def test_llm_profile_data(self):
        """Collect LLM metrics from profile export data and check values.

        Metrics
        * time to first tokens
            - experiment 1: [3 - 1, 4 - 2] = [2, 2]
            - experiment 2: [7 - 5, 6 - 3] = [2, 3]
        * inter token latencies
            - experiment 1: [5 - 3, 8 - 5, 7 - 4, 10 - 7] = [2, 3, 3, 4]
            - experiment 2: [8 - 7, 13 - 8, 18 - 13, 8 - 6, 11 - 8] = [1, 5, 5, 2, 3]
        * output token throughputs
            - experiment 1: [3/(8 - 3), 5/(11 - 4)] = [3/5, 5/7]
            - experiment 2: [4/(18 - 7), 5/(11 - 6)] = [4/11, 1]
        """
        pd = LLMProfileData("temp_profile_export.json")

        # experiment 1 statistics
        stat = pd.get_statistics(infer_mode="concurrency", level=10)
        self.assertEqual(stat.avg_time_to_first_token, 2)
        self.assertEqual(stat.avg_inter_token_latency, 3)
        self.assertEqual(stat.avg_output_token_throughput, 23 / 35)
        self.assertEqual(stat.p50_time_to_first_token, 2)
        self.assertEqual(stat.p50_inter_token_latency, 3)
        self.assertEqual(stat.p50_output_token_throughput, 23 / 35)
        self.assertEqual(stat.min_time_to_first_token, 2)
        self.assertEqual(stat.min_inter_token_latency, 2)
        self.assertEqual(stat.min_output_token_throughput, 0.6)
        self.assertEqual(stat.max_time_to_first_token, 2)
        self.assertEqual(stat.max_inter_token_latency, 4)
        self.assertEqual(stat.max_output_token_throughput, 5 / 7)
        self.assertEqual(stat.std_time_to_first_token, np.std([2, 2]))
        self.assertEqual(stat.std_inter_token_latency, np.std([2, 3, 3, 4]))
        self.assertEqual(stat.std_output_token_throughput, np.std([3 / 5, 5 / 7]))

        # experiment 2 statistics
        stat = pd.get_statistics(infer_mode="request_rate", level=2.0)
        self.assertEqual(stat.avg_time_to_first_token, 2.5)
        self.assertEqual(stat.avg_inter_token_latency, 3.2)
        self.assertAlmostEqual(stat.avg_output_token_throughput, 15 / 22)
        self.assertEqual(stat.p50_time_to_first_token, 2.5)
        self.assertEqual(stat.p50_inter_token_latency, 3)
        self.assertAlmostEqual(stat.p50_output_token_throughput, 15 / 22)
        self.assertEqual(stat.min_time_to_first_token, 2)
        self.assertEqual(stat.min_inter_token_latency, 1)
        self.assertEqual(stat.min_output_token_throughput, 4 / 11)
        self.assertEqual(stat.max_time_to_first_token, 3)
        self.assertEqual(stat.max_inter_token_latency, 5)
        self.assertEqual(stat.max_output_token_throughput, 1)
        self.assertEqual(stat.std_time_to_first_token, np.std([2, 3]))
        self.assertEqual(stat.std_inter_token_latency, np.std([1, 5, 5, 2, 3]))
        self.assertEqual(stat.std_output_token_throughput, np.std([4 / 11, 1]))

        # check non-existing profile data
        with self.assertRaises(KeyError):
            pd.get_statistics(infer_mode="concurrency", level=30)

    def tearDown(self) -> None:
        self.path.unlink(missing_ok=True)
