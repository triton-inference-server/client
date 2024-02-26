#!/usr/bin/env python3

# Copyright 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import unittest
from unittest.mock import patch

from model_analyzer.result.model_config_measurement import ModelConfigMeasurement
from tests.common.test_utils import convert_non_gpu_metrics_to_data, default_encode

from .common import test_result_collector as trc


class TestModelConfigMeasurement(trc.TestResultCollector):
    def setUp(self):
        self.model_config_name = "modelA"
        self.model_specific_pa_params = {"batch_size": 1, "concurrency-range": 1}

        self.non_gpu_metric_values = {
            "perf_throughput": 1000,
            "perf_latency_p99": 20,
            "cpu_used_ram": 1000,
        }

        self.mcmA = self._construct_model_config_measurement(
            self.model_config_name,
            self.model_specific_pa_params,
            self.non_gpu_metric_values,
        )

        mcmB_non_gpu_metric_values = {
            "perf_throughput": 2000,
            "perf_latency_p99": 40,
            "cpu_used_ram": 1000,
        }

        self.mcmB = self._construct_model_config_measurement(
            "modelB", self.model_specific_pa_params, mcmB_non_gpu_metric_values
        )

        self.mcmC = self._construct_model_config_measurement(
            "modelC", self.model_specific_pa_params, {}
        )

        self.mcmD = self._construct_model_config_measurement(
            "modelD", self.model_specific_pa_params, {}
        )

    def tearDown(self):
        patch.stopall()

    def test_init(self):
        """
        Test that values are properly initialized
        """
        self.assertEqual(self.mcmA.model_config_name(), self.model_config_name)
        self.assertEqual(
            self.mcmA.model_specific_pa_params(), self.model_specific_pa_params
        )
        self.assertEqual(
            self.mcmA.non_gpu_data(),
            convert_non_gpu_metrics_to_data(self.non_gpu_metric_values),
        )

    def test_get_metric_found(self):
        """
        Test that non-gpu metrics can be correctly returned
        """
        non_gpu_data = convert_non_gpu_metrics_to_data(self.non_gpu_metric_values)

        self.assertEqual(self.mcmA.get_metric("perf_throughput"), non_gpu_data[0])
        self.assertEqual(self.mcmA.get_metric("perf_latency_p99"), non_gpu_data[1])
        self.assertEqual(self.mcmA.get_metric("cpu_used_ram"), non_gpu_data[2])

    def test_get_metric_not_found(self):
        """
        Test that an incorrect metric search returns None
        """
        self.assertEqual(self.mcmA.get_metric("XXXXX"), None)

    def test_get_metric_value_found(self):
        """
        Test that non-gpu metric values can be correctly returned
        """
        self.assertEqual(
            self.mcmA.get_metric_value("perf_throughput"),
            self.non_gpu_metric_values["perf_throughput"],
        )
        self.assertEqual(
            self.mcmA.get_metric_value("perf_latency_p99"),
            self.non_gpu_metric_values["perf_latency_p99"],
        )
        self.assertEqual(
            self.mcmA.get_metric_value("cpu_used_ram"),
            self.non_gpu_metric_values["cpu_used_ram"],
        )

    def test_get_metric_value_not_found(self):
        """
        Test that an incorrect metric value search returns the correct value
        """
        self.assertEqual(self.mcmA.get_metric_value("XXXXX"), 0)
        self.assertEqual(self.mcmA.get_metric_value("XXXXX", 100), 100)

    def test_set_metric_weighting(self):
        # Default
        self.assertEqual({"perf_throughput": 1}, self.mcmA._metric_weights)

        self.mcmA.set_metric_weighting({"perf_throughput": 2, "perf_latency_p99": 3})
        expected_mw = {"perf_throughput": 2 / 5, "perf_latency_p99": 3 / 5}
        self.assertEqual(expected_mw, self.mcmA._metric_weights)

    def test_calculate_weighted_percentage_gain(self):
        """
        Test that weighted percentages are returned correctly
        """
        self.mcmA.set_metric_weighting({"perf_throughput": 1})
        self.mcmB.set_metric_weighting({"perf_throughput": 1})

        # throuhput: mcmA: 1000, mcmB: 2000
        self.assertEqual(self.mcmA.calculate_weighted_percentage_gain(self.mcmB), -50)
        self.assertEqual(self.mcmB.calculate_weighted_percentage_gain(self.mcmA), 100)

        self.mcmA.set_metric_weighting({"perf_latency_p99": 1})
        self.mcmB.set_metric_weighting({"perf_latency_p99": 1})

        # latency: mcmA: 20, mcmB: 40
        self.assertEqual(self.mcmA.calculate_weighted_percentage_gain(self.mcmB), 100)
        self.assertEqual(self.mcmB.calculate_weighted_percentage_gain(self.mcmA), -50)

        # This illustrates why we need to use score, not percentages to determine
        # which model is better. In both cases we will (correctly) report that
        # mcmA/B is 25% better than the other, even though they are equal
        #
        # mcmA has 50% worse throughput, but 100% better latency
        # mcmB has 100% better latency, but 50% worse throughput
        self.mcmA.set_metric_weighting({"perf_throughput": 1, "perf_latency_p99": 1})
        self.mcmB.set_metric_weighting({"perf_throughput": 1, "perf_latency_p99": 1})
        self.assertEqual(self.mcmA, self.mcmB)
        self.assertEqual(self.mcmA.calculate_weighted_percentage_gain(self.mcmB), 25)
        self.assertEqual(self.mcmB.calculate_weighted_percentage_gain(self.mcmA), 25)

    def test_is_better_than(self):
        """
        Test that individual metric comparison works as expected
        """
        self.mcmA.set_metric_weighting({"perf_throughput": 1})

        # throughput: 1000 is not better than 2000
        self.assertFalse(self.mcmA.is_better_than(self.mcmB))
        self.assertLess(self.mcmA, self.mcmB)

        self.mcmA.set_metric_weighting({"perf_latency_p99": 1})

        # latency: 20 is better than 40
        self.assertTrue(self.mcmA.is_better_than(self.mcmB))
        self.assertGreater(self.mcmA, self.mcmB)

    def test_is_better_than_combo(self):
        """
        Test that combination metric comparison works as expected
        """
        # throuhput: 1000 vs. 2000 (worse), latency: 20 vs. 40 (better)
        # with latency bias mcmA is better
        self.mcmA.set_metric_weighting({"perf_throughput": 1, "perf_latency_p99": 3})

        self.assertTrue(self.mcmA.is_better_than(self.mcmB))

    def test_is_better_than_empty(self):
        """
        Test for correct return values when comparing for/against an empty set
        """
        self.mcmA.set_metric_weighting({"perf_throughput": 1})
        self.mcmC.set_metric_weighting({"perf_throughput": 1})

        self.assertTrue(self.mcmA.is_better_than(self.mcmC))
        self.assertFalse(self.mcmC.is_better_than(self.mcmA))
        self.assertEqual(self.mcmC, self.mcmD)

    def test__eq__(self):
        """
        Test that individual metric equality works as expected
        """
        self.mcmA.set_metric_weighting({"cpu_used_ram": 10})

        self.assertEqual(self.mcmA, self.mcmB)

    def test__eq__combo(self):
        """
        Test that combination metric equality works as expected
        """
        # throuhput: 1000 vs. 2000 (worse), latency: 20 vs. 40 (better)
        # with no bias they are equal
        self.mcmA.set_metric_weighting({"perf_throughput": 1, "perf_latency_p99": 1})

        self.assertEqual(self.mcmA, self.mcmB)

    def test_from_dict(self):
        """
        Test to ensure class can be correctly restored from a dictionary
        """
        mcmA_json = json.dumps(self.mcmA, default=default_encode)

        mcmA_from_dict = ModelConfigMeasurement.from_dict(json.loads(mcmA_json))

        self.assertEqual(
            mcmA_from_dict.model_config_name(), self.mcmA.model_config_name()
        )
        self.assertEqual(
            mcmA_from_dict.model_specific_pa_params(),
            self.mcmA.model_specific_pa_params(),
        )
        self.assertEqual(mcmA_from_dict.non_gpu_data(), self.mcmA.non_gpu_data())

        # Catchall in case something new is added
        self.assertEqual(mcmA_from_dict, self.mcmA)

    def _construct_model_config_measurement(
        self, model_config_name, model_specific_pa_params, non_gpu_metric_values
    ):
        non_gpu_data = convert_non_gpu_metrics_to_data(non_gpu_metric_values)

        return ModelConfigMeasurement(
            model_config_name, model_specific_pa_params, non_gpu_data
        )


if __name__ == "__main__":
    unittest.main()
