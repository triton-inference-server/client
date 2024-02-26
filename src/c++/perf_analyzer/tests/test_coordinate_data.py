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

from unittest.mock import MagicMock

from model_analyzer.config.generate.coordinate import Coordinate
from model_analyzer.config.generate.coordinate_data import CoordinateData

from .common import test_result_collector as trc
from .common.test_utils import construct_run_config_measurement


class TestCoordinateData(trc.TestResultCollector):
    def _construct_rcm(
        self, throughput: float, latency: float, config_name: str, model_name: str
    ):
        model_config_name = [config_name]

        # yapf: disable
        non_gpu_metric_values = [{
            "perf_throughput": throughput,
            "perf_latency_avg": latency
        }]
        # yapf: enable

        metric_objectives = [{"perf_throughput": 1}]
        weights = [1]

        rcm = construct_run_config_measurement(
            model_name=model_name,
            model_config_names=model_config_name,
            model_specific_pa_params=MagicMock(),
            gpu_metric_values={},
            non_gpu_metric_values=non_gpu_metric_values,
            metric_objectives=metric_objectives,
            model_config_weights=weights,
        )
        return rcm

    def test_basic(self):
        result_data = CoordinateData()

        coordinate = Coordinate([0, 0, 0])
        self.assertEqual(result_data.get_measurement(coordinate), None)
        self.assertEqual(result_data.get_visit_count(coordinate), 0)
        self.assertEqual(result_data.is_measured(coordinate), False)
        self.assertEqual(result_data.has_valid_measurement(coordinate), False)

    def test_visit_count(self):
        result_data = CoordinateData()

        coordinate1 = Coordinate([0, 0, 0])
        coordinate2 = Coordinate([0, 4, 1])

        result_data.increment_visit_count(coordinate1)
        self.assertEqual(1, result_data.get_visit_count(coordinate1))

        result_data.increment_visit_count(coordinate2)
        self.assertEqual(1, result_data.get_visit_count(coordinate2))

        result_data.increment_visit_count(coordinate1)
        result_data.increment_visit_count(coordinate1)
        self.assertEqual(3, result_data.get_visit_count(coordinate1))
        self.assertEqual(1, result_data.get_visit_count(coordinate2))

    def test_measurement(self):
        """
        Test if CoordinateData can properly set and get measurements

        Also confirm that is_measured() and has_valid_measurement() work properly
        """
        coordinate_data = CoordinateData()

        coordinate0 = Coordinate([0, 0, 0])
        coordinate1 = Coordinate([0, 4, 1])
        coordinate2 = Coordinate([1, 2, 3])

        rcm0 = self._construct_rcm(
            10, 5, config_name="modelA_config_0", model_name="modelA"
        )
        rcm1 = self._construct_rcm(
            20, 8, config_name="modelB_config_0", model_name="modelB"
        )
        rcm2 = None

        coordinate_data.set_measurement(coordinate0, rcm0)
        coordinate_data.set_measurement(coordinate1, rcm1)
        coordinate_data.set_measurement(coordinate2, rcm2)

        self.assertEqual(coordinate_data.is_measured(coordinate0), True)
        self.assertEqual(coordinate_data.is_measured(coordinate1), True)
        self.assertEqual(coordinate_data.is_measured(coordinate2), True)

        self.assertEqual(coordinate_data.has_valid_measurement(coordinate0), True)
        self.assertEqual(coordinate_data.has_valid_measurement(coordinate1), True)
        self.assertEqual(coordinate_data.has_valid_measurement(coordinate2), False)

        measurement0 = coordinate_data.get_measurement(coordinate0)
        self.assertEqual("modelA_config_0", measurement0.model_variants_name())
        self.assertEqual(10, measurement0.get_non_gpu_metric_value("perf_throughput"))
        self.assertEqual(5, measurement0.get_non_gpu_metric_value("perf_latency_avg"))
        self.assertTrue(coordinate_data.is_measured(coordinate0))

        measurement1 = coordinate_data.get_measurement(coordinate1)
        self.assertEqual("modelB_config_0", measurement1.model_variants_name())
        self.assertEqual(20, measurement1.get_non_gpu_metric_value("perf_throughput"))
        self.assertEqual(8, measurement1.get_non_gpu_metric_value("perf_latency_avg"))

        measurement2 = coordinate_data.get_measurement(coordinate2)
        self.assertEqual(measurement2, None)
