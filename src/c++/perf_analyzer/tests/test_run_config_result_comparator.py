#!/usr/bin/env python3

# Copyright 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import unittest
from typing import List
from unittest.mock import MagicMock, patch

from model_analyzer.result.run_config_result_comparator import RunConfigResultComparator

from .common import test_result_collector as trc
from .common.test_utils import construct_run_config_result


class TestRunConfigResultComparatorMethods(trc.TestResultCollector):
    def setUp(self):
        self._initialize_metrics()

    def tearDown(self):
        patch.stopall()

    def test_throughput_driven(self):
        objective_spec = [{"perf_throughput": 2, "perf_latency_p99": 1}]
        model_weights = [1]

        self._check_run_config_result_comparison(
            objective_spec=objective_spec,
            model_weights=model_weights,
            avg_gpu_metrics1=self.avg_gpu_metrics1,
            avg_non_gpu_metrics1=self.avg_non_gpu_metrics1,
            avg_gpu_metrics2=self.avg_gpu_metrics2,
            avg_non_gpu_metrics2=self.avg_non_gpu_metrics2,
            expected_result=False,
        )

    def test_throughput_driven_multi_model(self):
        objective_spec = [
            {"perf_throughput": 2, "perf_latency_p99": 1},
            {"perf_throughput": 2, "perf_latency_p99": 1},
        ]
        model_weights = [1, 1]

        self._check_run_config_result_comparison(
            objective_spec=objective_spec,
            model_weights=model_weights,
            avg_gpu_metrics1=self.avg_gpu_metrics1,
            avg_non_gpu_metrics1=self.avg_non_gpu_metrics_multi1,
            avg_gpu_metrics2=self.avg_gpu_metrics2,
            avg_non_gpu_metrics2=self.avg_non_gpu_metrics_multi2,
            expected_result=False,
            model_name="test_model",
            model_config_names=["test_model_config_0", "test_model_config_1"],
        )

    def test_latency_driven(self):
        objective_spec = [
            {"perf_throughput": 1, "perf_latency_p99": 2},
        ]
        model_weights = [1]

        self._check_run_config_result_comparison(
            objective_spec=objective_spec,
            model_weights=model_weights,
            avg_gpu_metrics1=self.avg_gpu_metrics1,
            avg_non_gpu_metrics1=self.avg_non_gpu_metrics1,
            avg_gpu_metrics2=self.avg_gpu_metrics2,
            avg_non_gpu_metrics2=self.avg_non_gpu_metrics2,
            expected_result=True,
        )

    def test_latency_driven_multi(self):
        objective_spec = [
            {"perf_throughput": 1, "perf_latency_p99": 2},
            {"perf_throughput": 1, "perf_latency_p99": 2},
        ]
        model_weights = [1, 1]

        self._check_run_config_result_comparison(
            objective_spec=objective_spec,
            model_weights=model_weights,
            avg_gpu_metrics1=self.avg_gpu_metrics1,
            avg_non_gpu_metrics1=self.avg_non_gpu_metrics_multi1,
            avg_gpu_metrics2=self.avg_gpu_metrics2,
            avg_non_gpu_metrics2=self.avg_non_gpu_metrics_multi2,
            expected_result=True,
            model_name="test_model",
            model_config_names=["test_model_config_0", "test_model_config_1"],
        )

    def test_equal_weight(self):
        objective_spec = [{"perf_throughput": 1, "perf_latency_p99": 1}]
        model_weights = [1]

        self._check_run_config_result_comparison(
            objective_spec=objective_spec,
            model_weights=model_weights,
            avg_gpu_metrics1=self.avg_gpu_metrics1,
            avg_non_gpu_metrics1=self.avg_non_gpu_metrics1,
            avg_gpu_metrics2=self.avg_gpu_metrics2,
            avg_non_gpu_metrics2=self.avg_non_gpu_metrics2,
            value_step2=2,
            expected_result=False,
        )

    def test_equal_weight_multi(self):
        objective_spec = [
            {"perf_throughput": 1, "perf_latency_p99": 1},
            {"perf_throughput": 1, "perf_latency_p99": 1},
        ]
        model_weights = [1, 1]

        self._check_run_config_result_comparison(
            objective_spec=objective_spec,
            model_weights=model_weights,
            avg_gpu_metrics1=self.avg_gpu_metrics1,
            avg_non_gpu_metrics1=self.avg_non_gpu_metrics_multi1,
            avg_gpu_metrics2=self.avg_gpu_metrics2,
            avg_non_gpu_metrics2=self.avg_non_gpu_metrics_multi2,
            value_step2=2,
            expected_result=False,
            model_name="test_model",
            model_config_names=["test_model_config_0", "test_model_config_1"],
        )

    def test_unequal_weight_multi(self):
        """
        Tests that changing just the model weighting will result in a different
        config being selected as better
        """
        objective_spec = [
            {"perf_throughput": 1, "perf_latency_p99": 1},
            {"perf_throughput": 1, "perf_latency_p99": 1},
        ]

        # With equal weighting - config 1 is better
        model_weights = [1, 1]
        self._check_run_config_result_comparison(
            objective_spec=objective_spec,
            model_weights=model_weights,
            avg_gpu_metrics1=self.avg_gpu_metrics1,
            avg_non_gpu_metrics1=self.avg_non_gpu_metrics_weighted_multi1,
            avg_gpu_metrics2=self.avg_gpu_metrics2,
            avg_non_gpu_metrics2=self.avg_non_gpu_metrics_weighted_multi2,
            value_step2=2,
            expected_result=False,
            model_name="test_model",
            model_config_names=["test_model_config_0", "test_model_config_1"],
        )

        # With an unequal weighting - config 0 is better
        model_weights = [3, 1]
        self._check_run_config_result_comparison(
            objective_spec=objective_spec,
            model_weights=model_weights,
            avg_gpu_metrics1=self.avg_gpu_metrics1,
            avg_non_gpu_metrics1=self.avg_non_gpu_metrics_weighted_multi1,
            avg_gpu_metrics2=self.avg_gpu_metrics2,
            avg_non_gpu_metrics2=self.avg_non_gpu_metrics_weighted_multi2,
            value_step2=2,
            expected_result=True,
            model_name="test_model",
            model_config_names=["test_model_config_0", "test_model_config_1"],
        )

    def _check_run_config_result_comparison(
        self,
        objective_spec,
        model_weights: List[int],
        avg_gpu_metrics1,
        avg_gpu_metrics2,
        avg_non_gpu_metrics1,
        avg_non_gpu_metrics2,
        value_step1=1,
        value_step2=1,
        expected_result=0,
        model_name="test_model",
        model_config_names=["test_model"],
    ):
        """
        Helper function that takes all the data needed to
        construct two RunConfigResults, constructs and runs a
        comparator and checks that it produces the expected
        value.
        """

        result_comparator = RunConfigResultComparator(
            metric_objectives_list=objective_spec, model_weights=model_weights
        )

        result1 = construct_run_config_result(
            avg_gpu_metric_values=avg_gpu_metrics1,
            avg_non_gpu_metric_values_list=avg_non_gpu_metrics1,
            comparator=result_comparator,
            value_step=value_step1,
            run_config=MagicMock(),
            constraint_manager=MagicMock(),
            model_name=model_name,
            model_config_names=model_config_names,
        )

        result2 = construct_run_config_result(
            avg_gpu_metric_values=avg_gpu_metrics2,
            avg_non_gpu_metric_values_list=avg_non_gpu_metrics2,
            comparator=result_comparator,
            value_step=value_step2,
            run_config=MagicMock(),
            constraint_manager=MagicMock(),
            model_name=model_name,
            model_config_names=model_config_names,
        )

        self.assertEqual(
            result_comparator.is_better_than(result1, result2), expected_result
        )

    def _initialize_metrics(self):
        self.avg_gpu_metrics1 = {0: {"gpu_used_memory": 5000, "gpu_utilization": 50}}
        self.avg_gpu_metrics2 = {0: {"gpu_used_memory": 6000, "gpu_utilization": 60}}

        self.avg_non_gpu_metrics1 = [{"perf_throughput": 100, "perf_latency_p99": 4000}]

        self.avg_non_gpu_metrics2 = [{"perf_throughput": 200, "perf_latency_p99": 8000}]

        self.avg_non_gpu_metrics_multi1 = [
            {"perf_throughput": 100, "perf_latency_p99": 4000},
            {"perf_throughput": 150, "perf_latency_p99": 6000},
        ]

        self.avg_non_gpu_metrics_multi2 = [
            {"perf_throughput": 200, "perf_latency_p99": 8000},
            {"perf_throughput": 250, "perf_latency_p99": 10000},
        ]

        self.avg_non_gpu_metrics_weighted_multi1 = [
            {"perf_throughput": 1000, "perf_latency_p99": 50},
            {"perf_throughput": 1000, "perf_latency_p99": 100},
        ]

        self.avg_non_gpu_metrics_weighted_multi2 = [
            {"perf_throughput": 2000, "perf_latency_p99": 200},
            {"perf_throughput": 500, "perf_latency_p99": 20},
        ]


if __name__ == "__main__":
    unittest.main()
