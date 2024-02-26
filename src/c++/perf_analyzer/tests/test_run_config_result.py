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

import unittest
from unittest.mock import MagicMock, patch

from model_analyzer.result.run_config_result import RunConfigResult
from tests.common.test_utils import (
    construct_constraint_manager,
    construct_run_config_measurement,
    convert_non_gpu_metrics_to_data,
)

from .common import test_result_collector as trc


class TestRunConfigResult(trc.TestResultCollector):
    def setUp(self):
        self.default_constraint_manager = construct_constraint_manager(
            """
            profile_models:
              modelA:
                constraints:
                  perf_latency_p99:
                    max: 100
              modelB:
                constraints:
                  perf_latency_p99:
                    max: 50
            """
        )
        self._construct_empty_rcr()
        self._construct_throughput_with_latency_constraint_rcr()
        self._construct_throughput_with_latency_constraint_multi_model_rcr()

    def tearDown(self):
        patch.stopall()

    def test_model_name(self):
        """
        Test that model_name is correctly returned
        """
        self.assertEqual(self.rcr_empty.model_name(), self.model_name)

    def test_run_config(self):
        """
        Test that run_config is correctly returned
        """
        self.assertEqual(self.rcr_empty.run_config(), self.run_config)

    def test_failing_empty(self):
        """
        Test that failing returns true if no measurements are added
        """
        self.assertTrue(self.rcr_empty.failing())

    def test_failing_true(self):
        """
        Test that failing returns true if only failing measurements are present
        """
        rcr = self._rcr_throughput_with_latency_constraint

        for i in range(2, 6):
            self._add_rcm_to_rcr(rcr, throughput_value=10 * i, latency_value=100 * i)

        self.assertTrue(rcr.failing())

    def test_failing_true_multi_model(self):
        """
        Test that failing returns true if only failing measurements are present
        in a multi-model configuration
        """
        rcr = self._rcr_throughput_with_latency_constraint_multi_model

        # Model A will have some passing measurements,
        # Model B will have all failing measurements
        for i in range(1, 6):
            self._add_multi_model_rcm_to_rcr(
                rcr, throughput_values=[5 * i, 20 * i], latency_values=[50 * i, 200 * i]
            )

        self.assertTrue(rcr.failing())

    def test_failing_false(self):
        """
        Test that failing returns false if any passing measurements are present
        """
        rcr = self._rcr_throughput_with_latency_constraint

        # This also tests the boundary condition: exact match (100) is passing
        for i in range(1, 6):
            self._add_rcm_to_rcr(rcr, throughput_value=10 * i, latency_value=100 * i)

        self.assertFalse(rcr.failing())

    def test_failing_false_multi_model(self):
        """
        Test that failing returns false if any passing measurements are present
        in a multi-model configuration
        """
        rcr = self._rcr_throughput_with_latency_constraint_multi_model

        for i in range(1, 6):
            self._add_multi_model_rcm_to_rcr(
                rcr, throughput_values=[3 * i, 4 * i], latency_values=[30 * i, 40 * i]
            )

        self.assertFalse(rcr.failing())

    def test_passing_failing_measurements(self):
        """
        Test that passing/failing measurements have the correct number
        of entries
        """
        rcr = self._rcr_throughput_with_latency_constraint

        # 4 passing, 6 failing
        for i in range(1, 11):
            self._add_rcm_to_rcr(rcr, throughput_value=10 * i, latency_value=25 * i)

        # The heap is unordered so just checking for correct count
        self.assertEqual(len(rcr.passing_measurements()), 4)
        self.assertEqual(len(rcr.failing_measurements()), 6)

    def test_passing_failing_measurements_multi_model(self):
        """
        Test that passing/failing measurements have the correct number
        of entries
        """
        rcr = self._rcr_throughput_with_latency_constraint_multi_model

        # 2 passing, 10 failing
        for i in range(1, 13):
            self._add_multi_model_rcm_to_rcr(
                rcr, throughput_values=[10 * i, 15 * i], latency_values=[15 * i, 20 * i]
            )

        # The heap is unordered so just checking for correct count
        self.assertEqual(len(rcr.passing_measurements()), 2)
        self.assertEqual(len(rcr.failing_measurements()), 10)

    def test_top_n_failing(self):
        """
        Test that the top N failing measurements are returned
        if no passing measurements are present
        """
        rcr = self._rcr_throughput_with_latency_constraint

        for i in range(2, 6):
            self._add_rcm_to_rcr(rcr, throughput_value=10 * i, latency_value=100 * i)

        # Failing measurements are returned most to least throughput
        failing_non_gpu_data = [
            convert_non_gpu_metrics_to_data(
                {"perf_throughput": 10 * i, "perf_latency_p99": 100 * i}
            )
            for i in range(5, 1, -1)
        ]

        top_n_measurements = rcr.top_n_measurements(3)

        for i in range(3):
            self.assertEqual(
                top_n_measurements[i].non_gpu_data(), [failing_non_gpu_data[i]]
            )

    def test_top_n_failing_multi_model(self):
        """
        Test that the top N failing measurements are returned
        if no passing measurements are present in a multi-model
        configuration
        """
        rcr = self._rcr_throughput_with_latency_constraint_multi_model

        for i in range(1, 6):
            self._add_multi_model_rcm_to_rcr(
                rcr, throughput_values=[5 * i, 20 * i], latency_values=[50 * i, 200 * i]
            )

        # Failing measurements are returned most to least throughput
        failing_non_gpu_data = []
        failing_non_gpu_data.append(
            [
                convert_non_gpu_metrics_to_data(
                    {"perf_throughput": 5 * i, "perf_latency_p99": 50 * i}
                )
                for i in range(5, 1, -1)
            ]
        )

        failing_non_gpu_data.append(
            [
                convert_non_gpu_metrics_to_data(
                    {"perf_throughput": 20 * i, "perf_latency_p99": 200 * i}
                )
                for i in range(5, 1, -1)
            ]
        )

        top_n_measurements = rcr.top_n_measurements(3)

        for i in range(3):
            self.assertEqual(
                top_n_measurements[i].non_gpu_data(),
                [failing_non_gpu_data[0][i], failing_non_gpu_data[1][i]],
            )

    def test_top_n_passing(self):
        """
        Test that the top N passing measurements are returned
        """
        rcr = self._rcr_throughput_with_latency_constraint

        # 4 passing, 6 failing
        for i in range(1, 11):
            self._add_rcm_to_rcr(rcr, throughput_value=10 * i, latency_value=25 * i)

        # Passing measurements are returned most to least throughput
        passing_non_gpu_data = [
            convert_non_gpu_metrics_to_data(
                {"perf_throughput": 10 * i, "perf_latency_p99": 25 * i}
            )
            for i in range(4, 0, -1)
        ]

        top_n_measurements = rcr.top_n_measurements(3)

        for i in range(3):
            self.assertEqual(
                top_n_measurements[i].non_gpu_data(), [passing_non_gpu_data[i]]
            )

    def test_top_n_passing_multi_model(self):
        """
        Test that the top N passing measurements are returned
        in a multi-model configuration
        """

        rcr = self._rcr_throughput_with_latency_constraint_multi_model

        # 4 passing, 6 failing
        for i in range(1, 11):
            self._add_rcm_to_rcr(rcr, throughput_value=10 * i, latency_value=25 * i)

        # Passing measurements are returned most to least throughput
        passing_non_gpu_data = [
            convert_non_gpu_metrics_to_data(
                {"perf_throughput": 10 * i, "perf_latency_p99": 25 * i}
            )
            for i in range(4, 0, -1)
        ]

        top_n_measurements = rcr.top_n_measurements(3)

        for i in range(3):
            self.assertEqual(
                top_n_measurements[i].non_gpu_data(), [passing_non_gpu_data[i]]
            )

    def _construct_empty_rcr(self):
        self.model_name = MagicMock()
        self.run_config = MagicMock()

        self.rcr_empty = RunConfigResult(
            model_name=self.model_name,
            run_config=self.run_config,
            comparator=MagicMock(),
            constraint_manager=MagicMock(),
        )

    def _construct_throughput_with_latency_constraint_rcr(self):
        self._rcr_throughput_with_latency_constraint = RunConfigResult(
            model_name=MagicMock(),
            run_config=MagicMock(),
            comparator=[{"perf_throughput": 1}],
            constraint_manager=self.default_constraint_manager,
        )

    def _construct_throughput_with_latency_constraint_multi_model_rcr(self):
        self._rcr_throughput_with_latency_constraint_multi_model = RunConfigResult(
            model_name=MagicMock(),
            run_config=MagicMock(),
            comparator=[{"perf_throughput": 1}],
            constraint_manager=self.default_constraint_manager,
        )

    def _add_rcm_to_rcr(self, rcr, throughput_value, latency_value):
        rcr.add_run_config_measurement(
            self._construct_single_model_rcm(throughput_value, latency_value)
        )

    def _add_multi_model_rcm_to_rcr(self, rcr, throughput_values, latency_values):
        rcr.add_run_config_measurement(
            self._construct_multi_model_rcm(throughput_values, latency_values)
        )

    def _construct_single_model_rcm(self, throughput_value, latency_value):
        return construct_run_config_measurement(
            model_name="modelA",
            model_config_names=["modelA_config_0"],
            model_specific_pa_params=MagicMock(),
            gpu_metric_values=MagicMock(),
            non_gpu_metric_values=[
                {"perf_throughput": throughput_value, "perf_latency_p99": latency_value}
            ],
        )

    def _construct_multi_model_rcm(self, throughput_values, latency_values):
        return construct_run_config_measurement(
            model_name="modelA,modelB",
            model_config_names=["modelA_config_0", "modelB_config_0"],
            model_specific_pa_params=MagicMock(),
            gpu_metric_values=MagicMock(),
            non_gpu_metric_values=[
                {
                    "perf_throughput": throughput_values[0],
                    "perf_latency_p99": latency_values[0],
                },
                {
                    "perf_throughput": throughput_values[1],
                    "perf_latency_p99": latency_values[1],
                },
            ],
        )


if __name__ == "__main__":
    unittest.main()
