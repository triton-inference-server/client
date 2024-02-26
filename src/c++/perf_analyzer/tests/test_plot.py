#!/usr/bin/env python3

# Copyright 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from model_analyzer.plots.simple_plot import SimplePlot

from .common import test_result_collector as trc
from .common.test_utils import construct_run_config_measurement
from .mocks.mock_matplotlib import MockMatplotlibMethods


class TestPlotMethods(trc.TestResultCollector):
    def setUp(self):
        # mocks
        self.matplotlib_mock = MockMatplotlibMethods()
        self.matplotlib_mock.start()

    def test_create_plot(self):
        # Create a plot and check for call to subplots
        SimplePlot(
            name="test_plot",
            title="test_title",
            x_axis="perf_throughput",
            y_axis="perf_latency_p99",
        )

        self.matplotlib_mock.assert_called_subplots()

    def test_add_measurement(self):
        plot = SimplePlot(
            name="test_plot",
            title="test_title",
            x_axis="perf_throughput",
            y_axis="perf_latency_p99",
        )

        gpu_metric_values = {0: {"gpu_used_memory": 5000, "gpu_utilization": 50}}

        non_gpu_metric_values = {"perf_throughput": 200, "perf_latency_p99": 8000}

        objective_spec = {"perf_throughput": 10, "perf_latency_p99": 5}

        measurement = construct_run_config_measurement(
            model_name="test_model",
            model_config_names=["test_model_config_0"],
            model_specific_pa_params=MagicMock(),
            gpu_metric_values=gpu_metric_values,
            non_gpu_metric_values=[non_gpu_metric_values],
            metric_objectives=[objective_spec],
        )

        # Add above measurement
        plot.add_run_config_measurement("test_model_label1", measurement)
        self.assertDictEqual(
            plot.data(), {"test_model_label1": {"x_data": [200], "y_data": [8000]}}
        )

        # Add measurement again with different label
        plot.add_run_config_measurement("test_model_label2", measurement)
        self.assertDictEqual(
            plot.data(),
            {
                "test_model_label1": {"x_data": [200], "y_data": [8000]},
                "test_model_label2": {"x_data": [200], "y_data": [8000]},
            },
        )

    def test_plot_data(self):
        plot = SimplePlot(
            name="test_plot",
            title="test_title",
            x_axis="perf_throughput",
            y_axis="perf_latency_p99",
        )

        gpu_metric_values = {0: {"gpu_used_memory": 5000, "gpu_utilization": 50}}

        non_gpu_metric_values = {"perf_throughput": 200, "perf_latency_p99": 8000}

        objective_spec = {"perf_throughput": 10, "perf_latency_p99": 5}

        measurement = construct_run_config_measurement(
            model_name="test_model",
            model_config_names=["test_model_config_0"],
            model_specific_pa_params=MagicMock(),
            gpu_metric_values=gpu_metric_values,
            non_gpu_metric_values=[non_gpu_metric_values],
            metric_objectives=[objective_spec],
        )

        plot.add_run_config_measurement("test_model_label", measurement)

        # Call plot and assert args
        plot.plot_data_and_constraints(constraints={})
        self.matplotlib_mock.assert_called_plot_with_args(
            x_data=[200], y_data=[8000], marker="o", label="test_model_label"
        )

    def test_save(self):
        plot = SimplePlot(
            name="test_plot",
            title="test_title",
            x_axis="perf_throughput",
            y_axis="perf_latency_p99",
        )

        plot.save("test_path")
        self.matplotlib_mock.assert_called_save_with_args("test_path/test_plot")

    def tearDown(self):
        self.matplotlib_mock.stop()
        patch.stopall()


if __name__ == "__main__":
    unittest.main()
