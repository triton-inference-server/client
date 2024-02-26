#!/usr/bin/env python3

# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from math import log2
from unittest.mock import MagicMock, patch

from model_analyzer.config.input.config_command_profile import ConfigCommandProfile
from model_analyzer.config.input.config_defaults import (
    DEFAULT_RUN_CONFIG_MAX_CONCURRENCY,
    DEFAULT_RUN_CONFIG_MAX_REQUEST_RATE,
    DEFAULT_RUN_CONFIG_MIN_CONCURRENCY,
    DEFAULT_RUN_CONFIG_MIN_REQUEST_RATE,
)
from model_analyzer.constants import THROUGHPUT_MINIMUM_CONSECUTIVE_PARAMETER_TRIES
from model_analyzer.model_analyzer_exceptions import TritonModelAnalyzerException
from model_analyzer.result.constraint_manager import ConstraintManager
from model_analyzer.result.parameter_search import ParameterSearch
from model_analyzer.result.run_config_measurement import RunConfigMeasurement

from .common import test_result_collector as trc
from .common.test_utils import construct_run_config_measurement, evaluate_mock_config


class TestParameterSearch(trc.TestResultCollector):
    def setUp(self):
        self._min_concurrency_index = int(log2(DEFAULT_RUN_CONFIG_MIN_CONCURRENCY))
        self._max_concurrency_index = int(log2(DEFAULT_RUN_CONFIG_MAX_CONCURRENCY))

        self._expected_concurrencies = [
            2**c
            for c in range(self._min_concurrency_index, self._max_concurrency_index + 1)
        ]
        self._concurrencies = []

        self._min_request_rate_index = int(log2(DEFAULT_RUN_CONFIG_MIN_REQUEST_RATE))
        self._max_request_rate_index = int(log2(DEFAULT_RUN_CONFIG_MAX_REQUEST_RATE))

        self._expected_request_rates = [
            2**rr
            for rr in range(
                self._min_request_rate_index, self._max_request_rate_index + 1
            )
        ]
        self._request_rates = []

    def tearDown(self):
        patch.stopall()

    def test_concurrency_sweep(self):
        """
        Test sweeping concurrency from min to max, when no constraints are present
        and throughput is linearly increasing
        """
        config = self._create_single_model_no_constraints()
        constraint_manager = ConstraintManager(config)
        concurrency_search = ParameterSearch(config)

        for concurrency in concurrency_search.search_parameters():
            self._concurrencies.append(concurrency)

            concurrency_search.add_run_config_measurement(
                run_config_measurement=self._construct_rcm(
                    throughput=100 * concurrency,
                    latency=10,
                    concurrency=concurrency,
                    constraint_manager=constraint_manager,
                )
            )

        self.assertEqual(self._concurrencies, self._expected_concurrencies)

    def test_request_rate_sweep(self):
        """
        Test sweeping request rate from min to max, when no constraints are present
        and throughput is linearly increasing
        """
        config = self._create_single_model_no_constraints()
        constraint_manager = ConstraintManager(config)
        concurrency_search = ParameterSearch(
            config, model_parameters={"request_rate": "True"}
        )

        for request_rate in concurrency_search.search_parameters():
            self._request_rates.append(request_rate)

            concurrency_search.add_run_config_measurement(
                run_config_measurement=self._construct_rcm(
                    throughput=100 * request_rate,
                    latency=10,
                    request_rate=request_rate,
                    constraint_manager=constraint_manager,
                )
            )

        self.assertEqual(self._request_rates, self._expected_request_rates)

    def test_saturating_sweep(self):
        """
        Test sweeping concurrency from min to max, when no constraints are present
        and throughput increases and then saturates
        """
        config = self._create_single_model_no_constraints()
        constraint_manager = ConstraintManager(config)
        concurrency_search = ParameterSearch(config)
        INCREASE_THROUGHPUT_COUNT = 4

        # [100, 200, 400, 800, 1000, 1000,...]
        throughputs = [
            100 * 2**c if c < INCREASE_THROUGHPUT_COUNT else 1000
            for c in range(self._min_concurrency_index, self._max_concurrency_index + 1)
        ]

        for i, concurrency in enumerate(concurrency_search.search_parameters()):
            self._concurrencies.append(concurrency)

            concurrency_search.add_run_config_measurement(
                run_config_measurement=self._construct_rcm(
                    throughput=throughputs[i],
                    latency=10,
                    concurrency=concurrency,
                    constraint_manager=constraint_manager,
                )
            )

        expected_concurrencies = [
            2**c
            for c in range(
                INCREASE_THROUGHPUT_COUNT
                + THROUGHPUT_MINIMUM_CONSECUTIVE_PARAMETER_TRIES
            )
        ]
        self.assertEqual(self._concurrencies, expected_concurrencies)

    def test_sweep_with_constraints_decreasing(self):
        """
        Test sweeping concurrency from min to max, with 95ms latency constraint
        and throughput is linearly increasing - which causes a decreasing binary search
        """
        config = self._create_single_model_with_constraints("95")
        constraint_manager = ConstraintManager(config)
        concurrency_search = ParameterSearch(config)

        self._expected_concurrencies.extend([12, 10, 9])
        latencies = [10 * c for c in self._expected_concurrencies]

        for i, concurrency in enumerate(concurrency_search.search_parameters()):
            self._concurrencies.append(concurrency)

            concurrency_search.add_run_config_measurement(
                run_config_measurement=self._construct_rcm(
                    throughput=100 * concurrency,
                    latency=latencies[i],
                    concurrency=concurrency,
                    constraint_manager=constraint_manager,
                )
            )

        self.assertEqual(self._concurrencies, self._expected_concurrencies)

    def test_sweep_with_constraints_decrease_then_increase(self):
        """
        Test sweeping concurrency from min to max, with 155ms latency constraint
        and throughput is linearly increasing - which causes a decreasing, then increasing binary search
        """
        config = self._create_single_model_with_constraints("155")
        constraint_manager = ConstraintManager(config)
        concurrency_search = ParameterSearch(config)

        self._expected_concurrencies.extend([12, 14, 15])
        latencies = [10 * c for c in self._expected_concurrencies]

        for i, concurrency in enumerate(concurrency_search.search_parameters()):
            self._concurrencies.append(concurrency)

            concurrency_search.add_run_config_measurement(
                run_config_measurement=self._construct_rcm(
                    throughput=100 * concurrency,
                    latency=latencies[i],
                    concurrency=concurrency,
                    constraint_manager=constraint_manager,
                )
            )

        self.assertEqual(self._concurrencies, self._expected_concurrencies)

    def test_sweep_with_multiple_violation_areas(self):
        """
        Test sweeping concurrency from min to max, with 155ms latency constraint
        with violations in two separate locations
        """
        config = self._create_single_model_with_constraints("155")
        constraint_manager = ConstraintManager(config)
        concurrency_search = ParameterSearch(config)

        self._expected_concurrencies.extend([12, 14, 15])
        latencies = [10 * c for c in self._expected_concurrencies]
        # this adds an early constraint violation which should be ignored
        latencies[1] = 200

        for i, concurrency in enumerate(concurrency_search.search_parameters()):
            self._concurrencies.append(concurrency)

            concurrency_search.add_run_config_measurement(
                run_config_measurement=self._construct_rcm(
                    throughput=100 * concurrency,
                    latency=latencies[i],
                    concurrency=concurrency,
                    constraint_manager=constraint_manager,
                )
            )

        self.assertEqual(self._concurrencies, self._expected_concurrencies)

    def test_sweep_with_constraints_hitting_limit(self):
        """
        Test sweeping concurrency from min to max, with 970ms latency constraint
        and throughput matches concurrency, this will cause BCS to
        quit after DEFAULT_RUN_CONFIG_MAX_BINARY_SEARCH_STEPS (5) attempts
        """
        config = self._create_single_model_with_constraints("970")
        constraint_manager = ConstraintManager(config)
        concurrency_search = ParameterSearch(config)

        self._expected_concurrencies.extend([768, 896, 960, 992, 976])
        latencies = self._expected_concurrencies

        for i, concurrency in enumerate(concurrency_search.search_parameters()):
            self._concurrencies.append(concurrency)

            concurrency_search.add_run_config_measurement(
                run_config_measurement=self._construct_rcm(
                    throughput=100 * concurrency,
                    latency=latencies[i],
                    concurrency=concurrency,
                    constraint_manager=constraint_manager,
                )
            )

        self.assertEqual(self._concurrencies, self._expected_concurrencies)

    def test_not_adding_measurements(self):
        """
        Test that an exception is raised if measurements are not added
        """
        config = self._create_single_model_no_constraints()
        constraint_manager = ConstraintManager(config)
        concurrency_search = ParameterSearch(config)

        with self.assertRaises(TritonModelAnalyzerException):
            for concurrency in concurrency_search.search_parameters():
                self._concurrencies.append(concurrency)

                if concurrency < 32:
                    concurrency_search.add_run_config_measurement(
                        run_config_measurement=self._construct_rcm(
                            throughput=100 * concurrency,
                            latency=10,
                            concurrency=concurrency,
                            constraint_manager=constraint_manager,
                        )
                    )

    def _create_single_model_no_constraints(self):
        args = ["model-analyzer", "profile", "--profile-models", "test_model"]
        yaml_str = ""
        config = evaluate_mock_config(args, yaml_str)

        return config

    def _create_single_model_with_constraints(
        self, latency_budget: str
    ) -> ConfigCommandProfile:
        args = [
            "model-analyzer",
            "profile",
            "--profile-models",
            "test_model",
            "--latency-budget",
            latency_budget,
        ]
        yaml_str = ""
        config = evaluate_mock_config(args, yaml_str)

        return config

    def _construct_rcm(
        self,
        throughput: int,
        latency: int,
        constraint_manager: ConstraintManager,
        concurrency: int = 0,
        request_rate: int = 0,
    ) -> RunConfigMeasurement:
        if concurrency:
            self.model_specific_pa_params = [
                {"batch_size": 1, "concurrency": concurrency}
            ]
        else:
            self.model_specific_pa_params = [
                {"batch_size": 1, "request_rate": request_rate}
            ]

        self.rcm0_non_gpu_metric_values = [
            {
                "perf_throughput": throughput,
                "perf_latency_p99": latency,
                "cpu_used_ram": 1000,
            }
        ]

        return construct_run_config_measurement(
            model_name="test_model",
            model_config_names=["test_model_config_0"],
            model_specific_pa_params=self.model_specific_pa_params,
            gpu_metric_values=MagicMock(),
            non_gpu_metric_values=self.rcm0_non_gpu_metric_values,
            constraint_manager=constraint_manager,
        )


if __name__ == "__main__":
    unittest.main()
