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
from random import sample
from unittest.mock import MagicMock, patch

import yaml

from model_analyzer.result.run_config_result_comparator import RunConfigResultComparator
from model_analyzer.result.sorted_results import SortedResults

from .common import test_result_collector as trc
from .common.test_utils import (
    construct_constraint_manager,
    construct_run_config,
    construct_run_config_result,
)


class TestSortedResults(trc.TestResultCollector):
    def setUp(self):
        objective_spec = {"perf_throughput": 10, "perf_latency_p99": 5}
        self.sorted_results = SortedResults()
        self.result_comparator = RunConfigResultComparator(
            metric_objectives_list=[objective_spec], model_weights=[1]
        )

    def tearDown(self):
        patch.stopall()

    def test_add_results(self):
        avg_gpu_metrics = {0: {"gpu_used_memory": 6000, "gpu_utilization": 60}}
        avg_non_gpu_metrics = {"perf_throughput": 100, "perf_latency_p99": 4000}

        for _ in range(10):
            self.sorted_results.add_result(
                construct_run_config_result(
                    avg_gpu_metric_values=avg_gpu_metrics,
                    avg_non_gpu_metric_values_list=[avg_non_gpu_metrics],
                    constraint_manager=MagicMock(),
                    comparator=self.result_comparator,
                )
            )

        results = self.sorted_results.results()
        self.assertEqual(len(results), 10)

    def test_add_results_same_config(self):
        """
        Test that additional results with the same model config variant
        will be added to the same entry in the list
        """
        avg_gpu_metrics = {0: {"gpu_used_memory": 6000, "gpu_utilization": 60}}
        avg_non_gpu_metrics = {"perf_throughput": 100, "perf_latency_p99": 4000}

        constraint_manager = construct_constraint_manager(
            """
            profile_models:
              test_model
            """
        )
        for _ in range(10):
            run_config = construct_run_config("modelA", "model_config_0", "key_A")
            self.sorted_results.add_result(
                construct_run_config_result(
                    avg_gpu_metric_values=avg_gpu_metrics,
                    avg_non_gpu_metric_values_list=[avg_non_gpu_metrics],
                    comparator=self.result_comparator,
                    constraint_manager=constraint_manager,
                    run_config=run_config,
                )
            )

        results = self.sorted_results.results()
        self.assertEqual(len(results), 1)

    def test_add_results_dynamic_sorting(self):
        """
        Tests the case where configs A & B where A is better than B
        Then add a new measurement to B, which now makes B better than A
        """
        avg_gpu_metrics = {0: {"gpu_used_memory": 6000, "gpu_utilization": 60}}

        run_config_A = construct_run_config("model", "model_config_A", "key_A")
        run_config_B = construct_run_config("model", "model_config_B", "key_B")
        run_config_list = [run_config_A, run_config_B]

        constraint_manager = construct_constraint_manager(
            """
            profile_models:
              model
            """
        )

        for i in sample(range(2), 2):
            avg_non_gpu_metrics = {
                "perf_throughput": 100 - 10 * i,
                "perf_latency_p99": 4000,
            }
            self.sorted_results.add_result(
                construct_run_config_result(
                    avg_gpu_metric_values=avg_gpu_metrics,
                    avg_non_gpu_metric_values_list=[avg_non_gpu_metrics],
                    comparator=self.result_comparator,
                    model_name="model",
                    model_config_names=["model_config_0"],
                    constraint_manager=constraint_manager,
                    run_config=run_config_list[i],
                )
            )

        all_results = self.sorted_results.top_n_results(n=SortedResults.GET_ALL_RESULTS)

        self.assertEqual(
            all_results[0].run_config().model_variants_name(), "model_config_A"
        )
        self.assertEqual(
            all_results[1].run_config().model_variants_name(), "model_config_B"
        )

        avg_non_gpu_metrics = {"perf_throughput": 200, "perf_latency_p99": 4000}

        self.sorted_results.add_result(
            construct_run_config_result(
                avg_gpu_metric_values=avg_gpu_metrics,
                avg_non_gpu_metric_values_list=[avg_non_gpu_metrics],
                comparator=self.result_comparator,
                model_name="model",
                model_config_names=["model_config_0"],
                constraint_manager=constraint_manager,
                run_config=run_config_B,
            )
        )

        all_results = self.sorted_results.top_n_results(n=SortedResults.GET_ALL_RESULTS)

        self.assertEqual(
            all_results[0].run_config().model_variants_name(), "model_config_B"
        )
        self.assertEqual(
            all_results[1].run_config().model_variants_name(), "model_config_A"
        )

    def test_add_results_all_failing(self):
        """
        Test the case where we have only failing results
        """
        # Create constraint_manager for all model names
        constraint_manager = construct_constraint_manager(
            yaml.dump(
                {
                    "profile_models": {
                        str(i): {"constraints": {"perf_throughput": {"min": 1000}}}
                        for i in range(10)
                    }
                }
            )
        )

        avg_gpu_metrics = {0: {"gpu_used_memory": 6000, "gpu_utilization": 60}}
        for i in sample(range(10), 10):
            model_name = str(i)
            model_config_names = [f"{model_name}_config_0"]

            avg_non_gpu_metrics = {
                "perf_throughput": 100 + 10 * i,
                "perf_latency_p99": 4000,
            }
            self.sorted_results.add_result(
                construct_run_config_result(
                    avg_gpu_metric_values=avg_gpu_metrics,
                    avg_non_gpu_metric_values_list=[avg_non_gpu_metrics],
                    comparator=self.result_comparator,
                    constraint_manager=constraint_manager,
                    model_config_names=model_config_names,
                    model_name=model_name,
                )
            )

        top_5_results = self.sorted_results.top_n_results(n=5)
        self.assertEqual(top_5_results[0].model_name(), "9")
        self.assertEqual(top_5_results[1].model_name(), "8")
        self.assertEqual(top_5_results[2].model_name(), "7")
        self.assertEqual(top_5_results[3].model_name(), "6")
        self.assertEqual(top_5_results[4].model_name(), "5")

    def test_add_results_failing_to_passing(self):
        """
        Test the case where we have only failing results
        and then a measurement makes one of the results passing
        """

        # Create constraint_manager for all model names
        constraint_manager = construct_constraint_manager(
            yaml.dump(
                {
                    "profile_models": {
                        str(i): {"constraints": {"perf_throughput": {"min": 1000}}}
                        for i in sample(range(10), 10)
                    }
                }
            )
        )

        # Create 10 failing results
        avg_gpu_metrics = {0: {"gpu_used_memory": 6000, "gpu_utilization": 60}}
        for i in sample(range(10), 10):
            model_name = str(i)
            model_config_names = [f"{model_name}_config_0"]

            avg_non_gpu_metrics = {
                "perf_throughput": 100 + 10 * i,
                "perf_latency_p99": 4000,
            }
            self.sorted_results.add_result(
                construct_run_config_result(
                    avg_gpu_metric_values=avg_gpu_metrics,
                    avg_non_gpu_metric_values_list=[avg_non_gpu_metrics],
                    comparator=self.result_comparator,
                    constraint_manager=constraint_manager,
                    model_config_names=model_config_names,
                    model_name=model_name,
                )
            )

        # Now add a measurement to the last result so that it now passes the constraint
        avg_non_gpu_metrics = {"perf_throughput": 2000, "perf_latency_p99": 4000}

        model_name = "9"
        model_config_names = [f"{model_name}_config_0"]
        self.sorted_results.add_result(
            construct_run_config_result(
                avg_gpu_metric_values=avg_gpu_metrics,
                avg_non_gpu_metric_values_list=[avg_non_gpu_metrics],
                comparator=self.result_comparator,
                constraint_manager=constraint_manager,
                model_config_names=model_config_names,
                model_name=model_name,
            )
        )

        top_5_results = self.sorted_results.top_n_results(n=5)
        self.assertEqual(len(top_5_results), 1)
        self.assertEqual(top_5_results[0].model_name(), "9")

    def test_top_n_results(self):
        avg_gpu_metrics = {0: {"gpu_used_memory": 6000, "gpu_utilization": 60}}
        constraint_manager = construct_constraint_manager(
            yaml.dump({"profile_models": {str(i): {} for i in range(10)}})
        )

        for i in sample(range(10), 10):
            model_name = str(i)
            model_config_names = [f"{model_name}_config_0"]

            avg_non_gpu_metrics = {
                "perf_throughput": 100 + 10 * i,
                "perf_latency_p99": 4000,
            }
            self.sorted_results.add_result(
                construct_run_config_result(
                    avg_gpu_metric_values=avg_gpu_metrics,
                    avg_non_gpu_metric_values_list=[avg_non_gpu_metrics],
                    comparator=self.result_comparator,
                    model_name=model_name,
                    model_config_names=model_config_names,
                    constraint_manager=constraint_manager,
                )
            )

        top_5_results = self.sorted_results.top_n_results(n=5)
        self.assertEqual(top_5_results[0].model_name(), "9")
        self.assertEqual(top_5_results[1].model_name(), "8")
        self.assertEqual(top_5_results[2].model_name(), "7")
        self.assertEqual(top_5_results[3].model_name(), "6")
        self.assertEqual(top_5_results[4].model_name(), "5")

        all_results = self.sorted_results.top_n_results(n=SortedResults.GET_ALL_RESULTS)
        self.assertEqual(all_results[0].model_name(), "9")
        self.assertEqual(all_results[1].model_name(), "8")
        self.assertEqual(all_results[2].model_name(), "7")
        self.assertEqual(all_results[3].model_name(), "6")
        self.assertEqual(all_results[4].model_name(), "5")
        self.assertEqual(all_results[5].model_name(), "4")
        self.assertEqual(all_results[6].model_name(), "3")
        self.assertEqual(all_results[7].model_name(), "2")
        self.assertEqual(all_results[8].model_name(), "1")
        self.assertEqual(all_results[9].model_name(), "0")


if __name__ == "__main__":
    unittest.main()
