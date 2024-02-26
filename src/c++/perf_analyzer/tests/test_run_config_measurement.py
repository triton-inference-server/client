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
from unittest.mock import MagicMock, patch

from model_analyzer.result.run_config_measurement import RunConfigMeasurement
from tests.common.test_utils import (
    construct_constraint_manager,
    construct_run_config_measurement,
    convert_avg_gpu_metrics_to_data,
    convert_gpu_metrics_to_data,
    convert_non_gpu_metrics_to_data,
    default_encode,
)

from .common import test_result_collector as trc


class TestRunConfigMeasurement(trc.TestResultCollector):
    def setUp(self):
        self._construct_rcm0()
        self._construct_rcm1()
        self._construct_rcm2()
        self._construct_rcm3()
        self._construct_rcm4()
        self._construct_rcm5()

    def tearDown(self):
        patch.stopall()

    def test_model_variants_name(self):
        """
        Test that the model_variants_name was initialized correctly
        """
        self.assertEqual(self.rcm0.model_variants_name(), self.model_variants_name)

    def test_gpu_data(self):
        """
        Test that the gpu data is correct
        """
        self.assertEqual(
            self.rcm0.gpu_data(), convert_gpu_metrics_to_data(self.gpu_metric_values)
        )

    def test_non_gpu_data(self):
        """
        Test that the non-gpu data is correct
        """
        self.assertEqual(
            self.rcm0.non_gpu_data(),
            [
                convert_non_gpu_metrics_to_data(ngvm)
                for ngvm in self.rcm0_non_gpu_metric_values
            ],
        )

    def test_data(self):
        """
        Test that the gpu + non-gpu data is correct
        """
        avg_gpu_data = convert_avg_gpu_metrics_to_data(self.avg_gpu_metric_values)

        data = [
            list(avg_gpu_data.values()) + convert_non_gpu_metrics_to_data(ngvm)
            for ngvm in self.rcm0_non_gpu_metric_values
        ]

        self.assertEqual(list(self.rcm0.data().values()), data)

    def test_gpus_used(self):
        """
        Test that the list of gpus used is correct
        """
        self.assertEqual(self.rcm0.gpus_used(), ["0", "1"])

    def test_get_gpu_metric(self):
        """
        Test that the gpu metric data is correct
        """
        avg_gpu_data = convert_avg_gpu_metrics_to_data(self.avg_gpu_metric_values)

        self.assertEqual(
            self.rcm0.get_gpu_metric("gpu_used_memory"), avg_gpu_data["gpu_used_memory"]
        )

        self.assertEqual(
            self.rcm0.get_gpu_metric("gpu_utilization"), avg_gpu_data["gpu_utilization"]
        )

    def test_gpu_gpu_metric_not_found(self):
        """
        Test that an incorrect metric search returns None
        """
        self.assertEqual(self.rcm0.get_gpu_metric("XXXXX"), None)

    def test_get_non_gpu_metric(self):
        """
        Test that the non-gpu metric data is correct
        """
        non_gpu_data = [
            convert_non_gpu_metrics_to_data(non_gpu_metric_value)
            for non_gpu_metric_value in self.rcm0_non_gpu_metric_values
        ]

        self.assertEqual(
            self.rcm0.get_non_gpu_metric("perf_throughput"),
            [non_gpu_data[0][0], non_gpu_data[1][0]],
        )
        self.assertEqual(
            self.rcm0.get_non_gpu_metric("perf_latency_p99"),
            [non_gpu_data[0][1], non_gpu_data[1][1]],
        )
        self.assertEqual(
            self.rcm0.get_non_gpu_metric("cpu_used_ram"),
            [non_gpu_data[0][2], non_gpu_data[1][2]],
        )

    def test_get_weighted_non_gpu_metric(self):
        """
        Test that the weighted non-gpu metric data is correct
        """
        non_gpu_data = [
            convert_non_gpu_metrics_to_data(weighted_non_gpu_metric_value)
            for weighted_non_gpu_metric_value in self.rcm0_weighted_non_gpu_metric_values
        ]

        self.assertEqual(
            self.rcm0.get_weighted_non_gpu_metric("perf_throughput"),
            [non_gpu_data[0][0], non_gpu_data[1][0]],
        )
        self.assertEqual(
            self.rcm0.get_weighted_non_gpu_metric("perf_latency_p99"),
            [non_gpu_data[0][1], non_gpu_data[1][1]],
        )
        self.assertEqual(
            self.rcm0.get_weighted_non_gpu_metric("cpu_used_ram"),
            [non_gpu_data[0][2], non_gpu_data[1][2]],
        )

    def test_non_gpu_get_metric_value(self):
        """
        Test that the non-gpu metric value is correct
        """
        self.assertEqual(
            self.rcm0.get_non_gpu_metric_value("perf_throughput"),
            sum(
                [
                    self.rcm0_non_gpu_metric_values[0]["perf_throughput"],
                    self.rcm0_non_gpu_metric_values[1]["perf_throughput"],
                ]
            ),
        )

    def test_gpu_get_metric_value(self):
        """
        Test that the gpu metric value is correct
        """
        self.assertEqual(
            self.rcm0.get_gpu_metric_value("gpu_used_memory"),
            self.avg_gpu_metric_values["gpu_used_memory"],
        )

    def test_get_weighted_non_gpu_metric_value(self):
        """
        Test that the non-gpu weighted metric value is correct
        """
        sum_weighted_metric_value = (
            (self.rcm0_non_gpu_metric_values[0]["perf_latency_p99"] * self.weights[0])
            + (self.rcm0_non_gpu_metric_values[1]["perf_latency_p99"] * self.weights[1])
        ) / sum(self.weights)

        mean_weighted_metric_value = sum_weighted_metric_value / len(
            self.rcm0_weighted_non_gpu_metric_values
        )

        self.assertEqual(
            self.rcm0.get_weighted_non_gpu_metric_value("perf_latency_p99"),
            mean_weighted_metric_value,
        )

    def test_model_specific_pa_params(self):
        """
        Test that the model specific PA params are correct
        """
        self.assertEqual(
            self.rcm0.model_specific_pa_params(), self.model_specific_pa_params
        )

    def test_is_better_than(self):
        """
        Test to ensure measurement comparison is working as intended
        """
        # RCM0: 1000, 40    RCM1: 500, 30  weights:[1,3]
        # RCM0-A's throughput is better than RCM1-A (0.5)
        # RCM0-B's latency is worse than RCM1-B (-0.25)
        # Factoring in model config weighting
        # tips this is favor of RCM1 (0.125, -0.1875)
        self.assertFalse(self.rcm0.is_better_than(self.rcm1))

        # This tips the scale in the favor of RCM0 (0.2, -0.15)
        self.rcm0.set_model_config_weighting([2, 3])
        self.assertTrue(self.rcm0.is_better_than(self.rcm1))
        self.assertGreater(self.rcm0, self.rcm1)

    def test_is_better_than_consistency(self):
        """
        Test to ensure measurement comparison is working correctly
        when the percentage gain is very close
        """
        # Throughputs
        #   RCM2{A/B}: {336,223}   RCM3{A/B}: {270,272}
        #   RCM2-A is 21.8% better than RCM3-A
        #   RMC2-B is 19.8% worse than RCM3-B
        # Therefore, RCM2 is (very slightly) better than RCM3
        self.assertTrue(self.rcm2.is_better_than(self.rcm3))
        self.assertFalse(self.rcm3.is_better_than(self.rcm2))

    def test_compare_measurements(self):
        """
        Test to ensure compare measurement function returns
        the correct magnitude
        # RCM4's throughput is 1000
        # RCM5's throughput is 2000
        # Therefore, the magnitude is (RCM5 - RCM4) / avg throughput
        #                             (2000 - 1000) / 1500
        """
        rcm4_vs_rcm5 = self.rcm4.compare_measurements(self.rcm5)
        self.assertEqual(rcm4_vs_rcm5, 1000 / 1500)

        rcm5_vs_rcm4 = self.rcm5.compare_measurements(self.rcm4)
        self.assertEqual(rcm5_vs_rcm4, -1000 / 1500)

    def test_is_passing_constraints_none(self):
        """
        Test to ensure constraints are reported as passing
        if none were specified
        """
        self.rcm5.set_constraint_manager(
            construct_constraint_manager(
                """
            profile_models:
              modelA
            """
            )
        )
        self.assertTrue(self.rcm5.is_passing_constraints())

    def test_is_passing_constraints(self):
        """
        Test to ensure constraints are reported as
        passing/failing if model is above/below
        throughput threshold
        """
        constraint_manager = construct_constraint_manager(
            """
            profile_models:
              modelA:
                constraints:
                  perf_throughput:
                    min: 500
            """
        )
        self.rcm5.set_constraint_manager(constraint_manager)

        self.assertTrue(self.rcm5.is_passing_constraints())

        constraint_manager = construct_constraint_manager(
            """
            profile_models:
              modelA:
                constraints:
                  perf_throughput:
                    min: 3000
            """
        )
        self.rcm5.set_constraint_manager(constraint_manager)

        self.assertFalse(self.rcm5.is_passing_constraints())

    def test_compare_constraints_none(self):
        """
        Checks case where either self or other is passing constraints
        """
        # RCM4's throughput is 1000
        # RCM5's throughput is 2000
        constraint_manager = construct_constraint_manager(
            """
            profile_models:
              modelA:
                constraints:
                  perf_throughput:
                    min: 500
            """
        )
        self.rcm4.set_constraint_manager(constraint_manager)

        constraint_manager = construct_constraint_manager(
            """
            profile_models:
              modelA:
                constraints:
                  perf_throughput:
                    min: 2500
            """
        )
        self.rcm5.set_constraint_manager(constraint_manager)

        self.assertEqual(self.rcm4.compare_constraints(self.rcm5), None)
        self.assertEqual(self.rcm5.compare_constraints(self.rcm4), None)

    def test_compare_constraints_equal(self):
        """
        Test to ensure compare constraints reports zero when both
        RCMs are missing constraints by the same amount
        """
        # RCM4's throughput is 1000
        # RCM5's throughput is 2000
        constraint_manager = construct_constraint_manager(
            """
            profile_models:
              modelA:
                constraints:
                  perf_throughput:
                    min: 1250
            """
        )
        self.rcm4.set_constraint_manager(constraint_manager)

        constraint_manager = construct_constraint_manager(
            """
            profile_models:
              modelA:
                constraints:
                  perf_throughput:
                    min: 2500
            """
        )
        self.rcm5.set_constraint_manager(constraint_manager)

        # RCM4 is failing by 20%, RCM5 is failing by 20%
        self.assertEqual(self.rcm4.compare_constraints(self.rcm5), 0)
        self.assertEqual(self.rcm5.compare_constraints(self.rcm4), 0)

    def test_compare_constraints_unequal(self):
        """
        Test to ensure compare constraints reports the correct
        value when the RCMs are both failing constraints by different
        amounts
        """
        # RCM4's throughput is 1000
        # RCM5's throughput is 2000
        constraint_manager = construct_constraint_manager(
            """
            profile_models:
              modelA:
                constraints:
                  perf_throughput:
                    min: 2000
            """
        )
        self.rcm4.set_constraint_manager(constraint_manager)

        constraint_manager = construct_constraint_manager(
            """
            profile_models:
              modelA:
                constraints:
                  perf_throughput:
                    min: 2500
            """
        )
        self.rcm5.set_constraint_manager(constraint_manager)

        # RCM4 is failing by 50%, RCM5 is failing by 20%
        self.assertEqual(self.rcm4.compare_constraints(self.rcm5), 0.30)
        self.assertEqual(self.rcm5.compare_constraints(self.rcm4), -0.30)

    def test_calculate_weighted_percentage_gain(self):
        """
        Test to ensure weighted percentage gain is being calculated correctly
        """

        # RCM0: 1000, 40    RCM1: 500, 30  weights:[1,3]
        # RCM0-A's throughput is better than RCM1-A (0.5)
        # RCM0-B's latency is worse than RCM1-B (-0.25)
        # Factoring in model config weighting
        # tips this is favor of RCM1 (0.125, -0.1875)
        # However, by percentage RCM0 will be evaluated as slightly better
        # 100% on throughput, -25% on latency
        # Factoring in weighting, RCM0 is slightly better (100 - 75) / 4 = 6.25%
        self.assertEqual(self.rcm0.calculate_weighted_percentage_gain(self.rcm1), 6.25)

        # Changing the weighting tips the scale in the favor of RCM0 (0.2, -0.15)
        # And, from a percentage standpoint we get: (200 - 75) / 5 = 25%
        self.rcm0.set_model_config_weighting([2, 3])
        self.assertEqual(self.rcm0.calculate_weighted_percentage_gain(self.rcm1), 25.0)

    def test_from_dict(self):
        """
        Test to ensure class can be correctly restored from a dictionary
        """
        rcm0_json = json.dumps(self.rcm0, default=default_encode)

        rcm0_from_dict = RunConfigMeasurement.from_dict(json.loads(rcm0_json))

        self.assertEqual(
            rcm0_from_dict.model_variants_name(), self.rcm0.model_variants_name()
        )
        self.assertEqual(rcm0_from_dict.gpu_data(), self.rcm0.gpu_data())
        self.assertEqual(rcm0_from_dict.non_gpu_data(), self.rcm0.non_gpu_data())
        self.assertEqual(
            list(rcm0_from_dict.data().values()), list(self.rcm0.data().values())
        )
        self.assertEqual(
            rcm0_from_dict._model_config_measurements,
            self.rcm0._model_config_measurements,
        )
        self.assertEqual(rcm0_from_dict._model_config_weights, [])

    def _construct_rcm0(self):
        self.model_name = "modelA,modelB"
        self.model_config_name = ["modelA_config_0", "modelB_config_1"]
        self.model_variants_name = "".join(self.model_config_name)
        self.model_specific_pa_params = [
            {"batch_size": 1, "concurrency": 1},
            {"batch_size": 2, "concurrency": 2},
        ]

        self.gpu_metric_values = {
            "0": {"gpu_used_memory": 6000, "gpu_utilization": 60},
            "1": {"gpu_used_memory": 10000, "gpu_utilization": 20},
        }
        self.avg_gpu_metric_values = {"gpu_used_memory": 8000, "gpu_utilization": 40}

        self.rcm0_non_gpu_metric_values = [
            {
                # modelA_config_0
                "perf_throughput": 1000,
                "perf_latency_p99": 20,
                "cpu_used_ram": 1000,
            },
            {
                # modelB_config_1
                "perf_throughput": 2000,
                "perf_latency_p99": 40,
                "cpu_used_ram": 1500,
            },
        ]

        self.metric_objectives = [{"perf_throughput": 1}, {"perf_latency_p99": 1}]

        self.weights = [1, 3]

        self.rcm0_weighted_non_gpu_metric_values = []
        for index, non_gpu_metric_values in enumerate(self.rcm0_non_gpu_metric_values):
            self.rcm0_weighted_non_gpu_metric_values.append(
                {
                    objective: value * self.weights[index] / sum(self.weights)
                    for (objective, value) in non_gpu_metric_values.items()
                }
            )

        self.rcm0 = construct_run_config_measurement(
            self.model_name,
            self.model_config_name,
            self.model_specific_pa_params,
            self.gpu_metric_values,
            self.rcm0_non_gpu_metric_values,
            MagicMock(),
            self.metric_objectives,
            self.weights,
        )

    def _construct_rcm1(self):
        model_name = "modelA,modelB"
        model_config_name = ["modelA_config_2", "modelB_config_3"]
        model_specific_pa_params = [
            {"batch_size": 3, "concurrency": 3},
            {"batch_size": 4, "concurrency": 4},
        ]

        gpu_metric_values = {
            "0": {"gpu_used_memory": 7000, "gpu_utilization": 40},
            "1": {"gpu_used_memory": 12000, "gpu_utilization": 30},
        }

        self.rcm1_non_gpu_metric_values = [
            {
                # modelA_config_2
                "perf_throughput": 500,
                "perf_latency_p99": 20,
                "cpu_used_ram": 1000,
            },
            {
                # modelB_config_3
                "perf_throughput": 1200,
                "perf_latency_p99": 30,
                "cpu_used_ram": 1500,
            },
        ]

        metric_objectives = [{"perf_throughput": 1}, {"perf_throughput": 1}]

        weights = [1, 3]

        self.rcm1_weighted_non_gpu_metric_values = []
        for index, non_gpu_metric_values in enumerate(self.rcm1_non_gpu_metric_values):
            self.rcm1_weighted_non_gpu_metric_values.append(
                {
                    objective: value * self.weights[index] / sum(weights)
                    for (objective, value) in non_gpu_metric_values.items()
                }
            )

        self.rcm1 = construct_run_config_measurement(
            model_name,
            model_config_name,
            model_specific_pa_params,
            gpu_metric_values,
            self.rcm1_non_gpu_metric_values,
            MagicMock(),
            metric_objectives,
            weights,
        )

    def _construct_rcm2(self):
        model_name = "modelA,modelB"
        model_config_name = ["modelA_config_1", "modelB_config_2"]
        model_specific_pa_params = [
            {"batch_size": 3, "concurrency": 3},
            {"batch_size": 4, "concurrency": 4},
        ]

        gpu_metric_values = {
            "0": {"gpu_used_memory": 7000, "gpu_utilization": 40},
            "1": {"gpu_used_memory": 12000, "gpu_utilization": 30},
        }

        self.rcm2_non_gpu_metric_values = [
            {
                # modelA_config_1
                "perf_throughput": 336,
                "perf_latency_p99": 20,
                "cpu_used_ram": 1000,
            },
            {
                # modelB_config_2
                "perf_throughput": 223,
                "perf_latency_p99": 30,
                "cpu_used_ram": 1500,
            },
        ]

        metric_objectives = [{"perf_throughput": 1}, {"perf_throughput": 1}]

        weights = [1, 1]

        self.rcm2_weighted_non_gpu_metric_values = []
        for index, non_gpu_metric_values in enumerate(self.rcm2_non_gpu_metric_values):
            self.rcm2_weighted_non_gpu_metric_values.append(
                {
                    objective: value * self.weights[index] / sum(weights)
                    for (objective, value) in non_gpu_metric_values.items()
                }
            )

        self.rcm2 = construct_run_config_measurement(
            model_name,
            model_config_name,
            model_specific_pa_params,
            gpu_metric_values,
            self.rcm2_non_gpu_metric_values,
            MagicMock(),
            metric_objectives,
            weights,
        )

    def _construct_rcm3(self):
        model_name = "modelA,modelB"
        model_config_name = ["modelA_config_1", "modelB_config_2"]
        model_specific_pa_params = [
            {"batch_size": 3, "concurrency": 3},
            {"batch_size": 4, "concurrency": 4},
        ]

        gpu_metric_values = {
            "0": {"gpu_used_memory": 7000, "gpu_utilization": 40},
            "1": {"gpu_used_memory": 12000, "gpu_utilization": 30},
        }

        self.rcm3_non_gpu_metric_values = [
            {
                # modelA_config_1
                "perf_throughput": 270,
                "perf_latency_p99": 20,
                "cpu_used_ram": 1000,
            },
            {
                # modelB_config_2
                "perf_throughput": 272,
                "perf_latency_p99": 30,
                "cpu_used_ram": 1500,
            },
        ]

        metric_objectives = [{"perf_throughput": 1}, {"perf_throughput": 1}]

        weights = [1, 1]

        self.rcm3_weighted_non_gpu_metric_values = []
        for index, non_gpu_metric_values in enumerate(self.rcm3_non_gpu_metric_values):
            self.rcm3_weighted_non_gpu_metric_values.append(
                {
                    objective: value * self.weights[index] / sum(weights)
                    for (objective, value) in non_gpu_metric_values.items()
                }
            )

        self.rcm3 = construct_run_config_measurement(
            model_name,
            model_config_name,
            model_specific_pa_params,
            gpu_metric_values,
            self.rcm3_non_gpu_metric_values,
            MagicMock(),
            metric_objectives,
            weights,
        )

    def _construct_rcm4(self):
        model_name = "modelA"
        model_config_name = ["modelA_config_0"]
        model_specific_pa_params = [
            {"batch_size": 1, "concurrency": 1},
            {"batch_size": 2, "concurrency": 2},
        ]

        gpu_metric_values = {
            "0": {"gpu_used_memory": 6000, "gpu_utilization": 60},
            "1": {"gpu_used_memory": 10000, "gpu_utilization": 20},
        }

        self.rcm4_non_gpu_metric_values = [
            {
                # modelA_config_0
                "perf_throughput": 1000,
                "perf_latency_p99": 20,
                "cpu_used_ram": 1000,
            },
        ]

        metric_objectives = [{"perf_throughput": 1}]

        weights = [1]

        self.rcm4_weighted_non_gpu_metric_values = []
        for index, non_gpu_metric_values in enumerate(self.rcm4_non_gpu_metric_values):
            self.rcm4_weighted_non_gpu_metric_values.append(
                {
                    objective: value * self.weights[index] / sum(self.weights)
                    for (objective, value) in non_gpu_metric_values.items()
                }
            )

        self.rcm4 = construct_run_config_measurement(
            model_name,
            model_config_name,
            model_specific_pa_params,
            gpu_metric_values,
            self.rcm4_non_gpu_metric_values,
            MagicMock(),
            metric_objectives,
            weights,
        )

    def _construct_rcm5(self):
        model_name = "modelA"
        model_config_name = ["modelA_config_0"]
        model_specific_pa_params = [
            {"batch_size": 1, "concurrency": 1},
            {"batch_size": 2, "concurrency": 2},
        ]

        gpu_metric_values = {
            "0": {"gpu_used_memory": 6000, "gpu_utilization": 60},
            "1": {"gpu_used_memory": 10000, "gpu_utilization": 20},
        }

        self.rcm5_non_gpu_metric_values = [
            {
                # modelA_config_0
                "perf_throughput": 2000,
                "perf_latency_p99": 20,
                "cpu_used_ram": 1000,
            },
        ]

        metric_objectives = [{"perf_throughput": 1}]

        weights = [1]

        self.rcm5_weighted_non_gpu_metric_values = []
        for index, non_gpu_metric_values in enumerate(self.rcm5_non_gpu_metric_values):
            self.rcm5_weighted_non_gpu_metric_values.append(
                {
                    objective: value * self.weights[index] / sum(self.weights)
                    for (objective, value) in non_gpu_metric_values.items()
                }
            )

        self.rcm5 = construct_run_config_measurement(
            model_name,
            model_config_name,
            model_specific_pa_params,
            gpu_metric_values,
            self.rcm5_non_gpu_metric_values,
            MagicMock(),
            metric_objectives,
            weights,
        )


if __name__ == "__main__":
    unittest.main()
