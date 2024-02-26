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

from model_analyzer.constants import GLOBAL_CONSTRAINTS_KEY
from model_analyzer.result.constraint_manager import ConstraintManager
from model_analyzer.result.model_constraints import ModelConstraints

from .common import test_result_collector as trc
from .common.test_utils import construct_run_config_measurement, evaluate_mock_config


class TestConstraintManager(trc.TestResultCollector):
    def tearDown(self):
        patch.stopall()

    def test_single_model_no_constraints(self):
        """
        Test that constraints are empty
        """
        config = self._create_single_model_no_constraints()
        constraint_manager = ConstraintManager(config)
        constraints = constraint_manager.get_constraints_for_all_models()

        self.assertEqual(constraints["model_A"], ModelConstraints({}))
        self.assertEqual(constraints[GLOBAL_CONSTRAINTS_KEY], ModelConstraints({}))

    def test_single_model_with_constraints(self):
        """
        Test that model specific constraints are set
        """
        config = self._create_single_model_with_constraints()
        constraint_manager = ConstraintManager(config)
        constraints = constraint_manager.get_constraints_for_all_models()

        self.assertEqual(
            constraints["model_A"], ModelConstraints({"perf_latency_p99": {"max": 100}})
        )
        self.assertEqual(constraints[GLOBAL_CONSTRAINTS_KEY], ModelConstraints({}))

    def test_single_model_with_global_constraints(self):
        """
        Test that global constraints are attributed to a model
        """
        config = self._create_single_model_global_constraints()
        constraint_manager = ConstraintManager(config)
        constraints = constraint_manager.get_constraints_for_all_models()

        self.assertEqual(
            constraints["model_A"], ModelConstraints({"perf_throughput": {"min": 100}})
        )
        self.assertEqual(
            constraints[GLOBAL_CONSTRAINTS_KEY],
            ModelConstraints({"perf_throughput": {"min": 100}}),
        )

    def test_single_model_with_both_constraints(self):
        """
        Test that model specific constraints override global
        """
        config = self._create_single_model_both_constraints()
        constraint_manager = ConstraintManager(config)
        constraints = constraint_manager.get_constraints_for_all_models()

        self.assertEqual(
            constraints["model_A"], ModelConstraints({"perf_latency_p99": {"max": 50}})
        )
        self.assertEqual(
            constraints[GLOBAL_CONSTRAINTS_KEY],
            ModelConstraints({"perf_latency_p99": {"max": 100}}),
        )

    def test_multi_model_with_no_global_constraints(self):
        """
        Test multi-model with only model constraints and no global constraints
        """
        config = self._create_multi_model_with_no_global_constraints()
        constraint_manager = ConstraintManager(config)
        constraints = constraint_manager.get_constraints_for_all_models()

        self.assertEqual(
            constraints["model_A"], ModelConstraints({"perf_latency_p99": {"max": 50}})
        )
        self.assertEqual(
            constraints["model_B"], ModelConstraints({"perf_throughput": {"min": 100}})
        )

    def test_multi_model_with_matching_global_constraints(self):
        """
        Test multi-model with Global constraints and individual overrides
        """
        config = self._create_multi_model_with_matching_global_constraints()
        constraint_manager = ConstraintManager(config)
        constraints = constraint_manager.get_constraints_for_all_models()

        self.assertEqual(
            constraints["model_A"], ModelConstraints({"perf_latency_p99": {"max": 75}})
        )
        self.assertEqual(
            constraints["model_B"], ModelConstraints({"perf_throughput": {"min": 125}})
        )
        self.assertEqual(
            constraints[GLOBAL_CONSTRAINTS_KEY],
            ModelConstraints(
                {"perf_throughput": {"min": 100}, "gpu_used_memory": {"max": 1000}}
            ),
        )

    def test_multi_model_with_different_global_constrants(self):
        """
        Test multi-model with different global and individual model constraints
        """
        config = self._create_multi_model_with_different_global_constrants()
        constraint_manager = ConstraintManager(config)
        constraints = constraint_manager.get_constraints_for_all_models()

        self.assertEqual(
            constraints["model_A"], ModelConstraints({"perf_latency_p99": {"max": 100}})
        )
        self.assertEqual(
            constraints["model_B"], ModelConstraints({"perf_throughput": {"min": 150}})
        )
        self.assertEqual(
            constraints[GLOBAL_CONSTRAINTS_KEY],
            ModelConstraints({"gpu_used_memory": {"max": 2000}}),
        )

    def test_single_model_max_constraint_checks(self):
        """
        Test that satisfies_constraints works for a single model
        with a max style constraint
        """
        config = self._create_single_model_with_constraints()
        constraint_manager = ConstraintManager(config)

        # Constraint is P99 Latency max of 100
        rcm = self._construct_rcm({"perf_latency_p99": 101}, constraint_manager)
        self.assertFalse(constraint_manager.satisfies_constraints(rcm))

        rcm = self._construct_rcm({"perf_latency_p99": 100}, constraint_manager)
        self.assertTrue(constraint_manager.satisfies_constraints(rcm))

        rcm = self._construct_rcm({"perf_latency_p99": 99}, constraint_manager)
        self.assertTrue(constraint_manager.satisfies_constraints(rcm))

    def test_single_model_min_constraint_checks(self):
        """
        Test that satisfies_constraints works for a single model
        with a min style constraint
        """
        config = self._create_single_model_global_constraints()
        constraint_manager = ConstraintManager(config)

        # Constraint is throughput min of 100
        rcm = self._construct_rcm({"perf_throughput": 101}, constraint_manager)
        self.assertTrue(constraint_manager.satisfies_constraints(rcm))

        rcm = self._construct_rcm({"perf_throughput": 100}, constraint_manager)
        self.assertTrue(constraint_manager.satisfies_constraints(rcm))

        rcm = self._construct_rcm({"perf_throughput": 99}, constraint_manager)
        self.assertFalse(constraint_manager.satisfies_constraints(rcm))

    def test_multi_model_max_constraint_checks(self):
        """
        Test that satisfies_constraints works for multi-model
        with a max style constraints
        """
        # Constraints are:
        #  Model A: P99 Latency max of 50
        #  Model B: Throughput min of 100
        constraint_manager = ConstraintManager(
            config=self._create_multi_model_with_no_global_constraints()
        )

        # Model A & B are both at boundaries
        rcm = self._construct_mm_rcm(
            [
                {"perf_latency_p99": 50, "perf_throughput": 0},
                {"perf_latency_p99": 0, "perf_throughput": 100},
            ],
            constraint_manager,
        )
        self.assertTrue(constraint_manager.satisfies_constraints(rcm))

        # Model A exceeds latency
        rcm = self._construct_mm_rcm(
            [
                {"perf_latency_p99": 51, "perf_throughput": 0},
                {"perf_latency_p99": 0, "perf_throughput": 100},
            ],
            constraint_manager,
        )
        self.assertFalse(constraint_manager.satisfies_constraints(rcm))

        # Constraints are:
        #  Model A: P99 Latency max of 75
        #  Model B: Throughput min of 125
        constraint_manager = ConstraintManager(
            config=self._create_multi_model_with_matching_global_constraints()
        )

        # Model A & B are both at boundaries
        rcm = self._construct_mm_rcm(
            [
                {"perf_latency_p99": 75, "perf_throughput": 0},
                {"perf_latency_p99": 0, "perf_throughput": 125},
            ],
            constraint_manager,
        )
        self.assertTrue(constraint_manager.satisfies_constraints(rcm))

        # Model A exceeds latency
        rcm = self._construct_mm_rcm(
            [
                {"perf_latency_p99": 78, "perf_throughput": 0},
                {"perf_latency_p99": 0, "perf_throughput": 125},
            ],
            constraint_manager,
        )
        self.assertFalse(constraint_manager.satisfies_constraints(rcm))

        # Constraints are:
        #  Model A: P99 Latency max of 100
        #  Model B: Throughput min of 150
        constraint_manager = ConstraintManager(
            config=self._create_multi_model_with_different_global_constrants()
        )

        # Model A & B are both at boundaries
        rcm = self._construct_mm_rcm(
            [
                {"perf_latency_p99": 100, "perf_throughput": 0},
                {"perf_latency_p99": 0, "perf_throughput": 150},
            ],
            constraint_manager,
        )
        self.assertTrue(constraint_manager.satisfies_constraints(rcm))

        # Model A exceeds latency
        rcm = self._construct_mm_rcm(
            [
                {"perf_latency_p99": 105, "perf_throughput": 0},
                {"perf_latency_p99": 0, "perf_throughput": 150},
            ],
            constraint_manager,
        )
        self.assertFalse(constraint_manager.satisfies_constraints(rcm))

    def test_multi_model_min_constraint_checks(self):
        """
        Test that satisfies_constraints works for multi-model
        with a min style constraints
        """
        # Constraints are:
        #  Model A: P99 Latency max of 50
        #  Model B: Throughput min of 100
        constraint_manager = ConstraintManager(
            config=self._create_multi_model_with_no_global_constraints()
        )

        # Model B doesn't have enough throughput
        rcm = self._construct_mm_rcm(
            [
                {"perf_latency_p99": 50, "perf_throughput": 0},
                {"perf_latency_p99": 0, "perf_throughput": 99},
            ],
            constraint_manager,
        )
        self.assertFalse(constraint_manager.satisfies_constraints(rcm))

        # Constraints are:
        #  Model A: P99 Latency max of 75
        #  Model B: Throughput min of 125
        constraint_manager = ConstraintManager(
            config=self._create_multi_model_with_matching_global_constraints()
        )

        # Model B doesn't have enough throughput
        rcm = self._construct_mm_rcm(
            [
                {"perf_latency_p99": 75, "perf_throughput": 0},
                {"perf_latency_p99": 0, "perf_throughput": 120},
            ],
            constraint_manager,
        )
        self.assertFalse(constraint_manager.satisfies_constraints(rcm))

        # Constraints are:
        #  Model A: P99 Latency max of 100
        #  Model B: Throughput min of 150
        constraint_manager = ConstraintManager(
            config=self._create_multi_model_with_different_global_constrants()
        )

        # Model B doesn't have enough throughput
        rcm = self._construct_mm_rcm(
            [
                {"perf_latency_p99": 100, "perf_throughput": 0},
                {"perf_latency_p99": 0, "perf_throughput": 140},
            ],
            constraint_manager,
        )
        self.assertFalse(constraint_manager.satisfies_constraints(rcm))

    def test_single_model_max_failure_percentage(self):
        """
        Test that constraint_failure_percentage works for a single model
        with a max style constraint
        """
        config = self._create_single_model_with_constraints()
        constraint_manager = ConstraintManager(config)

        # Constraint is P99 Latency max of 100
        rcm = self._construct_rcm({"perf_latency_p99": 225}, constraint_manager)
        self.assertEqual(constraint_manager.constraint_failure_percentage(rcm), 125)
        rcm = self._construct_rcm({"perf_latency_p99": 150}, constraint_manager)
        self.assertEqual(constraint_manager.constraint_failure_percentage(rcm), 50)

        rcm = self._construct_rcm({"perf_latency_p99": 100}, constraint_manager)
        self.assertEqual(constraint_manager.constraint_failure_percentage(rcm), 0)

        rcm = self._construct_rcm({"perf_latency_p99": 99}, constraint_manager)
        self.assertEqual(constraint_manager.constraint_failure_percentage(rcm), 0)

    def test_single_model_min_failure_percentage(self):
        """
        Test that constraint_failure_percentage works for a single model
        with a min style constraint
        """

        config = self._create_single_model_global_constraints()
        constraint_manager = ConstraintManager(config)

        # Constraint is throughput min of 100
        rcm = self._construct_rcm({"perf_throughput": 25}, constraint_manager)
        self.assertEqual(constraint_manager.constraint_failure_percentage(rcm), 75)
        rcm = self._construct_rcm({"perf_throughput": 50}, constraint_manager)
        self.assertEqual(constraint_manager.constraint_failure_percentage(rcm), 50)

        rcm = self._construct_rcm({"perf_throughput": 100}, constraint_manager)
        self.assertEqual(constraint_manager.constraint_failure_percentage(rcm), 0)

        rcm = self._construct_rcm({"perf_throughput": 101}, constraint_manager)
        self.assertEqual(constraint_manager.constraint_failure_percentage(rcm), 0)

    def test_multi_model_failure_percentage(self):
        """
        Test that constraint_failure_percentage works for multi-model
        with a max style constraints
        """
        # Constraints are:
        #  Model A: P99 Latency max of 50
        #  Model B: Throughput min of 100
        constraint_manager = ConstraintManager(
            config=self._create_multi_model_with_no_global_constraints()
        )

        # Model A & B are both at boundaries
        rcm = self._construct_mm_rcm(
            [
                {"perf_latency_p99": 50, "perf_throughput": 0},
                {"perf_latency_p99": 0, "perf_throughput": 100},
            ],
            constraint_manager,
        )
        self.assertEqual(constraint_manager.constraint_failure_percentage(rcm), 0)

        # Model A exceeds latency, Model B misses on throughput - each by 20%
        rcm = self._construct_mm_rcm(
            [
                {"perf_latency_p99": 60, "perf_throughput": 0},
                {"perf_latency_p99": 0, "perf_throughput": 80},
            ],
            constraint_manager,
        )
        self.assertEqual(constraint_manager.constraint_failure_percentage(rcm), 40)

        # Model A exceeds latency by 40%, Model B misses on throughput by 10%
        rcm = self._construct_mm_rcm(
            [
                {"perf_latency_p99": 70, "perf_throughput": 0},
                {"perf_latency_p99": 0, "perf_throughput": 90},
            ],
            constraint_manager,
        )
        self.assertEqual(constraint_manager.constraint_failure_percentage(rcm), 50)

        # Constraints are:
        #  Model A: P99 Latency max of 75
        #  Model B: Throughput min of 125
        constraint_manager = ConstraintManager(
            config=self._create_multi_model_with_matching_global_constraints()
        )

        # Model A & B are both at boundaries
        rcm = self._construct_mm_rcm(
            [
                {"perf_latency_p99": 75, "perf_throughput": 0},
                {"perf_latency_p99": 0, "perf_throughput": 125},
            ],
            constraint_manager,
        )
        self.assertEqual(constraint_manager.constraint_failure_percentage(rcm), 0)

        # Model A exceeds latency, Model B misses on throughput - each by 25%
        rcm = self._construct_mm_rcm(
            [
                {"perf_latency_p99": 93.75, "perf_throughput": 0},
                {"perf_latency_p99": 0, "perf_throughput": 93.75},
            ],
            constraint_manager,
        )
        self.assertEqual(constraint_manager.constraint_failure_percentage(rcm), 50)

        # Model A exceeds latency by 30%, Model B misses on throughput by 20%
        rcm = self._construct_mm_rcm(
            [
                {"perf_latency_p99": 97.5, "perf_throughput": 0},
                {"perf_latency_p99": 0, "perf_throughput": 100},
            ],
            constraint_manager,
        )
        self.assertEqual(constraint_manager.constraint_failure_percentage(rcm), 50)

        # Constraints are:
        #  Model A: P99 Latency max of 100
        #  Model B: Throughput min of 150
        constraint_manager = ConstraintManager(
            config=self._create_multi_model_with_different_global_constrants()
        )

        # Model A & B are both at boundaries
        rcm = self._construct_mm_rcm(
            [
                {"perf_latency_p99": 100, "perf_throughput": 0},
                {"perf_latency_p99": 0, "perf_throughput": 150},
            ],
            constraint_manager,
        )
        self.assertEqual(constraint_manager.constraint_failure_percentage(rcm), 0)

        # Model A exceeds latency, Model B misses on throughput - each by 30%
        rcm = self._construct_mm_rcm(
            [
                {"perf_latency_p99": 130, "perf_throughput": 0},
                {"perf_latency_p99": 0, "perf_throughput": 105},
            ],
            constraint_manager,
        )
        self.assertEqual(constraint_manager.constraint_failure_percentage(rcm), 60)

        # Model A exceeds latency by 35%, Model B misses on throughput by 25%
        rcm = self._construct_mm_rcm(
            [
                {"perf_latency_p99": 135, "perf_throughput": 0},
                {"perf_latency_p99": 0, "perf_throughput": 112.5},
            ],
            constraint_manager,
        )
        self.assertEqual(constraint_manager.constraint_failure_percentage(rcm), 60)

    def _create_single_model_no_constraints(self):
        args = self._create_args()
        yaml_str = """
            profile_models:
              model_A
        """
        config = evaluate_mock_config(args, yaml_str, subcommand="profile")

        return config

    def _create_single_model_with_constraints(self):
        args = self._create_args()
        yaml_str = """
            profile_models:
              model_A:
                constraints:
                  perf_latency_p99:
                    max: 100
        """
        config = evaluate_mock_config(args, yaml_str, subcommand="profile")

        return config

    def _create_single_model_global_constraints(self):
        args = self._create_args()
        yaml_str = """
            profile_models:
              model_A

            constraints:
                perf_throughput:
                  min: 100
        """
        config = evaluate_mock_config(args, yaml_str, subcommand="profile")

        return config

    def _create_single_model_both_constraints(self):
        args = self._create_args()
        yaml_str = """
            profile_models:
              model_A:
                constraints:
                  perf_latency_p99:
                    max: 50

            constraints:
                perf_latency_p99:
                  max: 100
        """
        config = evaluate_mock_config(args, yaml_str, subcommand="profile")

        return config

    def _create_multi_model_with_no_global_constraints(self):
        args = self._create_args()
        yaml_str = """
            profile_models:
              model_A:
                constraints:
                  perf_latency_p99:
                    max: 50
              model_B:
                constraints:
                  perf_throughput:
                    min: 100
        """
        config = evaluate_mock_config(args, yaml_str, subcommand="profile")

        return config

    def _create_multi_model_with_matching_global_constraints(self):
        args = self._create_args()
        yaml_str = """
            profile_models:
              model_A:
                constraints:
                  perf_latency_p99:
                    max: 75
              model_B:
                constraints:
                  perf_throughput:
                    min: 125
            constraints:
                perf_throughput:
                  min: 100
                gpu_used_memory:
                  max: 1000
        """
        config = evaluate_mock_config(args, yaml_str, subcommand="profile")

        return config

    def _create_multi_model_with_different_global_constrants(self):
        args = self._create_args()
        yaml_str = """
            profile_models:
              model_A:
                constraints:
                  perf_latency_p99:
                    max: 100
              model_B:
                constraints:
                  perf_throughput:
                    min: 150
            constraints:
                gpu_used_memory:
                  max: 2000
        """
        config = evaluate_mock_config(args, yaml_str, subcommand="profile")

        return config

    def _construct_rcm(self, non_gpu_metric_values, constraint_manager):
        rcm = construct_run_config_measurement(
            model_name=MagicMock(),
            model_config_names=["model_A_config_name_0"],
            constraint_manager=constraint_manager,
            model_specific_pa_params=MagicMock(),
            gpu_metric_values=MagicMock(),
            non_gpu_metric_values=[non_gpu_metric_values],
        )

        return rcm

    def _construct_mm_rcm(self, non_gpu_metric_values, constraint_manager):
        rcm = construct_run_config_measurement(
            model_name=MagicMock(),
            model_config_names=["model_A_config_name_A", "model_B_config_name_B"],
            constraint_manager=constraint_manager,
            model_specific_pa_params=MagicMock(),
            gpu_metric_values=MagicMock(),
            non_gpu_metric_values=non_gpu_metric_values,
        )

        return rcm

    def _create_args(self):
        return ["model-analyzer", "profile", "-f", "config.yml", "-m", "."]


if __name__ == "__main__":
    unittest.main()
