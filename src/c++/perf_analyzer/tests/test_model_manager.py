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
from unittest.mock import MagicMock, patch

from model_analyzer.config.input.objects.config_model_profile_spec import (
    ConfigModelProfileSpec,
)
from model_analyzer.model_analyzer_exceptions import TritonModelAnalyzerException
from model_analyzer.model_manager import ModelManager
from model_analyzer.record.metrics_manager import MetricsManager
from model_analyzer.state.analyzer_state_manager import AnalyzerStateManager
from model_analyzer.triton.model.model_config import ModelConfig

from .common import test_result_collector as trc
from .common.test_utils import construct_run_config_measurement, evaluate_mock_config
from .mocks.mock_model_config import MockModelConfig
from .mocks.mock_run_configs import MockRunConfigs


class MetricsManagerSubclass(MetricsManager):
    """
    Overrides execute_model_run_config() to gather a list of MockRunConfigs that
    contain the configured values of each would-be 'executed' run_config
    """

    def __init__(self, config, client, server, gpus, result_manager, state_manager):
        super().__init__(config, client, server, gpus, result_manager, state_manager)
        self._configs = MockRunConfigs()
        self._perf_throughput = 1

    def get_run_configs(self):
        """Return the list of configs that would have been 'executed'"""
        return self._configs

    def execute_run_config(self, config):
        self._configs.add_from_model_run_config(config.model_run_configs()[0])
        return self._get_next_measurements()

    def _get_next_measurements(self):
        """Return fake measurements as if the run_configs had been executed"""

        throughput_value = self._get_next_perf_throughput_value()
        if throughput_value is None:
            return None
        else:
            return construct_run_config_measurement(
                model_name=MagicMock(),
                model_config_names=["test_model_config_name"],
                model_specific_pa_params=MagicMock(),
                gpu_metric_values=MagicMock(),
                non_gpu_metric_values=[{"perf_throughput": throughput_value}],
            )

    def _get_next_perf_throughput_value(self):
        self._perf_throughput *= 2
        return self._perf_throughput


class TestModelManager(trc.TestResultCollector):
    def tearDown(self):
        patch.stopall()
        ModelConfig._default_config_dict = {}

    def __init__(self, methodname):
        super().__init__(methodname)
        self._args = [
            "model-analyzer",
            "profile",
            "--model-repository",
            "cli_repository",
            "-f",
            "path-to-config-file",
        ]

        self._model_config_protobuf = """
            name: "test_model"
            platform: "tensorflow_graphdef"
            max_batch_size: 8
            input [
            {
                name: "INPUT__0"
                data_type: TYPE_FP32
                dims: [16]
            }
            ]
            output [
            {
                name: "OUTPUT__0"
                data_type: TYPE_FP32
                dims: [16]
            }
            ]
            instance_group [
            {
                kind: KIND_CPU
                count: 1
            }
            ]
            """

    def test_full_sweep(self):
        """
        Test a normal full sweep of options
        """
        expected_ranges = [
            {
                "instances": [1, 2, 3, 4, 5],
                "kind": ["KIND_GPU"],
                "batching": [0],
                "batch_sizes": [1],
                "max_batch_size": [1, 2, 4, 8],
                "concurrency": [1, 2, 4, 8, 16, 32, 64, 128],
            },
            {
                "instances": [1],
                "kind": ["KIND_CPU"],
                "batching": [None],
                "batch_sizes": [1],
                "max_batch_size": [8],
                "concurrency": [1, 2, 4, 8, 16, 32, 64, 128],
            },
        ]

        yaml_str = """
            profile_models: test_model
            run_config_search_max_concurrency: 128
            run_config_search_max_instance_count: 5
            run_config_search_max_model_batch_size: 8
            run_config_search_disable: False
            """

        self._test_model_manager(yaml_str, expected_ranges)

    def test_another_full_sweep(self):
        """
        Test another full sweep of options
        """

        expected_ranges = [
            {
                "instances": [1, 2, 3, 4, 5, 6, 7],
                "kind": ["KIND_GPU"],
                "batching": [0],
                "batch_sizes": [1],
                "max_batch_size": [8],
                "concurrency": [1, 2, 4, 8, 16, 32],
            },
            {
                "instances": [1],
                "kind": ["KIND_CPU"],
                "batching": [None],
                "batch_sizes": [1],
                "max_batch_size": [8],
                "concurrency": [1, 2, 4, 8, 16, 32],
            },
        ]

        yaml_str = """
            profile_models: test_model
            run_config_search_max_concurrency: 32
            run_config_search_max_instance_count: 7
            run_config_search_min_model_batch_size: 8
            run_config_search_max_model_batch_size: 8
            run_config_search_disable: False
            """

        self._test_model_manager(yaml_str, expected_ranges)

    def test_run_search_disable(self):
        """
        Test with run_config_search_disable=True

        Expect 1 result that matches the default configuration because no manual
        search options provided and automatic search disabled/ignored
        """

        expected_ranges = [
            {
                "instances": [1],
                "kind": ["KIND_CPU"],
                "batching": [None],
                "batch_sizes": [1],
                "max_batch_size": [8],
                "concurrency": [1],
            },
            {
                "instances": [1],
                "kind": ["KIND_CPU"],
                "batching": [None],
                "batch_sizes": [1],
                "max_batch_size": [8],
                "concurrency": [1],
            },
        ]

        yaml_str = """
            profile_models: test_model
            run_config_search_max_concurrency: 32
            run_config_search_max_instance_count: 7
            run_config_search_min_model_batch_size: 8
            run_config_search_max_model_batch_size: 8
            run_config_search_disable: True
            """

        self._test_model_manager(yaml_str, expected_ranges)

    def test_manual_concurrency(self):
        """
        Test with manually specified concurrencies
        """
        expected_ranges = [
            {
                "instances": [1, 2, 3, 4, 5, 6, 7],
                "kind": ["KIND_GPU"],
                "batching": [0],
                "batch_sizes": [1],
                "max_batch_size": [8],
                "concurrency": [5, 7],
            },
            {
                "instances": [1],
                "kind": ["KIND_CPU"],
                "batching": [None],
                "batch_sizes": [1],
                "max_batch_size": [8],
                "concurrency": [5, 7],
            },
        ]

        yaml_str = """
            profile_models: test_model
            run_config_search_max_concurrency: 32
            run_config_search_max_instance_count: 7
            run_config_search_min_model_batch_size: 8
            run_config_search_max_model_batch_size: 8
            run_config_search_disable: False
            concurrency: [5, 7]
            """

        self._test_model_manager(yaml_str, expected_ranges)

    def test_remote_mode(self):
        """
        Test remote mode
        """

        expected_ranges = [
            {
                "instances": [1, 2],
                "kind": ["KIND_GPU"],
                "batching": [0],
                "batch_sizes": [1],
                "max_batch_size": [8],
                "concurrency": [1, 2, 4, 8, 16],
            },
            {
                "instances": [1],
                "kind": ["KIND_CPU"],
                "batching": [None],
                "batch_sizes": [1],
                "max_batch_size": [8],
                "concurrency": [1, 2, 4, 8, 16],
            },
        ]

        yaml_str = """
            profile_models: test_model
            run_config_search_max_concurrency: 16
            run_config_search_max_instance_count: 2
            run_config_search_min_model_batch_size: 8
            run_config_search_max_model_batch_size: 8
            run_config_search_disable: False
            triton_launch_mode: remote
            """

        self._test_model_manager(yaml_str, expected_ranges)

    def test_manual_parameters(self):
        """
        Test with manually specified concurrencies and batch sizes
        """

        expected_ranges = [
            {
                "instances": [1, 2, 3, 4, 5, 6, 7],
                "kind": ["KIND_GPU"],
                "batching": [0],
                "batch_sizes": [1, 2, 3],
                "max_batch_size": [8],
                "concurrency": [2, 10, 18, 26, 34, 42, 50, 58],
            },
            {
                "instances": [1],
                "kind": ["KIND_CPU"],
                "batching": [None],
                "batch_sizes": [1, 2, 3],
                "max_batch_size": [8],
                "concurrency": [2, 10, 18, 26, 34, 42, 50, 58],
            },
        ]

        yaml_str = """
            profile_models: test_model
            run_config_search_max_concurrency: 512
            run_config_search_max_instance_count: 7
            run_config_search_min_model_batch_size: 8
            run_config_search_max_model_batch_size: 8
            run_config_search_disable: False
            concurrency:
                start: 2
                stop: 64
                step: 8
            batch_sizes: 1,2,3
            """

        self._test_model_manager(yaml_str, expected_ranges)

    def test_no_early_exit_client_batch_size(self):
        """
        Test that we will NOT early exit client batch size by default
        """
        self._test_early_exit_client_batch_size_helper(early_exit=False)

    def test_early_exit_client_batch_size(self):
        """
        Test that we will early exit client batch size if
        --early-exit-enable is true
        """
        self._test_early_exit_client_batch_size_helper(early_exit=True)

    def _test_early_exit_client_batch_size_helper(self, early_exit):
        args = self._args.copy()

        if early_exit:
            args.append("--early-exit-enable")

            expected_ranges = [
                {
                    "instances": [1],
                    "kind": ["KIND_GPU"],
                    "batching": [0],
                    "batch_sizes": [1, 2, 3, 4],
                    "max_batch_size": [8],
                    "concurrency": [1, 2, 4, 8, 16],
                },
                {
                    "instances": [1],
                    "kind": ["KIND_CPU"],
                    "batching": [None],
                    "batch_sizes": [1, 2, 3, 4, 7],
                    "max_batch_size": [8],
                    "concurrency": [1, 2, 4, 8],
                },
            ]

        else:
            expected_ranges = [
                {
                    "instances": [1],
                    "kind": ["KIND_GPU"],
                    "batching": [0],
                    "batch_sizes": [1, 2, 3, 4, 7],
                    "max_batch_size": [8],
                    "concurrency": [1, 2, 4, 8, 16],
                },
                {
                    "instances": [1],
                    "kind": ["KIND_CPU"],
                    "batching": [None],
                    "batch_sizes": [1, 2, 3, 4, 7],
                    "max_batch_size": [8],
                    "concurrency": [1, 2, 4, 8, 16],
                },
            ]

        yaml_content = """
            profile_models: test_model
            run_config_search_max_instance_count: 1
            run_config_search_min_model_batch_size: 8
            run_config_search_max_model_batch_size: 8
            concurrency: 16,1,2,4,8
            batch_sizes: 3,7,1,4,2
            """

        with patch.object(
            MetricsManagerSubclass, "_get_next_perf_throughput_value"
        ) as mock_method:
            # yapf: disable
            side_effect = [
                # Default config, bs=1, concurrency 1,2,4,8
                # Will early exit for concurrency
                # "Best" result for bs early exit is 5
                5, 5, 5, 5,

                # Default config, bs=2, concurrency 1,2,4,8
                # Will early exit for concurrency
                # "Best" result for bs early exit is 4
                4, 4, 4, 4,

                # Default config, bs=3, concurrency 1,2,4,8
                # Will early exit for concurrency
                # "Best" result for bs early exit is 6
                6, 6, 6, 6,

                # Default config, bs=4, concurrency 1,2,4,8
                # Will early exit for concurrency
                # "Best" result for bs early exit is 5
                # We will not early exit batch size
                5, 5, 5, 5,

                # Default config, bs=7, concurrency 1,2,4,8
                # Will not early exit for concurrency
                # "Best" result for bs early exit is 1
                # We are done sweeping batch size
                1, 1, 1, 1,


                # 1 instance, bs=1, concurrency 1,2,4,8,16
                # Will not early exit for concurrency
                # "Best" result for bs early exit is 10
                1, 1, 10, 1, 1,

                # 1 instance, bs=2, concurrency 1,2,4,8,16
                # Will not early exit for concurrency
                # "Best" result for bs early exit is 9
                1, 9, 1, 1, 2,

                # 1 instance, bs=3, concurrency 1,2,4,8,16
                # Will not early exit for concurrency
                # "Best" result for bs early exit is 8
                1, 1, 1, 8, 3,

                # 1 instance, bs=4, concurrency 1,2,4,8,16
                # Will not early exit for concurrency
                # "Best" result for bs early exit is 7
                # Will early exit batch size now
                1, 1, 7, 1, 4
            ]
            # Add a bunch of extra results for the no-early-exit case
            side_effect.extend([1]*100)
            mock_method.side_effect = side_effect
            # yapf: enable

            self._test_model_manager(yaml_content, expected_ranges, args=args)

    def test_model_config_parameters(self):
        """
        Test with manually specified model config parameters

        In this case we don't automatically search instances or dynamic_batching
        since model config parameters are specified.
        """

        expected_ranges = [
            {
                "instances": [1],
                "kind": ["KIND_CPU"],
                "batch_sizes": [1],
                "max_batch_size": [1, 3, 7, 8, 13],
                "concurrency": [1, 2, 4, 8],
            }
        ]

        yaml_str = """
            run_config_search_max_concurrency: 8
            run_config_search_max_instance_count: 16
            run_config_search_min_model_batch_size: 8
            run_config_search_max_model_batch_size: 8
            run_config_search_disable: False
            profile_models:
                test_model:
                    model_config_parameters:
                        max_batch_size: [1,3,7,13]
            """

        self._test_model_manager(yaml_str, expected_ranges)

    def test_default_config_always_run_no_dynamic_batching_off(self):
        """
        Test that the default config is run even when manual search excludes that case
        In this case, default config is (1 instance, max_batch_size=8, dynamic batching off)
        We should have a case of dynamic_batching off even though manual search only has it on
        """

        expected_ranges = [
            {
                "instances": [1],
                "kind": ["KIND_CPU"],
                "batching": [0],
                "max_queue_delay": ["200", "300"],
                "batch_sizes": [1],
                "max_batch_size": [1, 2, 4, 8, 16],
                "concurrency": [1, 2, 4, 8],
            },
            {
                "instances": [1],
                "kind": ["KIND_CPU"],
                "batching": [None],
                "batch_sizes": [1],
                "max_batch_size": [8],
                "concurrency": [1, 2, 4, 8],
            },
        ]

        yaml_str = """
            run_config_search_max_concurrency: 8
            run_config_search_max_instance_count: 16
            run_config_search_min_model_batch_size: 8
            run_config_search_max_model_batch_size: 8
            run_config_search_disable: False
            profile_models:
                test_model:
                    model_config_parameters:
                        max_batch_size: [1,2,4,8,16]
                        dynamic_batching:
                            max_queue_delay_microseconds: [200, 300]
            """

        self._test_model_manager(yaml_str, expected_ranges)

    def test_default_config_always_run_wrong_instances(self):
        """
        Test that the default config is run even when manual search excludes that case
        In this case, default config is (2 instances, max_batch_size=8, dynamic batching off)
        We should have a 2-instance case even though manual search only has 1-instance
        """

        self._model_config_protobuf = """
            name: "test_model"
            platform: "tensorflow_graphdef"
            max_batch_size: 8
            input [
            {
                name: "INPUT__0"
                data_type: TYPE_FP32
                dims: [16]
            }
            ]
            output [
            {
                name: "OUTPUT__0"
                data_type: TYPE_FP32
                dims: [16]
            }
            ]
            instance_group [
            {
                kind: KIND_CPU
                count: 2
            }
            ]
            """

        expected_ranges = [
            {
                "instances": [1],
                "kind": ["KIND_GPU"],
                "batching": [None],
                "batch_sizes": [1],
                "max_batch_size": [8],
                "concurrency": [1, 2, 4],
            },
            {
                "instances": [2],
                "kind": ["KIND_CPU"],
                "batching": [None],
                "batch_sizes": [1],
                "max_batch_size": [8],
                "concurrency": [1, 2, 4],
            },
        ]

        yaml_str = """
            run_config_search_max_concurrency: 4
            run_config_search_max_instance_count: 16
            run_config_search_min_model_batch_size: 8
            run_config_search_max_model_batch_size: 8
            run_config_search_disable: False
            profile_models:
                test_model:
                    model_config_parameters:
                        instance_group:
                        -
                            kind: KIND_GPU
                            count: 1
            """

        self._test_model_manager(yaml_str, expected_ranges)

    def test_default_config_always_run_cpu_vs_gpu(self):
        """
        If the default configuration had KIND_CPU, make sure it is run (even if everything
        else is the same)
        """

        self._model_config_protobuf = """
            name: "test_model"
            platform: "tensorflow_graphdef"
            max_batch_size: 8
            input [
            {
                name: "INPUT__0"
                data_type: TYPE_FP32
                dims: [16]
            }
            ]
            output [
            {
                name: "OUTPUT__0"
                data_type: TYPE_FP32
                dims: [16]
            }
            ]
            instance_group [
            {
                kind: KIND_CPU
                count: 1
            }
            ]
            """

        expected_ranges = [
            {
                "instances": [1],
                "kind": ["KIND_CPU"],
                "batching": [None],
                "batch_sizes": [1],
                "max_batch_size": [8],
                "concurrency": [1, 2, 4],
            },
            {
                "instances": [1],
                "kind": ["KIND_GPU"],
                "batching": [None],
                "batch_sizes": [1],
                "max_batch_size": [8],
                "concurrency": [1, 2, 4],
            },
        ]

        yaml_str = """
            run_config_search_max_concurrency: 4
            run_config_search_max_instance_count: 16
            run_config_search_min_model_batch_size: 8
            run_config_search_max_model_batch_size: 8
            run_config_search_disable: False
            profile_models:
                test_model:
                    model_config_parameters:
                        instance_group:
                        -
                            kind: KIND_GPU
                            count: 1
            """

        self._test_model_manager(yaml_str, expected_ranges)

    def test_default_config_always_run_automatic_search(self):
        """
        Test that the default config is run even when automatic search excludes that case
        In this case, default config is (4 instance, CPU, max_batch_size=8, dynamic batching off)
        We should have this 4 instance case though run_config_search_max_instance_count=1
        """

        self._model_config_protobuf = """
            name: "test_model"
            platform: "tensorflow_graphdef"
            max_batch_size: 8
            input [
            {
                name: "INPUT__0"
                data_type: TYPE_FP32
                dims: [16]
            }
            ]
            output [
            {
                name: "OUTPUT__0"
                data_type: TYPE_FP32
                dims: [16]
            }
            ]
            instance_group [
            {
                kind: KIND_CPU
                count: 4
            }
            ]
            """

        expected_ranges = [
            {
                "instances": [1],
                "kind": ["KIND_GPU"],
                "batching": [0],
                "batch_sizes": [1],
                "max_batch_size": [8],
                "concurrency": [1, 2, 4],
            },
            {
                "instances": [4],
                "kind": ["KIND_CPU"],
                "batching": [None],
                "batch_sizes": [1],
                "max_batch_size": [8],
                "concurrency": [1, 2, 4],
            },
        ]

        yaml_str = """
            run_config_search_max_concurrency: 4
            run_config_search_max_instance_count: 1
            run_config_search_min_model_batch_size: 8
            run_config_search_max_model_batch_size: 8
            run_config_search_disable: False
            profile_models: test_model
            """
        self._test_model_manager(yaml_str, expected_ranges)

    def test_throughput_early_exit_minimum_runs(self):
        """
        Test that there is an early back off when sweeping concurrency

        The behavior is that MA will try at least 4 concurrencies. If
        at that point none of the last 3 attempts have had satisfactory
        gain, it will stop

        This test hardcodes the 'throughput' to 1, so for all model
        configs the gain will be invalid and it will only try 4
        concurrencies of (1,2,4,8) despite max_concurrency=128

        This test also has multiple batch sizes to make sure that hitting
        an early exit on one doesn't cause the next one to be skipped
        """

        expected_ranges = [
            {
                "instances": [1, 2],
                "kind": ["KIND_GPU"],
                "batching": [0],
                "batch_sizes": [1, 2],
                "max_batch_size": [8],
                "concurrency": [1, 2, 4, 8],
            },
            {
                "instances": [1],
                "kind": ["KIND_CPU"],
                "batching": [None],
                "batch_sizes": [1, 2],
                "max_batch_size": [8],
                "concurrency": [1, 2, 4, 8],
            },
        ]

        yaml_str = """
            profile_models: test_model
            run_config_search_max_concurrency: 128
            run_config_search_max_instance_count: 2
            run_config_search_min_model_batch_size: 8
            run_config_search_max_model_batch_size: 8
            run_config_search_disable: False
            batch_sizes: 1,2
            """

        with patch.object(
            MetricsManagerSubclass, "_get_next_perf_throughput_value"
        ) as mock_method:
            mock_method.return_value = 1
            self._test_model_manager(yaml_str, expected_ranges)

    def test_no_early_exit_if_not_auto_search(self):
        """
        Test that there is NOT an early back off when sweeping concurrency if not in auto sweep mode

        This test hardcodes the 'throughput' to 1, so for all model
        configs the gain will be invalid. However, it should still sweep
        to concurrency of 128 due to the fact that it was manually specified instead
        of an auto-search
        """

        expected_ranges = [
            {
                "instances": [1, 2],
                "kind": ["KIND_GPU"],
                "batching": [0],
                "batch_sizes": [1, 2],
                "max_batch_size": [8],
                "concurrency": [1, 2, 4, 8, 16, 32, 64, 128],
            },
            {
                "instances": [1],
                "kind": ["KIND_CPU"],
                "batching": [None],
                "batch_sizes": [1, 2],
                "max_batch_size": [8],
                "concurrency": [1, 2, 4, 8, 16, 32, 64, 128],
            },
        ]

        yaml_str = """
            profile_models: test_model
            run_config_search_max_concurrency: 128
            run_config_search_max_instance_count: 2
            run_config_search_min_model_batch_size: 8
            run_config_search_max_model_batch_size: 8
            run_config_search_disable: False
            batch_sizes: 1,2
            concurrency: 1,2,4,8,16,32,64,128
            """

        with patch.object(
            MetricsManagerSubclass, "_get_next_perf_throughput_value"
        ) as mock_method:
            mock_method.return_value = 1
            self._test_model_manager(yaml_str, expected_ranges)

    def test_throughput_early_exit(self):
        """
        Test that there is an early back off when sweeping concurrency

        The behavior is that MA stop if it had 4 concurrencies in a row
        without any valid gain amongst any of them

        This test sets the 'throughput' to [1,2,4,8,16,16,16,16], which
        will cause an early exit after trying the 8th concurrency (128)
        instead of searching all the way to 2048
        """

        expected_ranges = [
            {
                "instances": [1],
                "kind": ["KIND_GPU"],
                "batching": [0],
                "batch_sizes": [1],
                "max_batch_size": [8],
                "concurrency": [1, 2, 4, 8, 16, 32, 64, 128],
            },
            {
                "instances": [1],
                "kind": ["KIND_CPU"],
                "batching": [None],
                "batch_sizes": [1],
                "max_batch_size": [8],
                "concurrency": [1, 2, 4, 8, 16, 32, 64, 128],
            },
        ]

        yaml_str = """
            profile_models: test_model
            run_config_search_max_concurrency: 2048
            run_config_search_max_instance_count: 1
            run_config_search_min_model_batch_size: 8
            run_config_search_max_model_batch_size: 8
            run_config_search_disable: False
            """

        with patch.object(
            MetricsManagerSubclass, "_get_next_perf_throughput_value"
        ) as mock_method:
            mock_method.side_effect = [
                1,
                2,
                4,
                8,
                16,
                16,
                16,
                16,
                1,
                2,
                4,
                8,
                16,
                16,
                16,
                16,
            ]
            self._test_model_manager(yaml_str, expected_ranges)

    @patch("model_analyzer.model_manager.INVALID_MEASUREMENT_THRESHOLD", 999)
    def test_bad_result_early_PA_exit(self):
        """
        Test that there is an early back off for bad result (out of memory)

        If no measurements are returned in an attempt, no further concurrencies
        should be tried.

        This test hardcodes the measurements to be empty (bad result), so for all
        model configs it will only try 1 concurrency despite max_concurrency=128
        """

        expected_ranges = [
            {
                "instances": [1, 2],
                "kind": ["KIND_GPU"],
                "batching": [0],
                "batch_sizes": [1],
                "max_batch_size": [8],
                "concurrency": [1],
            },
            {
                "instances": [1],
                "kind": ["KIND_CPU"],
                "batching": [None],
                "batch_sizes": [1],
                "max_batch_size": [8],
                "concurrency": [1],
            },
        ]

        yaml_str = """
            profile_models: test_model
            run_config_search_max_concurrency: 128
            run_config_search_max_instance_count: 2
            run_config_search_min_model_batch_size: 8
            run_config_search_max_model_batch_size: 8
            run_config_search_disable: False
            """

        with patch.object(
            MetricsManagerSubclass, "_get_next_perf_throughput_value"
        ) as mock_method:
            mock_method.return_value = None
            self._test_model_manager(yaml_str, expected_ranges)

    def test_report_failure_no_measurements(self):
        """
        Test that MA takes an exception if we detect no measurements returned from
        PA at the start of profile
        """

        yaml_str = """
            profile_models: test_model
            run_config_search_max_concurrency: 128
            run_config_search_max_instance_count: 2
            run_config_search_min_model_batch_size: 8
            run_config_search_max_model_batch_size: 8
            run_config_search_disable: False
            """

        with patch.object(
            MetricsManagerSubclass, "_get_next_perf_throughput_value"
        ) as mock_method:
            mock_method.return_value = None
            with self.assertRaises(TritonModelAnalyzerException):
                self._test_model_manager(yaml_str, None)

    def test_lower_throughput_early_batch_size_exit(self):
        """
        Test that there is an early back off for throughput decreasing
        when sweeping max_batch_size

        If a list of measurements is provided with a lower max throughput than the previous
        list of measurements, then we should early exit that max_batch_size sweep

        The test is set up such that at instance count 1, going from max_batch_size of 16 to 32
        does not increase the throughput, and thus we should not continue stepping max_batch_size.
        The same is true for instance count 2, going from max_batch_size of 32 to 64
        """

        expected_ranges = [
            # Instance count of 1 will stop after max_batch_size=32
            {
                "instances": [1],
                "kind": ["KIND_GPU"],
                "batching": [0],
                "batch_sizes": [1],
                "max_batch_size": [8, 16, 32],
                "concurrency": [1, 2, 4],
            },
            # Instance count of 2 will stop after max_batch_size=64
            {
                "instances": [2],
                "kind": ["KIND_GPU"],
                "batching": [0],
                "batch_sizes": [1],
                "max_batch_size": [8, 16, 32, 64],
                "concurrency": [1, 2, 4],
            },
            # Default config
            {
                "instances": [1],
                "kind": ["KIND_CPU"],
                "batching": [None],
                "batch_sizes": [1],
                "max_batch_size": [8],
                "concurrency": [1, 2, 4],
            },
        ]

        yaml_str = """
            profile_models: test_model
            run_config_search_max_concurrency: 4
            run_config_search_max_instance_count: 2
            run_config_search_min_model_batch_size: 8
            run_config_search_disable: False
            """

        with patch.object(
            MetricsManagerSubclass, "_get_next_perf_throughput_value"
        ) as mock_method:
            # yapf: disable
            mock_method.side_effect = [
                1, 2, 4,     # Default config, concurrency 1,2,4
                1, 2, 4,     # 1 Instance, Batch size 8, concurrency 1,2,4
                2, 4, 8,     # 1 Instance, Batch size 16, concurrency 1,2,4
                2, 4, 8,     # 1 Instance, Batch size 32, concurrency 1,2,4
                1, 2, 4,     # 1 Instance, Batch size 8, concurrency 1,2,4
                8, 4, 2,     # 1 Instance, Batch size 16, concurrency 1,2,4
                4, 8, 16,    # 1 Instance, Batch size 32, concurrency 1,2,4
                4, 8, 16     # 1 Instance, Batch size 64, concurrency 1,2,4
            ]
            # yapf: enable

            mock_method.return_value = None
            self._test_model_manager(yaml_str, expected_ranges)

    def test_no_max_batch_size_sweep_if_protobuf_0(self):
        """
        Test that if the max_batch_size is 0 in the original model config,
        then do not sweep max batch size or dynamic batching
        """

        self._model_config_protobuf = """
            name: "test_model"
            platform: "tensorflow_graphdef"
            max_batch_size: 0
            instance_group [
            {
                kind: KIND_CPU
                count: 1
            }
            ]
            """

        expected_ranges = [
            {
                "instances": [1],
                "kind": ["KIND_GPU"],
                "batching": [None],
                "batch_sizes": [1],
                "concurrency": [1, 2, 4],
            },
            {
                "instances": [1],
                "kind": ["KIND_CPU"],
                "batching": [None],
                "batch_sizes": [1],
                "concurrency": [1, 2, 4],
            },
        ]

        yaml_str = """
            run_config_search_max_concurrency: 4
            run_config_search_max_instance_count: 1
            run_config_search_disable: False
            profile_models: test_model
            """
        self._test_model_manager(yaml_str, expected_ranges)

    def test_client_batch_size_never_above_max_batch_size(self):
        """
        When we are sweeping through values of PA batch size and model max_batch_size,
        it should never be the case that PA batch size > model max_batch_size
        """

        expected_ranges = [
            {
                "instances": [1],
                "kind": ["KIND_GPU"],
                "batching": [0],
                "batch_sizes": [1, 2, 4, 8],
                "max_batch_size": [8],
                "concurrency": [1, 2, 4, 8],
            },
            {
                "instances": [1],
                "kind": ["KIND_GPU"],
                "batching": [0],
                "batch_sizes": [1, 2, 4, 8, 16],
                "max_batch_size": [16],
                "concurrency": [1, 2, 4, 8],
            },
            {
                "instances": [1],
                "kind": ["KIND_GPU"],
                "batching": [0],
                "batch_sizes": [1, 2, 4, 8, 16, 32],
                "max_batch_size": [32],
                "concurrency": [1, 2, 4, 8],
            },
            {
                "instances": [1],
                "kind": ["KIND_CPU"],
                "batching": [None],
                "batch_sizes": [1, 2, 4, 8],
                "max_batch_size": [8],
                "concurrency": [1, 2, 4, 8],
            },
        ]

        yaml_str = """
            profile_models: test_model
            run_config_search_max_concurrency: 8
            run_config_search_max_instance_count: 1
            run_config_search_min_model_batch_size: 8
            run_config_search_max_model_batch_size: 32
            run_config_search_disable: False
            batch_sizes: 1,2,4,8,16,32,64
            """

        self._test_model_manager(yaml_str, expected_ranges)

    @patch(
        "model_analyzer.triton.model.model_config.ModelConfig.is_ensemble",
        return_value=True,
    )
    def test_ensemble_illegal_checks(self, *args):
        """
        Test that RCS mode isn't set to brute and that multiple models are
        not provided when profiling an ensemble model
        """
        yaml_str = """
                  profile_models: ensemble_model, test_model
                  """

        args = self._args.copy()
        args.append("--run-config-search-mode")
        args.append("brute")

        self.mock_model_config = MockModelConfig(self._model_config_protobuf)
        self.mock_model_config.start()
        config = evaluate_mock_config(args, yaml_str, subcommand="profile")

        state_manager = AnalyzerStateManager(config, MagicMock())
        metrics_manager = MetricsManagerSubclass(
            config, MagicMock(), MagicMock(), MagicMock(), MagicMock(), state_manager
        )
        model_manager = ModelManager(
            config,
            MagicMock(),
            MagicMock(),
            MagicMock(),
            metrics_manager,
            MagicMock(),
            state_manager,
            MagicMock(),
        )

        # Multiple model check
        models = [
            ConfigModelProfileSpec("ensemble_model"),
            ConfigModelProfileSpec("test_model"),
        ]
        with self.assertRaises(TritonModelAnalyzerException):
            model_manager._check_for_ensemble_model_incompatibility(models)

        # RunConfigSearch check
        models = [
            ConfigModelProfileSpec("ensemble_model"),
        ]
        with self.assertRaises(TritonModelAnalyzerException):
            model_manager._check_for_ensemble_model_incompatibility(models)
        self.mock_model_config.stop()

    @patch(
        "model_analyzer.triton.model.model_config.ModelConfig.is_ensemble",
        return_value=True,
    )
    def test_ensemble_makes_quick_default(self, *args):
        """
        Test that the default RCS mode is switched from brute to quick
        when profiling an ensemble model
        """
        yaml_str = """
                  profile_models: ensemble_model
                  """

        args = self._args.copy()

        self.mock_model_config = MockModelConfig(self._model_config_protobuf)
        self.mock_model_config.start()
        config = evaluate_mock_config(args, yaml_str, subcommand="profile")

        state_manager = AnalyzerStateManager(config, MagicMock())
        metrics_manager = MetricsManagerSubclass(
            config, MagicMock(), MagicMock(), MagicMock(), MagicMock(), state_manager
        )
        model_manager = ModelManager(
            config,
            MagicMock(),
            MagicMock(),
            MagicMock(),
            metrics_manager,
            MagicMock(),
            state_manager,
            MagicMock(),
        )

        models = [
            ConfigModelProfileSpec("ensemble_model"),
        ]
        model_manager._check_for_ensemble_model_incompatibility(models)

        self.assertEqual(config.run_config_search_mode, "quick")

    @patch(
        "model_analyzer.triton.model.model_config.ModelConfig.is_ensemble",
        return_value=False,
    )
    def test_cpu_only_composing_models_error(self, *args):
        """
        Test that --cpu-only-composing-models errors when
        set for non-ensemble/BLS models
        """
        yaml_str = """
                  profile_models: test_model
                  """

        args = self._args.copy()
        args.append("--cpu-only-composing-models")
        args.append("composing_modelA,composing_modelB")

        self.mock_model_config = MockModelConfig(self._model_config_protobuf)
        self.mock_model_config.start()
        config = evaluate_mock_config(args, yaml_str, subcommand="profile")

        state_manager = AnalyzerStateManager(config, MagicMock())
        metrics_manager = MetricsManagerSubclass(
            config, MagicMock(), MagicMock(), MagicMock(), MagicMock(), state_manager
        )
        model_manager = ModelManager(
            config,
            MagicMock(),
            MagicMock(),
            MagicMock(),
            metrics_manager,
            MagicMock(),
            state_manager,
            MagicMock(),
        )

        # RunConfigSearch check
        models = [
            ConfigModelProfileSpec("test_model"),
        ]
        with self.assertRaises(TritonModelAnalyzerException):
            model_manager._check_for_ensemble_model_incompatibility(models)
        self.mock_model_config.stop()

    def _test_model_manager(self, yaml_content, expected_ranges, args=None):
        """
        Test helper function that passes the given yaml_str into
        model_manager, runs the model, and confirms the result is as expected
        based on a full cartesian product of the lists in the input list of
        dicts expected_ranges
        """

        if args is None:
            args = self._args

        # Use mock model config or else TritonModelAnalyzerException will be thrown as it tries to read from disk
        self.mock_model_config = MockModelConfig(self._model_config_protobuf)
        self.mock_model_config.start()
        config = evaluate_mock_config(args, yaml_content, subcommand="profile")

        state_manager = AnalyzerStateManager(config, MagicMock())
        metrics_manager = MetricsManagerSubclass(
            config, MagicMock(), MagicMock(), MagicMock(), MagicMock(), state_manager
        )
        model_manager = ModelManager(
            config,
            MagicMock(),
            MagicMock(),
            MagicMock(),
            metrics_manager,
            MagicMock(),
            state_manager,
            MagicMock(),
        )

        model_manager.run_models([config.profile_models[0]])
        self.mock_model_config.stop()

        self._check_results(model_manager, expected_ranges)

    def _check_results(self, model_manager, expected_ranges):
        """
        Create a set of expected and actual run configs and confirm they are equal
        """
        run_configs = model_manager._metrics_manager.get_run_configs()
        expected_configs = MockRunConfigs()
        expected_configs.populate_from_ranges(expected_ranges)

        self.assertEqual(
            run_configs.get_configs_set(), expected_configs.get_configs_set()
        )
        self.assertEqual(
            run_configs.get_num_configs(), expected_configs.get_num_configs()
        )


if __name__ == "__main__":
    unittest.main()
