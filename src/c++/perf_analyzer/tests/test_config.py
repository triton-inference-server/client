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

import re
import unittest
from argparse import Namespace
from copy import deepcopy
from typing import Any, Dict, List, Optional
from unittest.mock import patch

from model_analyzer.cli.cli import CLI
from model_analyzer.config.input.config_command_profile import ConfigCommandProfile
from model_analyzer.config.input.config_command_report import ConfigCommandReport
from model_analyzer.config.input.config_enum import ConfigEnum
from model_analyzer.config.input.config_list_generic import ConfigListGeneric
from model_analyzer.config.input.config_list_numeric import ConfigListNumeric
from model_analyzer.config.input.config_list_string import ConfigListString
from model_analyzer.config.input.config_object import ConfigObject
from model_analyzer.config.input.config_primitive import ConfigPrimitive
from model_analyzer.config.input.config_sweep import ConfigSweep
from model_analyzer.config.input.config_union import ConfigUnion
from model_analyzer.config.input.objects.config_model_profile_spec import (
    ConfigModelProfileSpec,
)
from model_analyzer.config.input.objects.config_plot import ConfigPlot
from model_analyzer.constants import CONFIG_PARSER_FAILURE
from model_analyzer.model_analyzer_exceptions import TritonModelAnalyzerException
from model_analyzer.result.model_constraints import ModelConstraints

from .common import test_result_collector as trc
from .mocks.mock_config import MockConfig
from .mocks.mock_numba import MockNumba
from .mocks.mock_os import MockOSMethods


class TestConfig(trc.TestResultCollector):
    def _create_parameters(
        self, batch_sizes: List = [], concurrency: List = [], request_rate: List = []
    ) -> Dict:
        return {
            "batch_sizes": batch_sizes,
            "concurrency": concurrency,
            "request_rate": request_rate,
        }

    def _evaluate_config(
        self, args, yaml_content, subcommand="profile", numba_available=True
    ):
        mock_numba = MockNumba(
            mock_paths=["model_analyzer.config.input.config_command_profile"],
            is_available=numba_available,
        )

        mock_config = MockConfig(args, yaml_content)
        mock_config.start()
        mock_numba.start()

        if subcommand == "report":
            config = ConfigCommandReport()
        else:
            config = ConfigCommandProfile()
        cli = CLI()
        cli.add_subcommand(cmd=subcommand, config=config, help="Test subcommand help")
        cli.parse()
        mock_config.stop()
        mock_numba.stop()
        return config

    def _assert_error_on_evaluate_config(
        self, args, yaml_content, subcommand="profile"
    ):
        mock_config = MockConfig(args, yaml_content)
        mock_config.start()
        if subcommand == "report":
            config = ConfigCommandReport()
        else:
            config = ConfigCommandProfile()
        cli = CLI()
        cli.add_subcommand(cmd=subcommand, config=config, help="Test subcommand help")
        # When a required field is not specified, parse will lead to an
        # exception
        with self.assertRaises(TritonModelAnalyzerException):
            cli.parse()
        mock_config.stop()

    def _assert_equality_of_model_configs(self, model_configs, expected_model_configs):
        self.assertEqual(len(model_configs), len(expected_model_configs))
        for model_config, expected_model_config in zip(
            model_configs, expected_model_configs
        ):
            self.assertEqual(expected_model_config.cpu_only(), model_config.cpu_only())
            self.assertEqual(
                expected_model_config.model_name(), model_config.model_name()
            )
            self.assertEqual(
                expected_model_config.parameters(), model_config.parameters()
            )
            self.assertEqual(
                expected_model_config.constraints(), model_config.constraints()
            )
            self.assertEqual(
                expected_model_config.objectives(), model_config.objectives()
            )
            self.assertEqual(
                expected_model_config.model_config_parameters(),
                model_config.model_config_parameters(),
            )

    def _assert_equality_of_plot_configs(self, plot_configs, expected_plot_configs):
        self.assertEqual(len(plot_configs), len(expected_plot_configs))
        for plot_config, expected_plot_config in zip(
            plot_configs, expected_plot_configs
        ):
            self.assertEqual(expected_plot_config.name(), plot_config.name())
            self.assertEqual(expected_plot_config.title(), plot_config.title())
            self.assertEqual(expected_plot_config.x_axis(), plot_config.x_axis())
            self.assertEqual(expected_plot_config.y_axis(), plot_config.y_axis())
            self.assertEqual(expected_plot_config.monotonic(), plot_config.monotonic())

    def _assert_model_config_types(self, model_config):
        self.assertIsInstance(model_config.field_type(), ConfigUnion)

        if isinstance(model_config.field_type().raw_value(), ConfigListGeneric):
            self.assertIsInstance(
                model_config.field_type().raw_value().container_type(), ConfigUnion
            )
        else:
            self.assertIsInstance(model_config.field_type().raw_value(), ConfigObject)

    def _assert_model_object_types(
        self,
        model_config,
        model_name,
        check_parameters=False,
        check_concurrency=False,
        check_batch_size=False,
    ):
        if isinstance(model_config, ConfigUnion):
            self.assertIsInstance(model_config.raw_value(), ConfigObject)
            self.assertIsInstance(
                model_config.raw_value().raw_value()[model_name], ConfigObject
            )
        else:
            self.assertIsInstance(model_config, ConfigObject)
            if check_parameters:
                parameters_config = model_config.raw_value()["parameters"]
                self.assertIsInstance(parameters_config, ConfigObject)

                if check_concurrency:
                    self.assertIsInstance(
                        parameters_config.raw_value()["concurrency"], ConfigListNumeric
                    )

    def _assert_model_config_params(self, model_config_parameters):
        self.assertIsInstance(model_config_parameters, ConfigObject)

        input_param = model_config_parameters.raw_value()["input"]
        self.assertIsInstance(input_param, ConfigSweep)
        # Is list of params
        self.assertIsInstance(input_param.raw_value(), ConfigListGeneric)

        # Each subitem is also a list
        self.assertIsInstance(input_param.raw_value().container_type(), ConfigUnion)

        single_sweep_param = input_param.raw_value().raw_value()[0].raw_value()
        self.assertIsInstance(single_sweep_param.raw_value()[0], ConfigObject)

        # Check types for 'name'
        name_param = single_sweep_param.raw_value()[0].raw_value()["name"]
        self.assertIsInstance(name_param, ConfigSweep)
        self.assertIsInstance(name_param.raw_value(), ConfigListGeneric)
        self.assertIsInstance(name_param.raw_value().container_type(), ConfigUnion)

        # Check types for 'data_type'
        data_type_param = single_sweep_param.raw_value()[0].raw_value()["data_type"]
        self.assertIsInstance(data_type_param, ConfigSweep)
        self.assertIsInstance(data_type_param.raw_value(), ConfigListGeneric)
        self.assertIsInstance(data_type_param.raw_value().container_type(), ConfigUnion)
        self.assertIsInstance(data_type_param.raw_value().raw_value()[0], ConfigUnion)
        self.assertIsInstance(
            data_type_param.raw_value().raw_value()[0].raw_value(), ConfigEnum
        )

        # Check types for 'dims'
        dims_param = single_sweep_param.raw_value()[0].raw_value()["dims"]
        self.assertIsInstance(dims_param, ConfigSweep)
        self.assertIsInstance(dims_param.raw_value(), ConfigListGeneric)
        self.assertIsInstance(dims_param.raw_value().container_type(), ConfigUnion)
        self.assertIsInstance(dims_param.raw_value().raw_value()[0], ConfigUnion)
        self.assertIsInstance(
            dims_param.raw_value().raw_value()[0].raw_value(), ConfigListGeneric
        )
        self.assertIsInstance(
            dims_param.raw_value().raw_value()[0].raw_value().raw_value()[0],
            ConfigPrimitive,
        )

        # Check types for 'format'
        format_param = single_sweep_param.raw_value()[0].raw_value()["format"]
        self.assertIsInstance(format_param, ConfigSweep)
        self.assertIsInstance(format_param.raw_value(), ConfigListGeneric)
        self.assertIsInstance(format_param.raw_value().container_type(), ConfigUnion)

    def _assert_model_str_type(self, model_config):
        self.assertIsInstance(model_config, ConfigUnion)
        self.assertIsInstance(model_config.raw_value(), ConfigPrimitive)

    def setUp(self):
        # Mock path validation
        self.mock_os = MockOSMethods(
            mock_paths=["model_analyzer.config.input.config_utils"]
        )
        self.mock_os.start()

    def tearDown(self):
        self.mock_os.stop()
        patch.stopall()

    def test_config(self):
        args = [
            "model-analyzer",
            "profile",
            "--model-repository",
            "cli_repository",
            "-f",
            "path-to-config-file",
            "--profile-models",
            "vgg11",
        ]
        yaml_content = "model_repository: yaml_repository"
        config = self._evaluate_config(args, yaml_content)

        # CLI flag has the highest priority
        self.assertEqual(config.get_all_config()["model_repository"], "cli_repository")

        args = [
            "model-analyzer",
            "profile",
            "-f",
            "path-to-config-file",
            "--profile-models",
            "vgg11",
        ]
        yaml_content = "model_repository: yaml_repository"
        config = self._evaluate_config(args, yaml_content)

        # If CLI flag doesn't exist, YAML config has the highest priority
        self.assertEqual(config.get_all_config()["model_repository"], "yaml_repository")

        args = ["model-analyzer", "profile", "-f", "path-to-config-file"]
        yaml_content = "model_repository: yaml_repository"
        self._assert_error_on_evaluate_config(args, yaml_content)

    def test_missing_config(self):
        """Test that we fail if the required option profile-models is omitted"""

        args = ["model-analyzer", "profile", "-f", "path-to-config-file"]
        yaml_content = "model_repository: yaml_repository"
        self._assert_error_on_evaluate_config(args, yaml_content)

    def test_conflicting_configs(self):
        """Test that we fail if an option is specified in both CLI and YAML"""

        args = [
            "model-analyzer",
            "profile",
            "--model-repository",
            "cli_repository",
            "-f",
            "path-to-config-file",
            "--profile-models",
            "vgg11",
        ]
        yaml_content = "profile_models: add_sub"
        self._assert_error_on_evaluate_config(args, yaml_content)

    def test_range_and_list_values(self):
        args = [
            "model-analyzer",
            "profile",
            "--model-repository",
            "cli_repository",
            "-f",
            "path-to-config-file",
        ]
        yaml_content = "profile_models: model_1,model_2"
        config = self._evaluate_config(args, yaml_content)
        expected_model_configs = [
            ConfigModelProfileSpec(
                "model_1",
                parameters={"batch_sizes": [1], "concurrency": [], "request_rate": []},
                objectives={"perf_throughput": 10},
            ),
            ConfigModelProfileSpec(
                "model_2",
                parameters={"batch_sizes": [1], "concurrency": [], "request_rate": []},
                objectives={"perf_throughput": 10},
            ),
        ]
        self._assert_equality_of_model_configs(
            config.get_all_config()["profile_models"], expected_model_configs
        )
        self.assertIsInstance(
            config.get_config()["profile_models"].field_type().raw_value(), ConfigObject
        )

        yaml_content = """
profile_models:
    - model_1
    - model_2
"""
        config = self._evaluate_config(args, yaml_content)
        self._assert_equality_of_model_configs(
            config.get_all_config()["profile_models"], expected_model_configs
        )

        model_config = config.get_config()["profile_models"]

        self._assert_model_config_types(model_config)
        self.assertIsInstance(
            model_config.field_type()
            .raw_value()
            .raw_value()["model_1"]
            .raw_value()["objectives"],
            ConfigUnion,
        )
        self.assertIsInstance(
            model_config.field_type()
            .raw_value()
            .raw_value()["model_1"]
            .raw_value()["parameters"],
            ConfigObject,
        )

        args = [
            "model-analyzer",
            "profile",
            "--model-repository",
            "cli_repository",
            "-f",
            "path-to-config-file",
            "--profile-models",
            "model_1,model_2",
        ]
        yaml_content = """
batch_sizes:
    - 2
    - 3
"""
        config = self._evaluate_config(args, yaml_content)
        self.assertEqual(config.get_all_config()["batch_sizes"], [2, 3])
        self.assertIsInstance(
            config.get_config()["batch_sizes"].field_type(), ConfigListNumeric
        )

        yaml_content = """
batch_sizes: 2
"""
        config = self._evaluate_config(args, yaml_content)
        self.assertEqual(config.get_all_config()["batch_sizes"], [2])
        self.assertIsInstance(
            config.get_config()["batch_sizes"].field_type(), ConfigListNumeric
        )

        yaml_content = """
concurrency: 2
"""
        config = self._evaluate_config(args, yaml_content)
        self.assertEqual(config.get_all_config()["concurrency"], [2])
        self.assertIsInstance(
            config.get_config()["concurrency"].field_type(), ConfigListNumeric
        )
        self.assertEqual(config.get_all_config()["batch_sizes"], [1])
        self.assertIsInstance(
            config.get_config()["batch_sizes"].field_type(), ConfigListNumeric
        )

        yaml_content = """
batch_sizes:
    start: 2
    stop: 6
"""
        config = self._evaluate_config(args, yaml_content)
        self.assertEqual(config.get_all_config()["batch_sizes"], [2, 3, 4, 5, 6])
        self.assertIsInstance(
            config.get_config()["batch_sizes"].field_type(), ConfigListNumeric
        )

        yaml_content = """
batch_sizes:
    start: 2
    stop: 6
    step: 2
"""
        config = self._evaluate_config(args, yaml_content)
        self.assertEqual(config.get_all_config()["batch_sizes"], [2, 4, 6])
        self.assertIsInstance(
            config.get_config()["batch_sizes"].field_type(), ConfigListNumeric
        )

    def test_object(self):
        args = [
            "model-analyzer",
            "profile",
            "--model-repository",
            "cli_repository",
            "-f",
            "path-to-config-file",
        ]
        yaml_content = """
profile_models:
  -
    vgg_16_graphdef:
      parameters:
        concurrency:
          - 1
          - 2
          - 3
          - 4
  - vgg_19_graphdef
"""
        config = self._evaluate_config(args, yaml_content)
        model_config = config.get_config()["profile_models"]
        self._assert_model_config_types(model_config)

        expected_model_objects = [
            ConfigModelProfileSpec(
                "vgg_16_graphdef",
                parameters={
                    "batch_sizes": [1],
                    "concurrency": [1, 2, 3, 4],
                    "request_rate": [],
                },
                objectives={"perf_throughput": 10},
            ),
            ConfigModelProfileSpec(
                "vgg_19_graphdef",
                parameters={"batch_sizes": [1], "concurrency": [], "request_rate": []},
                objectives={"perf_throughput": 10},
            ),
        ]

        # Check the types for the first value
        first_model = (
            model_config.field_type().raw_value().raw_value()["vgg_16_graphdef"]
        )
        self._assert_model_object_types(
            first_model,
            "vgg_16_graphdef",
            check_parameters=True,
            check_concurrency=True,
        )

        # Check the types for the second value
        second_model = (
            model_config.field_type().raw_value().raw_value()["vgg_19_graphdef"]
        )
        self._assert_model_object_types(second_model, "vgg_19_graphdef")

        self._assert_equality_of_model_configs(
            config.get_all_config()["profile_models"], expected_model_objects
        )

        yaml_content = """
profile_models:
  vgg_16_graphdef:
    parameters:
      concurrency:
        - 1
        - 2
        - 3
        - 4
  vgg_19_graphdef:
    parameters:
      concurrency:
        - 1
        - 2
        - 3
        - 4
      batch_sizes:
          start: 2
          stop: 6
          step: 2
"""
        expected_model_objects = [
            ConfigModelProfileSpec(
                "vgg_16_graphdef",
                parameters={
                    "batch_sizes": [1],
                    "concurrency": [1, 2, 3, 4],
                    "request_rate": [],
                },
                objectives={"perf_throughput": 10},
            ),
            ConfigModelProfileSpec(
                "vgg_19_graphdef",
                parameters={
                    "concurrency": [1, 2, 3, 4],
                    "batch_sizes": [2, 4, 6],
                    "request_rate": [],
                },
                objectives={"perf_throughput": 10},
            ),
        ]
        config = self._evaluate_config(args, yaml_content)
        self._assert_equality_of_model_configs(
            config.get_all_config()["profile_models"], expected_model_objects
        )

        model_config = config.get_config()["profile_models"]
        self._assert_model_config_types(model_config)

        # first model
        first_model = (
            model_config.field_type().raw_value().raw_value()["vgg_16_graphdef"]
        )
        self._assert_model_object_types(
            first_model,
            "vgg_16_graphdef",
            check_parameters=True,
            check_concurrency=True,
            check_batch_size=True,
        )

        # second model
        second_model = (
            model_config.field_type().raw_value().raw_value()["vgg_19_graphdef"]
        )
        self._assert_model_object_types(
            second_model,
            "vgg_19_graphdef",
            check_parameters=True,
            check_concurrency=True,
            check_batch_size=True,
        )

    def test_constraints(self):
        args = [
            "model-analyzer",
            "profile",
            "--model-repository",
            "cli_repository",
            "-f",
            "path-to-config-file",
        ]
        yaml_content = """
profile_models:
  -
    vgg_16_graphdef:
      parameters:
        concurrency:
          - 1
          - 2
          - 3
          - 4
      objectives:
        perf_throughput: 10
        gpu_used_memory: 5
      constraints:
        gpu_used_memory:
          max: 80
  - vgg_19_graphdef
"""
        config = self._evaluate_config(args, yaml_content)
        expected_model_objects = [
            ConfigModelProfileSpec(
                "vgg_16_graphdef",
                parameters={
                    "batch_sizes": [1],
                    "concurrency": [1, 2, 3, 4],
                    "request_rate": [],
                },
                objectives={"perf_throughput": 10, "gpu_used_memory": 5},
                constraints={
                    "gpu_used_memory": {
                        "max": 80,
                    }
                },
            ),
            ConfigModelProfileSpec(
                "vgg_19_graphdef",
                parameters={"batch_sizes": [1], "concurrency": [], "request_rate": []},
                objectives={"perf_throughput": 10},
            ),
        ]
        self._assert_equality_of_model_configs(
            config.get_all_config()["profile_models"], expected_model_objects
        )

        # GPU Memory shouldn't have min
        yaml_content = """
profile_models:
  -
    vgg_16_graphdef:
      parameters:
        concurrency:
          - 1
          - 2
          - 3
          - 4
      objectives:
        - perf_throughput
        - gpu_used_memory
      constraints:
        gpu_memory:
          max: 80
          min: 45
  - vgg_19_graphdef
"""
        self._assert_error_on_evaluate_config(args, yaml_content)

        # Test objective key that is not one of the supported metrics
        yaml_content = """
profile_models:
  -
    vgg_16_graphdef:
      parameters:
        concurrency:
          - 1
          - 2
          - 3
          - 4
      objectives:
        - throughput
      constraints:
        gpu_used_memory:
          max: 80
  - vgg_19_graphdef
"""
        self._assert_error_on_evaluate_config(args, yaml_content)

    def test_validation(self):
        args = [
            "model-analyzer",
            "profile",
            "--model-repository",
            "cli_repository",
            "-f",
            "path-to-config-file",
        ]

        # end key should not be included in concurrency
        yaml_content = """
profile_models:
  -
    vgg_16_graphdef:
      parameters:
        concurrency:
            start: 4
            stop: 12
            end: 2
"""
        self._assert_error_on_evaluate_config(args, yaml_content)

        args = [
            "model-analyzer",
            "profile",
            "--model-repository",
            "cli_repository",
            "-f",
            "path-to-config-file",
        ]

        yaml_content = """
profile_models:
  -
    vgg_16_graphdef:
      parameters:
        concurrency:
            start: 13
            stop: 12
"""
        self._assert_error_on_evaluate_config(args, yaml_content)

    def test_config_model(self):
        args = [
            "model-analyzer",
            "profile",
            "--model-repository",
            "cli_repository",
            "-f",
            "path-to-config-file",
        ]
        yaml_content = """
profile_models:
  -
    vgg_16_graphdef:
        model_config_parameters:
            instance_group:
                -
                    kind: KIND_GPU
                    count: 1

"""
        config = self._evaluate_config(args, yaml_content)
        model_configs = config.get_all_config()["profile_models"]
        expected_model_configs = [
            ConfigModelProfileSpec(
                "vgg_16_graphdef",
                parameters={"batch_sizes": [1], "concurrency": [], "request_rate": []},
                objectives={"perf_throughput": 10},
                model_config_parameters={
                    "instance_group": [[{"kind": ["KIND_GPU"], "count": [1]}]]
                },
            )
        ]
        self._assert_equality_of_model_configs(model_configs, expected_model_configs)

        yaml_content = """
profile_models:
  -
    vgg_16_graphdef:
        model_config_parameters:
            instance_group:
                -
                    kind: KIND_GPU
                    count: 1

"""
        config = self._evaluate_config(args, yaml_content)
        model_configs = config.get_all_config()["profile_models"]
        expected_model_configs = [
            ConfigModelProfileSpec(
                "vgg_16_graphdef",
                parameters={"batch_sizes": [1], "concurrency": [], "request_rate": []},
                objectives={"perf_throughput": 10},
                model_config_parameters={
                    "instance_group": [[{"kind": ["KIND_GPU"], "count": [1]}]]
                },
            )
        ]
        self._assert_equality_of_model_configs(model_configs, expected_model_configs)

        args = [
            "model-analyzer",
            "profile",
            "--model-repository",
            "cli_repository",
            "-f",
            "path-to-config-file",
        ]
        yaml_content = """
            profile_models:
            -
                vgg_16_graphdef:
                    model_config_parameters:
                        instance_group:
                            -
                                kind: KIND_GPU
                                count: 1
                            -
                                kind: KIND_CPU
                                count: 1

            """
        config = self._evaluate_config(args, yaml_content)
        model_configs = config.get_all_config()["profile_models"]
        expected_model_configs = [
            ConfigModelProfileSpec(
                "vgg_16_graphdef",
                parameters={"batch_sizes": [1], "concurrency": [], "request_rate": []},
                objectives={"perf_throughput": 10},
                model_config_parameters={
                    "instance_group": [
                        [
                            {"kind": ["KIND_GPU"], "count": [1]},
                            {"kind": ["KIND_CPU"], "count": [1]},
                        ]
                    ]
                },
            )
        ]
        self._assert_equality_of_model_configs(model_configs, expected_model_configs)

        args = [
            "model-analyzer",
            "profile",
            "--model-repository",
            "cli_repository",
            "-f",
            "path-to-config-file",
        ]
        yaml_content = """
            profile_models:
            -
                vgg_16_graphdef:
                    model_config_parameters:
                        instance_group:
                        -
                            -
                                kind: KIND_GPU
                                count: 1
                        -
                            -
                                kind: KIND_CPU
                                count: 1

            """
        config = self._evaluate_config(args, yaml_content)
        model_configs = config.get_all_config()["profile_models"]
        expected_model_configs = [
            ConfigModelProfileSpec(
                "vgg_16_graphdef",
                parameters={"batch_sizes": [1], "concurrency": [], "request_rate": []},
                objectives={"perf_throughput": 10},
                model_config_parameters={
                    "instance_group": [
                        [{"kind": ["KIND_GPU"], "count": [1]}],
                        [{"kind": ["KIND_CPU"], "count": [1]}],
                    ]
                },
            )
        ]
        self._assert_equality_of_model_configs(model_configs, expected_model_configs)

        yaml_content = """
            profile_models:
            -
                vgg_16_graphdef:
                    model_config_parameters:
                        input:
                        -
                            name: NV_MODEL_INPUT
                            data_type: TYPE_FP32
                            format: FORMAT_NHWC
                            dims: [256, 256, 3]

            """
        config = self._evaluate_config(args, yaml_content)
        model_configs = config.get_all_config()["profile_models"]
        expected_model_configs = [
            ConfigModelProfileSpec(
                "vgg_16_graphdef",
                parameters={"batch_sizes": [1], "concurrency": [], "request_rate": []},
                objectives={"perf_throughput": 10},
                model_config_parameters={
                    "input": [
                        [
                            {
                                "name": ["NV_MODEL_INPUT"],
                                "data_type": ["TYPE_FP32"],
                                "format": ["FORMAT_NHWC"],
                                "dims": [[256, 256, 3]],
                            }
                        ]
                    ]
                },
            )
        ]

        model_config = config.get_config()["profile_models"]
        self._assert_model_config_types(model_config)

        model = model_config.field_type().raw_value().raw_value()["vgg_16_graphdef"]
        self._assert_model_object_types(model, "vgg_16_graphdef")

        model_config_parameters = model.raw_value()["model_config_parameters"]

        self._assert_model_config_params(model_config_parameters)
        self._assert_equality_of_model_configs(model_configs, expected_model_configs)

        yaml_content = """
            profile_models:
            -
                vgg_16_graphdef:
                    perf_analyzer_flags:
                        measurement-interval: 10000
                        model-version: 2
                        streaming: "header:value"

            """
        config = self._evaluate_config(args, yaml_content)
        model_configs = config.get_all_config()["profile_models"]
        expected_model_configs = [
            ConfigModelProfileSpec(
                "vgg_16_graphdef",
                parameters={"batch_sizes": [1], "concurrency": [], "request_rate": []},
                objectives={"perf_throughput": 10},
                perf_analyzer_flags={
                    "measurement-interval": 10000,
                    "model-version": 2,
                    "streaming": "header:value",
                },
            )
        ]
        self._assert_equality_of_model_configs(model_configs, expected_model_configs)

        yaml_content = """
            profile_models:
            -
                vgg_16_graphdef:
                    perf_analyzer_flags:
                        measurement-interval: 10000
                        model-version: 2
                        shape: ["name1:1,2,3", "name2:4,5,6"]

            """
        config = self._evaluate_config(args, yaml_content)
        model_configs = config.get_all_config()["profile_models"]
        expected_model_configs = [
            ConfigModelProfileSpec(
                "vgg_16_graphdef",
                parameters={"batch_sizes": [1], "concurrency": [], "request_rate": []},
                objectives={"perf_throughput": 10},
                perf_analyzer_flags={
                    "measurement-interval": 10000,
                    "model-version": 2,
                    "shape": ["name1:1,2,3", "name2:4,5,6"],
                },
            )
        ]
        self._assert_equality_of_model_configs(model_configs, expected_model_configs)

        yaml_content = """
            profile_models:
            -
                vgg_16_graphdef:
                    perf_analyzer_flags:
                        latency_report_file: ["file1", "file2"]

            """
        with self.assertRaises(TritonModelAnalyzerException):
            # latency_report_file is not additive
            self._evaluate_config(args, yaml_content)

        yaml_content = """
            profile_models:
            -
                vgg_16_graphdef:
                    perf_analyzer_flags:
                        disallowed-perf-flag: some_value

            """
        with self.assertRaises(TritonModelAnalyzerException):
            self._evaluate_config(args, yaml_content)

    def test_config_sweep(self):
        config_sweep = ConfigSweep(ConfigPrimitive(int))
        config_sweep.set_value(2)

    def test_config_plot(self):
        args = ["model-analyzer", "report", "-f", "path-to-config-file"]
        yaml_content = """
            report_model_configs: vgg_16_graphdef
            plots:
                test_plot:
                    title: Throughput vs. Latency
                    x_axis: perf_throughput
                    y_axis: perf_latency_p99
            """
        config = self._evaluate_config(args, yaml_content, subcommand="report")
        plot_configs = config.get_all_config()["plots"]
        expected_plot_configs = [
            ConfigPlot(
                "test_plot",
                title="Throughput vs. Latency",
                x_axis="perf_throughput",
                y_axis="perf_latency_p99",
            )
        ]
        self._assert_equality_of_plot_configs(plot_configs, expected_plot_configs)

        yaml_content = """
            report_model_configs: vgg_16_graphdef
            plots:
            - test_plot1:
                title: Throughput vs. Latency
                x_axis: perf_throughput
                y_axis: perf_latency_p99
            - test_plot2:
                title: GPU Memory vs. Latency
                x_axis: gpu_used_memory
                y_axis: perf_latency_p99
            """
        config = self._evaluate_config(args, yaml_content, subcommand="report")
        plot_configs = config.get_all_config()["plots"]
        expected_plot_configs = [
            ConfigPlot(
                "test_plot1",
                title="Throughput vs. Latency",
                x_axis="perf_throughput",
                y_axis="perf_latency_p99",
            ),
            ConfigPlot(
                "test_plot2",
                title="GPU Memory vs. Latency",
                x_axis="gpu_used_memory",
                y_axis="perf_latency_p99",
            ),
        ]
        self._assert_equality_of_plot_configs(plot_configs, expected_plot_configs)

        yaml_content = """
            report_model_configs: vgg_16_graphdef
            plots:
                test_plot1:
                    title: Throughput vs. Latency
                    x_axis: perf_throughput
                    y_axis: perf_latency_p99
                test_plot2:
                    title: GPU Memory vs. Latency
                    x_axis: gpu_used_memory
                    y_axis: perf_latency_p99
            """
        config = self._evaluate_config(args, yaml_content, subcommand="report")
        plot_configs = config.get_all_config()["plots"]
        expected_plot_configs = [
            ConfigPlot(
                "test_plot1",
                title="Throughput vs. Latency",
                x_axis="perf_throughput",
                y_axis="perf_latency_p99",
            ),
            ConfigPlot(
                "test_plot2",
                title="GPU Memory vs. Latency",
                x_axis="gpu_used_memory",
                y_axis="perf_latency_p99",
            ),
        ]
        self._assert_equality_of_plot_configs(plot_configs, expected_plot_configs)

        yaml_content = """
            report_model_configs:
              vgg_16_graphdef:
                plots:
                    test_plot1:
                        title: Throughput vs. Latency
                        x_axis: perf_throughput
                        y_axis: perf_latency_p99
                        monotonic: True
                    test_plot2:
                        title: GPU Memory vs. Latency
                        x_axis: gpu_used_memory
                        y_axis: perf_latency_p99
                        monotonic: False
            """
        config = self._evaluate_config(args, yaml_content, subcommand="report")
        plot_configs = config.get_all_config()["report_model_configs"][0].plots()
        expected_plot_configs = [
            ConfigPlot(
                "test_plot1",
                title="Throughput vs. Latency",
                x_axis="perf_throughput",
                y_axis="perf_latency_p99",
                monotonic=True,
            ),
            ConfigPlot(
                "test_plot2",
                title="GPU Memory vs. Latency",
                x_axis="gpu_used_memory",
                y_axis="perf_latency_p99",
                monotonic=False,
            ),
        ]
        self._assert_equality_of_plot_configs(plot_configs, expected_plot_configs)

    def test_error_messages(self):
        # ConfigListNumeric
        config_numeric = ConfigListNumeric(float)
        config_numeric.set_name("key")
        config_status = config_numeric.set_value(
            {"start": 12, "stop": 15, "undefined_key": 8}
        )

        config_message = config_status.message()
        result = re.search(".*'start'.*'stop'.*'undefined_key'.*", config_message)

        self.assertEqual(config_status.status(), CONFIG_PARSER_FAILURE)
        self.assertIsNotNone(config_message)
        self.assertIsNotNone(result)

        config_numeric = ConfigListNumeric(float)
        config_numeric.set_name("key")
        config_status = config_numeric.set_value({"start": 12, "stop": "two"})
        print(config_message)
        self.assertEqual(config_status.status(), CONFIG_PARSER_FAILURE)

        config_numeric = ConfigListNumeric(float)
        config_numeric.set_name("key")
        config_status = config_numeric.set_value({"start": "five", "stop": 2})
        config_message = config_status.message()
        print(config_message)
        self.assertEqual(config_status.status(), CONFIG_PARSER_FAILURE)

        config_numeric = ConfigListNumeric(float)
        config_numeric.set_name("key")
        config_status = config_numeric.set_value({"start": 10, "stop": 2})
        print(config_status.message())
        self.assertEqual(config_status.status(), CONFIG_PARSER_FAILURE)

        # ConfigUnion error message
        config_union = ConfigUnion([ConfigListNumeric(float), ConfigPrimitive(str)])
        config_union.set_name("key")

        # Dictionaries are not accepted.
        config_status = config_union.set_value({"a": "b"})
        print(config_status.message())
        self.assertEqual(config_status.status(), CONFIG_PARSER_FAILURE)

        # ConfigEnum
        config_enum = ConfigEnum(["a", "b"])
        config_enum.set_name("key")
        config_status = config_enum.set_value("c")
        print(config_status.message())
        self.assertEqual(config_status.status(), CONFIG_PARSER_FAILURE)

        # ConfigListGeneric
        config_list_generic = ConfigListGeneric(ConfigPrimitive(float))
        config_list_generic.set_name("key")
        config_status = config_list_generic.set_value({"a": "b"})
        print(config_status.message())
        self.assertEqual(config_status.status(), CONFIG_PARSER_FAILURE)

        # ConfigListString
        config_list_string = ConfigListString()
        config_list_string.set_name("key")
        config_status = config_list_string.set_value({"a": "b"})
        print(config_status.message())
        self.assertEqual(config_status.status(), CONFIG_PARSER_FAILURE)

        config_status = config_list_string.set_value([{"a": "b"}])
        print(config_status.message())
        self.assertEqual(config_status.status(), CONFIG_PARSER_FAILURE)

        # ConfigObject
        config_object = ConfigObject(schema={"key": ConfigPrimitive(float)})
        config_object.set_name("key")
        config_status = config_object.set_value({"undefiend_key": 2.0})
        print(config_status.message())
        self.assertEqual(config_status.status(), CONFIG_PARSER_FAILURE)

        config_status = config_object.set_value({"key": [1, 2, 3]})
        print(config_status.message())
        self.assertEqual(config_status.status(), CONFIG_PARSER_FAILURE)

        config_status = config_object.set_value([1, 2, 3])
        print(config_status.message())
        self.assertEqual(config_status.status(), CONFIG_PARSER_FAILURE)

        # ConfigPrimitive
        config_primitive = ConfigPrimitive(float)
        config_primitive.set_name("key")
        config_status = config_primitive.set_value("a")
        print(config_status.message())
        self.assertEqual(config_status.status(), CONFIG_PARSER_FAILURE)

    def test_autofill(self):
        args = [
            "model-analyzer",
            "profile",
            "--model-repository",
            "cli_repository",
            "-f",
            "path-to-config-file",
        ]
        yaml_content = """
        profile_models:
        -
            vgg_16_graphdef:
                model_config_parameters:
                    instance_group:
                        -
                            kind: KIND_GPU
                            count: 1

        """
        # Test defaults
        config = self._evaluate_config(args, yaml_content)
        model_configs = config.get_all_config()["profile_models"]
        expected_model_configs = [
            ConfigModelProfileSpec(
                "vgg_16_graphdef",
                parameters={"batch_sizes": [1], "concurrency": [], "request_rate": []},
                objectives={"perf_throughput": 10},
                model_config_parameters={
                    "instance_group": [[{"kind": ["KIND_GPU"], "count": [1]}]]
                },
            )
        ]
        self._assert_equality_of_model_configs(model_configs, expected_model_configs)

        yaml_content = """
objectives:
  perf_throughput: 10
  gpu_used_memory: 5
constraints:
  gpu_used_memory:
    max: 80
profile_models:
  -
    vgg_16_graphdef:
        parameters:
          batch_sizes:
            - 16
            - 32
          concurrency:
            start: 2
            stop: 4
            step: 2
        model_config_parameters:
            instance_group:
                -
                    kind: KIND_GPU
                    count: 1

"""
        # Test autofill objectives and constraints
        config = self._evaluate_config(args, yaml_content)
        model_configs = config.get_all_config()["profile_models"]
        expected_model_configs = [
            ConfigModelProfileSpec(
                "vgg_16_graphdef",
                parameters={
                    "batch_sizes": [16, 32],
                    "concurrency": [2, 4],
                    "request_rate": [],
                },
                objectives={"perf_throughput": 10, "gpu_used_memory": 5},
                constraints={
                    "gpu_used_memory": {
                        "max": 80,
                    }
                },
                model_config_parameters={
                    "instance_group": [[{"kind": ["KIND_GPU"], "count": [1]}]]
                },
            )
        ]
        self._assert_equality_of_model_configs(model_configs, expected_model_configs)

        yaml_content = """
concurrency:
  start: 2
  stop : 4
  step: 2
profile_models:
  -
    vgg_16_graphdef:
        parameters:
          batch_sizes:
            - 16
            - 32
        objectives:
          gpu_used_memory: 10
        constraints:
          perf_latency_p99:
            max: 8000
        model_config_parameters:
            instance_group:
                -
                    kind: KIND_GPU
                    count: 1

"""
        # Test autofill concurrency
        config = self._evaluate_config(args, yaml_content)
        model_configs = config.get_all_config()["profile_models"]
        expected_model_configs = [
            ConfigModelProfileSpec(
                "vgg_16_graphdef",
                parameters={
                    "batch_sizes": [16, 32],
                    "concurrency": [2, 4],
                    "request_rate": [],
                },
                objectives={"gpu_used_memory": 10},
                constraints={"perf_latency_p99": {"max": 8000}},
                model_config_parameters={
                    "instance_group": [[{"kind": ["KIND_GPU"], "count": [1]}]]
                },
            )
        ]
        self._assert_equality_of_model_configs(model_configs, expected_model_configs)

        yaml_content = """
batch_sizes:
  - 16
  - 32
profile_models:
  -
    vgg_16_graphdef:
        parameters:
          concurrency:
            start: 2
            stop : 4
            step: 2
        objectives:
          - gpu_used_memory
        constraints:
          perf_latency_p99:
            max: 8000
        model_config_parameters:
            instance_group:
                -
                    kind: KIND_GPU
                    count: 1

"""
        # Test autofill batch sizes
        config = self._evaluate_config(args, yaml_content)
        model_configs = config.get_all_config()["profile_models"]
        expected_model_configs = [
            ConfigModelProfileSpec(
                "vgg_16_graphdef",
                parameters={
                    "batch_sizes": [16, 32],
                    "concurrency": [2, 4],
                    "request_rate": [],
                },
                objectives={"gpu_used_memory": 10},
                constraints={"perf_latency_p99": {"max": 8000}},
                model_config_parameters={
                    "instance_group": [[{"kind": ["KIND_GPU"], "count": [1]}]]
                },
            )
        ]
        self._assert_equality_of_model_configs(model_configs, expected_model_configs)

        yaml_content = """
objectives:
  perf_throughput: 10
  perf_latency_p99: 5
constraints:
    perf_latency_p99:
      max: 8000
    gpu_used_memory:
      max: 10000
batch_sizes:
  - 16
  - 32
concurrency:
  start: 2
  stop: 4
  step: 2
profile_models:
  -
    vgg_16_graphdef:
        parameters:
          concurrency:
            start: 5
            stop : 7
        objectives:
          - gpu_used_memory
  -
    vgg_19_graphdef:
        parameters:
          batch_sizes:
            - 1
            - 2
        constraints:
          perf_latency_p99:
            max: 8000
"""
        # Test autofill batch sizes
        config = self._evaluate_config(args, yaml_content)
        model_configs = config.get_all_config()["profile_models"]
        expected_model_configs = [
            ConfigModelProfileSpec(
                "vgg_16_graphdef",
                parameters={
                    "batch_sizes": [16, 32],
                    "concurrency": [5, 6, 7],
                    "request_rate": [],
                },
                objectives={"gpu_used_memory": 10},
                constraints={
                    "perf_latency_p99": {"max": 8000},
                    "gpu_used_memory": {"max": 10000},
                },
            ),
            ConfigModelProfileSpec(
                "vgg_19_graphdef",
                parameters={
                    "batch_sizes": [1, 2],
                    "concurrency": [2, 4],
                    "request_rate": [],
                },
                objectives={"perf_throughput": 10, "perf_latency_p99": 5},
                constraints={"perf_latency_p99": {"max": 8000}},
            ),
        ]
        self._assert_equality_of_model_configs(model_configs, expected_model_configs)

        # Test autofill CPU_ONLY. It will only be false if no local gpus are available AND we are not in remote mode
        yaml_content = """
profile_models:
  - vgg_16_graphdef
"""
        for launch_mode in ["remote", "c_api", "docker", "local"]:
            for local_gpus_available in [True, False]:
                new_args = args.copy()
                new_args.extend(["--triton-launch-mode", launch_mode])
                config = self._evaluate_config(
                    new_args, yaml_content, numba_available=local_gpus_available
                )
                model_configs = config.get_all_config()["profile_models"]
                expected_cpu_only = not local_gpus_available and launch_mode != "remote"
                expected_model_configs = [
                    ConfigModelProfileSpec(
                        "vgg_16_graphdef",
                        cpu_only=expected_cpu_only,
                        parameters=self._create_parameters(batch_sizes=[1]),
                        objectives={"perf_throughput": 10},
                    )
                ]
                self._assert_equality_of_model_configs(
                    model_configs, expected_model_configs
                )

    def test_config_shorthands(self):
        """
        test flags like --latency-budget
        """

        for constraint_shorthand in [
            ("--latency-budget", "max", "perf_latency_p99"),
            ("--min-throughput", "min", "perf_throughput"),
        ]:
            args = [
                "model-analyzer",
                "profile",
                "--profile-models",
                "test_model",
                constraint_shorthand[0],
                "40",
                "--model-repository",
                "cli_repository",
            ]
            # check that global and model specific constraints are filled
            yaml_content = ""
            config = self._evaluate_config(args, yaml_content, subcommand="profile")
            self.assertDictEqual(
                config.get_all_config()["constraints"],
                {constraint_shorthand[2]: {constraint_shorthand[1]: 40}},
            )

            self.assertEqual(
                config.get_all_config()["profile_models"][0].constraints(),
                ModelConstraints(
                    {constraint_shorthand[2]: {constraint_shorthand[1]: 40}}
                ),
            )

            # check that model specific constraints are appended to
            args = [
                "model-analyzer",
                "profile",
                constraint_shorthand[0],
                "40",
                "-f",
                "path-to-config-file",
                "--model-repository",
                "cli_repository",
            ]
            yaml_content = """
            profile_models:
                test_model:
                    constraints:
                        gpu_used_memory:
                            max : 100
            """
            config = self._evaluate_config(args, yaml_content, subcommand="profile")
            self.assertDictEqual(
                config.get_all_config()["constraints"],
                {constraint_shorthand[2]: {constraint_shorthand[1]: 40}},
            )
            self.assertEqual(
                config.get_all_config()["profile_models"][0].constraints(),
                ModelConstraints(
                    {
                        constraint_shorthand[2]: {constraint_shorthand[1]: 40},
                        "gpu_used_memory": {"max": 100},
                    }
                ),
            )

            # check that model specific constraints are replaced
            yaml_content = f"""
            profile_models:
                test_model:
                    constraints:
                        {constraint_shorthand[2]}:
                            {constraint_shorthand[1]} : 100
            """
            config = self._evaluate_config(args, yaml_content, subcommand="profile")
            self.assertEqual(
                config.get_all_config()["profile_models"][0].constraints(),
                ModelConstraints(
                    {constraint_shorthand[2]: {constraint_shorthand[1]: 40}}
                ),
            )

            # check that global constraints are appended to
            yaml_content = """
            profile_models: test_model
            constraints:
                gpu_used_memory:
                    max : 100
            """

            config = self._evaluate_config(args, yaml_content, subcommand="profile")
            self.assertDictEqual(
                config.get_all_config()["constraints"],
                {
                    constraint_shorthand[2]: {constraint_shorthand[1]: 40},
                    "gpu_used_memory": {"max": 100},
                },
            )

            # check that global constraints are replaced
            yaml_content = f"""
            profile_models: test_model
            constraints:
                {constraint_shorthand[2]}:
                    {constraint_shorthand[1]} : 100
            """
            config = self._evaluate_config(args, yaml_content, subcommand="profile")
            self.assertDictEqual(
                config.get_all_config()["constraints"],
                {constraint_shorthand[2]: {constraint_shorthand[1]: 40}},
            )

    def test_triton_server_flags(self):
        args = [
            "model-analyzer",
            "profile",
            "--model-repository",
            "cli_repository",
            "-f",
            "path-to-config-file",
        ]
        yaml_content = """
            profile_models: model1, model2
            triton_server_flags:
                strict-model-config: false
                backend-config: test_backend_config
            """
        config = self._evaluate_config(args, yaml_content)
        self.assertDictEqual(
            config.get_all_config()["triton_server_flags"],
            {"strict-model-config": "False", "backend-config": "test_backend_config"},
        )

        yaml_content = """
            profile_models: model1, model2
            triton_server_flags:
                disallowed-config-option: some_value
                backend-config: test_backend_config
            """
        with self.assertRaises(TritonModelAnalyzerException):
            self._evaluate_config(args, yaml_content)

        yaml_content = """
            profile_models:
                model1:
                    triton_server_flags:
                        strict_model_config: false
                        backend_config: test_backend_config
            """
        config = self._evaluate_config(args, yaml_content)
        self.assertDictEqual(
            config.get_all_config()["profile_models"][0].triton_server_flags(),
            {"strict_model_config": "False", "backend_config": "test_backend_config"},
        )

    def test_triton_server_environment(self):
        args = [
            "model-analyzer",
            "profile",
            "--model-repository",
            "cli_repository",
            "-f",
            "path-to-config-file",
        ]
        yaml_content = """
            profile_models: model1, model2
            triton_server_environment:
                LD_PRELOAD: libtest.so
                LD_LIBRARY_PATH: /path/to/test/lib
            """
        config = self._evaluate_config(args, yaml_content)
        self.assertDictEqual(
            config.get_all_config()["triton_server_environment"],
            {"LD_PRELOAD": "libtest.so", "LD_LIBRARY_PATH": "/path/to/test/lib"},
        )

        yaml_content = """
            profile_models:
                model1:
                    triton_server_environment:
                        LD_PRELOAD: libtest.so
                        LD_LIBRARY_PATH: /path/to/test/lib
            """
        config = self._evaluate_config(args, yaml_content)
        self.assertDictEqual(
            config.get_all_config()["profile_models"][0].triton_server_environment(),
            {"LD_PRELOAD": "libtest.so", "LD_LIBRARY_PATH": "/path/to/test/lib"},
        )

    def test_triton_docker_args(self):
        args = [
            "model-analyzer",
            "profile",
            "--model-repository",
            "cli_repository",
            "-f",
            "path-to-config-file",
            "--triton-launch-mode",
            "docker",
        ]
        yaml_content = """
            profile_models: model1, model2
            triton_docker_args:
                image: foo
                command: goo
            """
        config = self._evaluate_config(args, yaml_content)
        self.assertDictEqual(
            config.get_all_config()["triton_docker_args"],
            {"image": "foo", "command": "goo"},
        )

        yaml_content = """
            profile_models:
                model1:
                    triton_docker_args:
                        image: foo
                        command: goo
            """
        config = self._evaluate_config(args, yaml_content)
        self.assertDictEqual(
            config.get_all_config()["profile_models"][0].triton_docker_args(),
            {"image": "foo", "command": "goo"},
        )

        # This is testing that all options can be parsed by the YAML
        yaml_content = """
            profile_models: model1, model2
            triton_docker_args:
                image: abc
                command: abc
                auto_remove: true
                blkio_weight_device: ['abc']
                blkio_weight: 1
                cap_add: ['abc']
                cap_drop: ['abc']
                cgroup_parent: abc
                cgroupns: abc
                cpu_count: 1
                cpu_percent: 1
                cpu_period: 1
                cpu_quota: 1
                cpu_rt_period: 1
                cpu_shares: 1
                cpuset_cpus: abc
                cpuset_mems: abc
                detach: true
                domainname: abc
                entrypoint: abc
                environment: ['abc']
                hostname: abc
                init: true
                init_path: abc
                ipc_mode: abc
                isolation: abc
                kernel_memory: abc
                labels: abc
                mac_address: abc
                mem_limit: abc
                mem_reservation: abc
                memswap_limit: abc
                name: abc
                nano_cpus: 1
                network: abc
                network_disabled: true
                network_mode: abc
                oom_kill_disable: true
                oom_score_adj: 1
                pid_mode: abc
                pids_limit: 1
                platform: abc
                privileged: true
                publish_all_ports: true
                remove: true
                runtime: abc
                shm_size: abc
                stdin_open: true
                stdout: true
                stderr: true
                stop_signal: abc
                stream: true
                tty: true
                use_config_proxy: true
                user: abc
                userns_mode: abc
                uts_mode: abc
                version: abc
                volume_driver: abc
                volumes: ['abc']
                working_dir: abc
            """
        _ = self._evaluate_config(args, yaml_content)

    def test_report_configs(self):
        args = ["model-analyzer", "report", "--report-model-configs", "test-model"]
        yaml_content = ""
        config = self._evaluate_config(args, yaml_content, subcommand="report")
        self.assertEqual(
            config.get_all_config()["report_model_configs"][0].model_config_name(),
            "test-model",
        )

        # check output format
        args = [
            "model-analyzer",
            "report",
            "--report-model-configs",
            "test-model",
            "--output-formats",
            "pdf",
        ]
        config = self._evaluate_config(args, yaml_content, subcommand="report")
        self.assertEqual(config.get_all_config()["output_formats"], ["pdf"])
        args = [
            "model-analyzer",
            "report",
            "--report-model-configs",
            "test-model",
            "--output-formats",
            "pdf,csv,svg",
        ]
        config = self._evaluate_config(args, yaml_content, subcommand="report")
        self.assertEqual(
            config.get_all_config()["output_formats"], ["pdf", "csv", "svg"]
        )

        # Check yaml report model config
        args = ["model-analyzer", "report", "-f", "path-to-config-file"]
        yaml_content = """
        report_model_configs:
            - test_model_config_0
        output_formats:
          - pdf
          - csv
          - png
        """

        config = self._evaluate_config(args, yaml_content, subcommand="report")
        self.assertEqual(
            config.get_all_config()["report_model_configs"][0].model_config_name(),
            "test_model_config_0",
        )
        self.assertEqual(
            config.get_all_config()["output_formats"], ["pdf", "csv", "png"]
        )

        # Check plots
        args = ["model-analyzer", "report", "-f", "path-to-config-file"]
        yaml_content = """
        report_model_configs:
           - test_model_config_0
           - test_model_config_1
        plots:
            throughput_v_latency:
                title: Throughput vs. Latency
                x_axis: perf_latency_p99
                y_axis: perf_throughput
                monotonic: True
        """

        config = self._evaluate_config(args, yaml_content, subcommand="report")
        self.assertEqual(
            config.get_all_config()["report_model_configs"][0].model_config_name(),
            "test_model_config_0",
        )
        self.assertEqual(
            config.get_all_config()["report_model_configs"][1].model_config_name(),
            "test_model_config_1",
        )
        expected_config_plot = {
            "throughput_v_latency": {
                "title": "Throughput vs. Latency",
                "x_axis": "perf_latency_p99",
                "y_axis": "perf_throughput",
                "monotonic": True,
            }
        }
        config_plot = config.get_all_config()["plots"][0]
        config_plot_dict = {
            config_plot.name(): {
                "title": config_plot.title(),
                "x_axis": config_plot.x_axis(),
                "y_axis": config_plot.y_axis(),
                "monotonic": config_plot.monotonic(),
            }
        }
        self.assertDictEqual(config_plot_dict, expected_config_plot)
        for report_model_config in config.report_model_configs:
            config_plot = report_model_config.plots()[0]
            config_plot_dict = {
                config_plot.name(): {
                    "title": config_plot.title(),
                    "x_axis": config_plot.x_axis(),
                    "y_axis": config_plot.y_axis(),
                    "monotonic": config_plot.monotonic(),
                }
            }
            self.assertDictEqual(config_plot_dict, expected_config_plot)

        # Check individual plots
        yaml_content = """
        report_model_configs:
            test_model_config_0:
                plots:
                  model_specific_throughput_v_latency:
                    title: model specific title
                    x_axis: perf_latency_p99
                    y_axis: perf_throughput
                    monotonic: True
        plots:
            throughput_v_latency:
                title: Throughput vs. Latency
                x_axis: perf_latency_p99
                y_axis: perf_throughput
                monotonic: True
        """

        config = self._evaluate_config(args, yaml_content, subcommand="report")
        global_config_plot = config.get_all_config()["plots"][0]
        global_config_plot_dict = {
            config_plot.name(): {
                "title": global_config_plot.title(),
                "x_axis": global_config_plot.x_axis(),
                "y_axis": global_config_plot.y_axis(),
                "monotonic": global_config_plot.monotonic(),
            }
        }
        self.assertDictEqual(expected_config_plot, global_config_plot_dict)
        model_specific_plot = config.get_all_config()["report_model_configs"][
            0
        ].plots()[0]
        self.assertEqual(
            model_specific_plot.name(), "model_specific_throughput_v_latency"
        )
        self.assertEqual(model_specific_plot.title(), "model specific title")

    def test_path_validation(self):
        # Test parent path validator
        args = [
            "model-analyzer",
            "profile",
            "--model-repository",
            "/",
            "-f",
            "path-to-config-file",
        ]
        yaml_content = """
        checkpoint_directory: /test
        profile_models:
            - model1
            - model2
        """

        self._evaluate_config(args, yaml_content, subcommand="profile")

        self.mock_os.set_os_path_exists_return_value(False)
        with self.assertRaises(TritonModelAnalyzerException):
            self._evaluate_config(args, yaml_content, subcommand="profile")
        self.mock_os.set_os_path_exists_return_value(True)

        # Test file path validator
        yaml_content = """
        triton_install_path: /opt/triton-model-analyzer/tests/test_config.py
        profile_models:
            - model1
            - model2
        """

        self._evaluate_config(args, yaml_content, subcommand="profile")

        self.mock_os.set_os_path_exists_return_value(False)
        with self.assertRaises(TritonModelAnalyzerException):
            self._evaluate_config(args, yaml_content, subcommand="profile")
        self.mock_os.set_os_path_exists_return_value(True)

        args = [
            "model-analyzer",
            "profile",
            "-f",
            "path-to-config-file",
            "--model-repository",
            "cli_repository",
        ]
        yaml_content = """
        export_path: /opt/triton-model-analyzer/tests
        profile_models:
            - model1
            - model2
        """

        self._evaluate_config(args, yaml_content, subcommand="profile")

        self.mock_os.set_os_path_exists_return_value(False)
        with self.assertRaises(TritonModelAnalyzerException):
            self._evaluate_config(args, yaml_content, subcommand="profile")
        self.mock_os.set_os_path_exists_return_value(True)

        # Test the binary path validator
        args = [
            "model-analyzer",
            "profile",
            "--model-repository",
            "/",
            "-f",
            "path-to-config-file",
        ]
        yaml_content = """
        triton_server_path: tritonserver
        profile_models:
            - model1
            - model2
        """

        self._evaluate_config(args, yaml_content, subcommand="profile")

        self.mock_os.set_os_path_exists_return_value(False)
        with self.assertRaises(TritonModelAnalyzerException):
            self._evaluate_config(args, yaml_content, subcommand="profile")
        self.mock_os.set_os_path_exists_return_value(True)

    def test_copy(self):
        """
        Test that deepcopy works correctly
        """
        args = [
            "model-analyzer",
            "profile",
            "--model-repository",
            "cli_repository",
            "-f",
            "path-to-config-file",
            "--profile-models",
            "vgg11",
        ]
        yaml_content = "model_repository: yaml_repository"
        configA = self._evaluate_config(args, yaml_content)

        configB = deepcopy(configA)

        self._assert_equality_of_model_configs(
            configA.get_all_config()["profile_models"],
            configB.get_all_config()["profile_models"],
        )

        self.assertEqual(configA.run_config_search_mode, configB.run_config_search_mode)

        configB.run_config_search_mode = "quick"

        self.assertNotEqual(
            configA.run_config_search_mode, configB.run_config_search_mode
        )

    def test_multi_model_search_mode(self):
        """
        Test that multi-model is only run in quick
        """
        args = [
            "model-analyzer",
            "profile",
            "--model-repository",
            "cli-repository",
            "--profile-models",
            "test_modelA,test_modelB",
            "--run-config-profile-models-concurrently-enable",
        ]

        yaml_content = ""

        # Tests the case where no search mode is specified (default is brute)
        self._evaluate_config(args, yaml_content, subcommand="profile")

        # Brute should fail
        new_args = list(args)
        new_args.append("--run-config-search-mode")
        new_args.append("brute")

        with self.assertRaises(TritonModelAnalyzerException):
            self._evaluate_config(new_args, yaml_content, subcommand="profile")

        # Quick should pass
        new_args = list(args)
        new_args.append("--run-config-search-mode")
        new_args.append("quick")

        self._evaluate_config(new_args, yaml_content, subcommand="profile")

    def test_quick_search_mode(self):
        """
        Test that only legal options are specified in quick search
        """
        args = [
            "model-analyzer",
            "profile",
            "--model-repository",
            "cli-repository",
            "--profile-models",
            "test_modelA",
            "--run-config-search-mode",
            "quick",
        ]

        yaml_content = ""

        self._evaluate_config(args, yaml_content)

        self._test_quick_search_with_rcs(
            args,
            yaml_content,
            "--run-config-search-disable",
            use_value=False,
            use_list=False,
        )
        self._test_quick_search_with_rcs(
            args, yaml_content, "--batch-sizes", use_value=False
        )
        self._test_quick_search_with_rcs(
            args, yaml_content, "--concurrency", use_value=False
        )

    def test_quick_search_model_specific(self):
        """
        Test for illegal model specific options in the YAML during quick search
        """
        args = [
            "model-analyzer",
            "profile",
            "--model-repository",
            "cli-repository",
            "--run-config-search-mode",
            "quick",
            "-f",
            "path-to-config-file",
        ]

        yaml_content = """
        profile_models:
          model_1:
            parameters:
              concurrency: 1,2,4,128
        """

        with self.assertRaises(TritonModelAnalyzerException):
            self._evaluate_config(args, yaml_content, subcommand="profile")

        yaml_content = """
        profile_models:
          model_1:
            parameters:
              batch size: 1,2,4,128
        """

        with self.assertRaises(TritonModelAnalyzerException):
            self._evaluate_config(args, yaml_content, subcommand="profile")

        yaml_content = """
        profile_models:
          model_1:
            model_config_parameters:
              max_batch_size: 2
        """

        with self.assertRaises(TritonModelAnalyzerException):
            self._evaluate_config(args, yaml_content, subcommand="profile")

        yaml_content = """
        profile_models:
          model_1:
            model_config_parameters:
              instance_group:
                - kind: KIND_GPU
        """

        with self.assertRaises(TritonModelAnalyzerException):
            self._evaluate_config(args, yaml_content, subcommand="profile")

        yaml_content = """
        profile_models:
          model_1:
            model_config_parameters:
               optimization:
                 execution_accelerators:
                   gpu_execution_accelerator:
                     - name: tensorrt
                       parameters:
                       precision_mode: [FP16, FP32]
        """

        with self.assertRaises(TritonModelAnalyzerException):
            self._evaluate_config(args, yaml_content, subcommand="profile")

    def test_model_weightings_profile(self):
        """
        Test that weightings can be specified only per model in the YAML
        """
        args = [
            "model-analyzer",
            "profile",
            "--model-repository",
            "cli-repository",
            "--run-config-search-mode",
            "quick",
            "-f",
            "path-to-config-file",
        ]

        # Global weighting - illegal
        yaml_content = """
        weighting: 3

        profile_models:
          resnet50_libtorch:
            objectives:
              perf_latency_p99: 1
          add_sub:
            objectives:
              perf_throughput: 1
        """

        with self.assertRaises(TritonModelAnalyzerException):
            self._evaluate_config(args, yaml_content, subcommand="profile")

        # Model specific weighting - legal
        yaml_content = """
        profile_models:
          resnet50_libtorch:
            weighting: 3
            objectives:
              perf_latency_p99: 1
          add_sub:
            weighting: 1
            objectives:
              perf_throughput: 1
        """

        self._evaluate_config(args, yaml_content, subcommand="profile")

    def _test_quick_search_with_rcs(
        self,
        args: Namespace,
        yaml_content: Optional[Dict[str, List]],
        rcs_string: str,
        use_value: bool = True,
        use_list: bool = True,
    ) -> None:
        """
        Tests that run-config-search options raise exceptions
        in quick search mode
        """
        new_args = list(args)
        new_args.append(rcs_string)
        if use_value:
            new_args.append("1")
        elif use_list:
            new_args.append(["1", "2", "4"])
        with self.assertRaises(TritonModelAnalyzerException):
            self._evaluate_config(new_args, yaml_content, subcommand="profile")

    def test_bls_composing_models(self):
        """
        Test that BLS composing models can be specified
        """
        args = [
            "model-analyzer",
            "profile",
            "--model-repository",
            "cli-repository",
            "--run-config-search-mode",
            "quick",
            "--profile-models",
            "modelA",
            "--bls-composing-models",
            "modelA,modelB",
        ]
        yaml_content = ""

        self._evaluate_config(args, yaml_content)

        args = [
            "model-analyzer",
            "profile",
            "--model-repository",
            "cli-repository",
            "--run-config-search-mode",
            "quick",
            "--profile-models",
            "modelA",
        ]

        yaml_content = """
        bls_composing_models: modelA,modelB
        """

        self._evaluate_config(args, yaml_content)

    def test_bls_illegal_config_combinations(self):
        """
        Test BLS illegal config combinations
        """
        # BLS with brute search mode
        args = [
            "model-analyzer",
            "profile",
            "--model-repository",
            "cli-repository",
            "--run-config-search-mode",
            "brute",
            "--profile-models",
            "modelA",
            "--bls-composing-models",
            "modelA,modelB",
        ]
        yaml_content = ""

        with self.assertRaises(TritonModelAnalyzerException):
            self._evaluate_config(args, yaml_content)

        # BLS with multiple models
        args = [
            "model-analyzer",
            "profile",
            "--model-repository",
            "cli-repository",
            "--run-config-search-mode",
            "quick",
            "--profile-models",
            "modelA,modelB",
            "--bls-composing-models",
            "modelA,modelB",
        ]
        yaml_content = ""

        with self.assertRaises(TritonModelAnalyzerException):
            self._evaluate_config(args, yaml_content)

        # BLS with concurrent search mode
        args = [
            "model-analyzer",
            "profile",
            "--model-repository",
            "cli-repository",
            "--run-config-search-mode",
            "quick",
            "--profile-models",
            "modelA",
            "--bls-composing-models",
            "modelA,modelB",
            "--run-config-profile-models-concurrently-enable",
        ]
        yaml_content = ""

        with self.assertRaises(TritonModelAnalyzerException):
            self._evaluate_config(args, yaml_content)

    def test_concurrency_rate_request_config_combinations(self):
        """
        Test for concurrency with rate request conflicts
        """
        base_args = [
            "model-analyzer",
            "profile",
            "--model-repository",
            "cli-repository",
            "--profile-models",
            "modelA",
            "-c",
            "1,2,3",
        ]
        yaml_content = ""

        self._test_request_rate_config_conflicts(base_args, yaml_content)

    def test_config_search_min_rate_request_config_combinations(self):
        """
        Test for concurrency min request with rate request conflicts
        """
        base_args = [
            "model-analyzer",
            "profile",
            "--model-repository",
            "cli-repository",
            "--profile-models",
            "modelA",
            "--run-config-search-min-concurrency",
            "1",
        ]
        yaml_content = ""

        self._test_request_rate_config_conflicts(base_args, yaml_content)

    def test_config_search_max_rate_request_config_combinations(self):
        """
        Test for concurrency max request with rate request conflicts
        """
        base_args = [
            "model-analyzer",
            "profile",
            "--model-repository",
            "cli-repository",
            "--profile-models",
            "modelA",
            "--run-config-search-max-concurrency",
            "1",
        ]
        yaml_content = ""

        self._test_request_rate_config_conflicts(base_args, yaml_content)

    def test_mixing_request_rate_and_concurrency(self):
        """
        Test that mixing models parameters of request-rate and concurrency is illegal
        """
        args = [
            "model-analyzer",
            "profile",
            "--model-repository",
            "cli-repository",
            "-f",
            "path-to-config-file",
        ]

        yaml_content = """
        profile_models:

          resnet50_libtorch:
            parameters:
              concurrency: 4
          add_sub:
            parameters:
              request_rate: 16
        """

        with self.assertRaises(TritonModelAnalyzerException):
            self._evaluate_config(args, yaml_content, subcommand="profile")

    def _test_request_rate_config_conflicts(
        self, base_args: List[Any], yaml_content: str
    ) -> None:
        self._test_arg_conflict(
            base_args, yaml_content, ["--request-rate-search-enable"]
        )
        self._test_arg_conflict(base_args, yaml_content, ["--request-rate", "1,2,3"])
        self._test_arg_conflict(
            base_args, yaml_content, ["--run-config-search-min-request-rate", "1"]
        )
        self._test_arg_conflict(
            base_args, yaml_content, ["--run-config-search-max-request-rate", "1"]
        )

    def _test_arg_conflict(
        self, base_args: List[Any], yaml_content: str, new_args: List[Any]
    ) -> None:
        args = base_args.copy() + new_args
        with self.assertRaises(TritonModelAnalyzerException):
            self._evaluate_config(args, yaml_content)


if __name__ == "__main__":
    unittest.main()
