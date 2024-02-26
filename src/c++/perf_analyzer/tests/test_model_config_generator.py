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

from model_analyzer.config.generate.base_model_config_generator import (
    BaseModelConfigGenerator,
)
from model_analyzer.config.generate.model_config_generator_factory import (
    ModelConfigGeneratorFactory,
)
from model_analyzer.config.generate.model_profile_spec import ModelProfileSpec
from model_analyzer.config.generate.model_variant_name_manager import (
    ModelVariantNameManager,
)
from model_analyzer.model_analyzer_exceptions import TritonModelAnalyzerException
from model_analyzer.triton.model.model_config import ModelConfig

from .common import test_result_collector as trc
from .common.test_utils import construct_run_config_measurement, evaluate_mock_config
from .mocks.mock_model_config import MockModelConfig
from .mocks.mock_os import MockOSMethods


class TestModelConfigGenerator(trc.TestResultCollector):
    def __init__(self, methodname):
        super().__init__(methodname)
        self._fake_throughput = 1

    def test_direct_no_params(self):
        """
        Test direct modes with no model_config_parameters specified

        It will just sweep instance count and max_batch_size (with dynamic batching on),
        and default config (empty dict) will be included
        """

        # yapf: disable
        yaml_str = ("""
            profile_models:
                - my-model
            """)

        expected_configs = [
            {'dynamic_batching': {}, 'max_batch_size': 1, 'instance_group': [{'count': 1, 'kind': 'KIND_GPU'}]},
            {'dynamic_batching': {}, 'max_batch_size': 1, 'instance_group': [{'count': 2, 'kind': 'KIND_GPU'}]},
            {'dynamic_batching': {}, 'max_batch_size': 1, 'instance_group': [{'count': 3, 'kind': 'KIND_GPU'}]},
            {'dynamic_batching': {}, 'max_batch_size': 1, 'instance_group': [{'count': 4, 'kind': 'KIND_GPU'}]},
            {'dynamic_batching': {}, 'max_batch_size': 1, 'instance_group': [{'count': 5, 'kind': 'KIND_GPU'}]},
            {'dynamic_batching': {}, 'max_batch_size': 2, 'instance_group': [{'count': 1, 'kind': 'KIND_GPU'}]},
            {'dynamic_batching': {}, 'max_batch_size': 2, 'instance_group': [{'count': 2, 'kind': 'KIND_GPU'}]},
            {'dynamic_batching': {}, 'max_batch_size': 2, 'instance_group': [{'count': 3, 'kind': 'KIND_GPU'}]},
            {'dynamic_batching': {}, 'max_batch_size': 2, 'instance_group': [{'count': 4, 'kind': 'KIND_GPU'}]},
            {'dynamic_batching': {}, 'max_batch_size': 2, 'instance_group': [{'count': 5, 'kind': 'KIND_GPU'}]},
            {'dynamic_batching': {}, 'max_batch_size': 4, 'instance_group': [{'count': 1, 'kind': 'KIND_GPU'}]},
            {'dynamic_batching': {}, 'max_batch_size': 4, 'instance_group': [{'count': 2, 'kind': 'KIND_GPU'}]},
            {'dynamic_batching': {}, 'max_batch_size': 4, 'instance_group': [{'count': 3, 'kind': 'KIND_GPU'}]},
            {'dynamic_batching': {}, 'max_batch_size': 4, 'instance_group': [{'count': 4, 'kind': 'KIND_GPU'}]},
            {'dynamic_batching': {}, 'max_batch_size': 4, 'instance_group': [{'count': 5, 'kind': 'KIND_GPU'}]},
            {'dynamic_batching': {}, 'max_batch_size': 8, 'instance_group': [{'count': 1, 'kind': 'KIND_GPU'}]},
            {'dynamic_batching': {}, 'max_batch_size': 8, 'instance_group': [{'count': 2, 'kind': 'KIND_GPU'}]},
            {'dynamic_batching': {}, 'max_batch_size': 8, 'instance_group': [{'count': 3, 'kind': 'KIND_GPU'}]},
            {'dynamic_batching': {}, 'max_batch_size': 8, 'instance_group': [{'count': 4, 'kind': 'KIND_GPU'}]},
            {'dynamic_batching': {}, 'max_batch_size': 8, 'instance_group': [{'count': 5, 'kind': 'KIND_GPU'}]},
            {'dynamic_batching': {}, 'max_batch_size': 16, 'instance_group': [{'count': 1, 'kind': 'KIND_GPU'}]},
            {'dynamic_batching': {}, 'max_batch_size': 16, 'instance_group': [{'count': 2, 'kind': 'KIND_GPU'}]},
            {'dynamic_batching': {}, 'max_batch_size': 16, 'instance_group': [{'count': 3, 'kind': 'KIND_GPU'}]},
            {'dynamic_batching': {}, 'max_batch_size': 16, 'instance_group': [{'count': 4, 'kind': 'KIND_GPU'}]},
            {'dynamic_batching': {}, 'max_batch_size': 16, 'instance_group': [{'count': 5, 'kind': 'KIND_GPU'}]},
            {'dynamic_batching': {}, 'max_batch_size': 32, 'instance_group': [{'count': 1, 'kind': 'KIND_GPU'}]},
            {'dynamic_batching': {}, 'max_batch_size': 32, 'instance_group': [{'count': 2, 'kind': 'KIND_GPU'}]},
            {'dynamic_batching': {}, 'max_batch_size': 32, 'instance_group': [{'count': 3, 'kind': 'KIND_GPU'}]},
            {'dynamic_batching': {}, 'max_batch_size': 32, 'instance_group': [{'count': 4, 'kind': 'KIND_GPU'}]},
            {'dynamic_batching': {}, 'max_batch_size': 32, 'instance_group': [{'count': 5, 'kind': 'KIND_GPU'}]},
            {'dynamic_batching': {}, 'max_batch_size': 64, 'instance_group': [{'count': 1, 'kind': 'KIND_GPU'}]},
            {'dynamic_batching': {}, 'max_batch_size': 64, 'instance_group': [{'count': 2, 'kind': 'KIND_GPU'}]},
            {'dynamic_batching': {}, 'max_batch_size': 64, 'instance_group': [{'count': 3, 'kind': 'KIND_GPU'}]},
            {'dynamic_batching': {}, 'max_batch_size': 64, 'instance_group': [{'count': 4, 'kind': 'KIND_GPU'}]},
            {'dynamic_batching': {}, 'max_batch_size': 64, 'instance_group': [{'count': 5, 'kind': 'KIND_GPU'}]},
            {'dynamic_batching': {}, 'max_batch_size': 128, 'instance_group': [{'count': 1, 'kind': 'KIND_GPU'}]},
            {'dynamic_batching': {}, 'max_batch_size': 128, 'instance_group': [{'count': 2, 'kind': 'KIND_GPU'}]},
            {'dynamic_batching': {}, 'max_batch_size': 128, 'instance_group': [{'count': 3, 'kind': 'KIND_GPU'}]},
            {'dynamic_batching': {}, 'max_batch_size': 128, 'instance_group': [{'count': 4, 'kind': 'KIND_GPU'}]},
            {'dynamic_batching': {}, 'max_batch_size': 128, 'instance_group': [{'count': 5, 'kind': 'KIND_GPU'}]}
        ]
        # yapf: enable

        self._run_and_test_model_config_generator(yaml_str, expected_configs)

    def test_direct_no_params_search_disable(self):
        """
        Test direct mode with no model_config_parameters specified and run_search disabled

        This will just return a single empty config, since there are no parameters to combine
        """

        # yapf: disable
        yaml_str = ("""
            run_config_search_disable: True
            profile_models:
                - my-model
            """)

        expected_configs = [{'max_batch_size': 8}]
        # yapf: enable

        self._run_and_test_model_config_generator(yaml_str, expected_configs)

    def test_direct_yes_params_search_disable(self):
        """
        Test direct modes with model_config_parameters specified and run_search disabled

        This will just combine all model_config_parameters
        """

        # yapf: disable
        yaml_str = ("""
            run_config_search_max_instance_count: 16
            run_config_search_disable: True
            profile_models:
                test_model:
                    model_config_parameters:
                        max_batch_size: [1,4,16]
                        instance_group:
                        -
                            kind: KIND_GPU
                            count: [1,2]
            """)

        expected_configs = [
            {'instance_group': [{'count': 1, 'kind': 'KIND_GPU'}], 'max_batch_size': 1},
            {'instance_group': [{'count': 1, 'kind': 'KIND_GPU'}], 'max_batch_size': 4},
            {'instance_group': [{'count': 1, 'kind': 'KIND_GPU'}], 'max_batch_size': 16},
            {'instance_group': [{'count': 2, 'kind': 'KIND_GPU'}], 'max_batch_size': 1},
            {'instance_group': [{'count': 2, 'kind': 'KIND_GPU'}], 'max_batch_size': 4},
            {'instance_group': [{'count': 2, 'kind': 'KIND_GPU'}], 'max_batch_size': 16}
        ]
        # yapf: enable

        self._run_and_test_model_config_generator(yaml_str, expected_configs)

    def test_run_config_search_options(self):
        """
        Test that ModelConfigGenerator will honor run_config_search_max_instance_count
        and run_config_search_max_model_batch_size and run_config_search_min_model_batch_size
        """

        # yapf: disable
        yaml_str = ("""
            run_config_search_max_instance_count: 3
            run_config_search_min_model_batch_size: 2
            run_config_search_max_model_batch_size: 16
            profile_models:
                - my-model
            """)

        expected_configs = [
            {'dynamic_batching': {}, 'max_batch_size': 2, 'instance_group': [{'count': 1, 'kind': 'KIND_GPU'}]},
            {'dynamic_batching': {}, 'max_batch_size': 2, 'instance_group': [{'count': 2, 'kind': 'KIND_GPU'}]},
            {'dynamic_batching': {}, 'max_batch_size': 2, 'instance_group': [{'count': 3, 'kind': 'KIND_GPU'}]},
            {'dynamic_batching': {}, 'max_batch_size': 4, 'instance_group': [{'count': 1, 'kind': 'KIND_GPU'}]},
            {'dynamic_batching': {}, 'max_batch_size': 4, 'instance_group': [{'count': 2, 'kind': 'KIND_GPU'}]},
            {'dynamic_batching': {}, 'max_batch_size': 4, 'instance_group': [{'count': 3, 'kind': 'KIND_GPU'}]},
            {'dynamic_batching': {}, 'max_batch_size': 8, 'instance_group': [{'count': 1, 'kind': 'KIND_GPU'}]},
            {'dynamic_batching': {}, 'max_batch_size': 8, 'instance_group': [{'count': 2, 'kind': 'KIND_GPU'}]},
            {'dynamic_batching': {}, 'max_batch_size': 8, 'instance_group': [{'count': 3, 'kind': 'KIND_GPU'}]},
            {'dynamic_batching': {}, 'max_batch_size': 16, 'instance_group': [{'count': 1, 'kind': 'KIND_GPU'}]},
            {'dynamic_batching': {}, 'max_batch_size': 16, 'instance_group': [{'count': 2, 'kind': 'KIND_GPU'}]},
            {'dynamic_batching': {}, 'max_batch_size': 16, 'instance_group': [{'count': 3, 'kind': 'KIND_GPU'}]}
        ]
        # yapf: enable

        self._run_and_test_model_config_generator(yaml_str, expected_configs)

    def test_run_config_search_min_instance_counts(self):
        """
        Test that ModelConfigGenerator will honor run_config_search_min_instance_count
        """

        # yapf: disable
        yaml_str = ("""
            run_config_search_min_instance_count: 2
            run_config_search_max_instance_count: 3
            run_config_search_min_model_batch_size: 2
            run_config_search_max_model_batch_size: 8
            profile_models:
                - my-model
            """)

        expected_configs = [
            {'dynamic_batching': {}, 'max_batch_size': 2, 'instance_group': [{'count': 2, 'kind': 'KIND_GPU'}]},
            {'dynamic_batching': {}, 'max_batch_size': 2, 'instance_group': [{'count': 3, 'kind': 'KIND_GPU'}]},
            {'dynamic_batching': {}, 'max_batch_size': 4, 'instance_group': [{'count': 2, 'kind': 'KIND_GPU'}]},
            {'dynamic_batching': {}, 'max_batch_size': 4, 'instance_group': [{'count': 3, 'kind': 'KIND_GPU'}]},
            {'dynamic_batching': {}, 'max_batch_size': 8, 'instance_group': [{'count': 2, 'kind': 'KIND_GPU'}]},
            {'dynamic_batching': {}, 'max_batch_size': 8, 'instance_group': [{'count': 3, 'kind': 'KIND_GPU'}]}
        ]
        # yapf: enable

        self._run_and_test_model_config_generator(yaml_str, expected_configs)

    def test_non_power_of_two_max_batch_size(self):
        """
        Test that ModelConfigGenerator will correctly sweep max_batch_size with
        input values that aren't a power of 2
        """

        # yapf: disable
        yaml_str = ("""
            run_config_search_max_instance_count: 2
            run_config_search_min_model_batch_size: 3
            run_config_search_max_model_batch_size: 15
            profile_models:
                - my-model
            """)

        expected_configs = [
            {'dynamic_batching': {}, 'max_batch_size': 3, 'instance_group': [{'count': 1, 'kind': 'KIND_GPU'}]},
            {'dynamic_batching': {}, 'max_batch_size': 3, 'instance_group': [{'count': 2, 'kind': 'KIND_GPU'}]},
            {'dynamic_batching': {}, 'max_batch_size': 6, 'instance_group': [{'count': 1, 'kind': 'KIND_GPU'}]},
            {'dynamic_batching': {}, 'max_batch_size': 6, 'instance_group': [{'count': 2, 'kind': 'KIND_GPU'}]},
            {'dynamic_batching': {}, 'max_batch_size': 12, 'instance_group': [{'count': 1, 'kind': 'KIND_GPU'}]},
            {'dynamic_batching': {}, 'max_batch_size': 12, 'instance_group': [{'count': 2, 'kind': 'KIND_GPU'}]}
        ]
        # yapf: enable

        self._run_and_test_model_config_generator(yaml_str, expected_configs)

    def test_direct_yes_params_specified(self):
        """
        Test direct mode with model_config_parameters specified

        It will combine all legal combinations of config values, and
        default config (None) will be included
        """

        # yapf: disable
        yaml_str = ("""
            run_config_search_max_instance_count: 16
            run_config_search_disable: False
            profile_models:
                test_model:
                    model_config_parameters:
                        max_batch_size: [1,4,16]
                        instance_group:
                        -
                            kind: KIND_GPU
                            count: [1,2]
            """)

        expected_configs = [
            {'instance_group': [{'count': 1, 'kind': 'KIND_GPU'}], 'max_batch_size': 1},
            {'instance_group': [{'count': 1, 'kind': 'KIND_GPU'}], 'max_batch_size': 4},
            {'instance_group': [{'count': 1, 'kind': 'KIND_GPU'}], 'max_batch_size': 16},
            {'instance_group': [{'count': 2, 'kind': 'KIND_GPU'}], 'max_batch_size': 1},
            {'instance_group': [{'count': 2, 'kind': 'KIND_GPU'}], 'max_batch_size': 4},
            {'instance_group': [{'count': 2, 'kind': 'KIND_GPU'}], 'max_batch_size': 16}
        ]
        # yapf: enable

        self._run_and_test_model_config_generator(yaml_str, expected_configs)

    def test_direct_cpu_only(self):
        """
        Test direct mode with cpu_only=true
        """

        # yapf: disable
        yaml_str = ("""
            run_config_search_max_instance_count: 2
            run_config_search_min_model_batch_size: 8
            run_config_search_max_model_batch_size: 16
            profile_models:
                - my-model:
                    cpu_only: True
            """)

        expected_configs = [
            {'dynamic_batching': {}, 'max_batch_size': 8, 'instance_group': [{'count': 1, 'kind': 'KIND_CPU'}]},
            {'dynamic_batching': {}, 'max_batch_size': 8, 'instance_group': [{'count': 2, 'kind': 'KIND_CPU'}]},
            {'dynamic_batching': {}, 'max_batch_size': 16, 'instance_group': [{'count': 1, 'kind': 'KIND_CPU'}]},
            {'dynamic_batching': {}, 'max_batch_size': 16, 'instance_group': [{'count': 2, 'kind': 'KIND_CPU'}]}
        ]
        # yapf: enable

        self._run_and_test_model_config_generator(yaml_str, expected_configs)

    def test_direct_max_batch_size_0(self):
        """
        Test direct mode with the the default config stating max_batch_size=0

        max_batch_size and dynamic_batching should not be part of the resulting configs
        """

        # yapf: disable
        protobuf = """
            platform: "fake_platform"
            max_batch_size: 0
            instance_group [
            {
                kind: KIND_CPU
                count: 1
            }
            ]
            """

        yaml_str = ("""
            run_config_search_max_instance_count: 4
            run_config_search_min_model_batch_size: 8
            run_config_search_max_model_batch_size: 8
            profile_models:
                - my-model
            """)

        expected_configs = [
            {'platform': "fake_platform", 'instance_group': [{'count': 1, 'kind': 'KIND_GPU'}]},
            {'platform': "fake_platform", 'instance_group': [{'count': 2, 'kind': 'KIND_GPU'}]},
            {'platform': "fake_platform", 'instance_group': [{'count': 3, 'kind': 'KIND_GPU'}]},
            {'platform': "fake_platform", 'instance_group': [{'count': 4, 'kind': 'KIND_GPU'}]}
        ]
        # yapf: enable

        self._run_and_test_model_config_generator(yaml_str, expected_configs, protobuf)

    def test_direct_max_batch_size_unspecified(self):
        """
        Test direct mode with the the default config not specifying max_batch_size

        max_batch_size and dynamic_batching should not be part of the resulting configs
        """

        # yapf: disable
        protobuf = """
            platform: "fake_platform"
            instance_group [
            {
                kind: KIND_CPU
                count: 1
            }
            ]
            """

        yaml_str = ("""
            run_config_search_max_instance_count: 4
            run_config_search_min_model_batch_size: 8
            run_config_search_max_model_batch_size: 8
            profile_models:
                - my-model
            """)

        expected_configs = [
            {'platform': "fake_platform", 'instance_group': [{'count': 1, 'kind': 'KIND_GPU'}]},
            {'platform': "fake_platform", 'instance_group': [{'count': 2, 'kind': 'KIND_GPU'}]},
            {'platform': "fake_platform", 'instance_group': [{'count': 3, 'kind': 'KIND_GPU'}]},
            {'platform': "fake_platform", 'instance_group': [{'count': 4, 'kind': 'KIND_GPU'}]}
        ]
        # yapf: enable

        self._run_and_test_model_config_generator(yaml_str, expected_configs, protobuf)

    def test_direct_sequence_batching(self):
        """
        Test direct mode with the the default config specifying sequence_batching

        dynamic_batching should not be part of the resulting configs
        """

        # yapf: disable
        protobuf = """
            platform: "fake_platform"
            max_batch_size: 4
            sequence_batching {}
            instance_group [
            {
                kind: KIND_CPU
                count: 1
            }
            ]
            """

        yaml_str = ("""
            run_config_search_max_instance_count: 4
            run_config_search_min_model_batch_size: 8
            run_config_search_max_model_batch_size: 16
            profile_models:
                - my-model
            """)

        expected_configs = [
            {'platform': "fake_platform", 'max_batch_size':  8, 'sequence_batching': {}, 'instance_group': [{'count': 1, 'kind': 'KIND_GPU'}]},
            {'platform': "fake_platform", 'max_batch_size':  8, 'sequence_batching': {}, 'instance_group': [{'count': 2, 'kind': 'KIND_GPU'}]},
            {'platform': "fake_platform", 'max_batch_size':  8, 'sequence_batching': {}, 'instance_group': [{'count': 3, 'kind': 'KIND_GPU'}]},
            {'platform': "fake_platform", 'max_batch_size':  8, 'sequence_batching': {}, 'instance_group': [{'count': 4, 'kind': 'KIND_GPU'}]},
            {'platform': "fake_platform", 'max_batch_size': 16, 'sequence_batching': {}, 'instance_group': [{'count': 1, 'kind': 'KIND_GPU'}]},
            {'platform': "fake_platform", 'max_batch_size': 16, 'sequence_batching': {}, 'instance_group': [{'count': 2, 'kind': 'KIND_GPU'}]},
            {'platform': "fake_platform", 'max_batch_size': 16, 'sequence_batching': {}, 'instance_group': [{'count': 3, 'kind': 'KIND_GPU'}]},
            {'platform': "fake_platform", 'max_batch_size': 16, 'sequence_batching': {}, 'instance_group': [{'count': 4, 'kind': 'KIND_GPU'}]}
        ]
        # yapf: enable

        self._run_and_test_model_config_generator(yaml_str, expected_configs, protobuf)

    def test_direct_nonempty_default_config(self):
        """
        Test direct mode with the the default config containing some values

        It will keep values that aren't part of the search, and will overwrite
        any values that are part of the search
        """

        # yapf: disable
        protobuf = """
            platform: "fake_platform"
            max_batch_size: 4
            instance_group [
            {
                kind: KIND_CPU
                count: 1
            }
            ]
            """

        yaml_str = ("""
            run_config_search_max_instance_count: 4
            run_config_search_min_model_batch_size: 8
            run_config_search_max_model_batch_size: 8
            profile_models:
                - my-model
            """)

        expected_configs = [
            {'platform': "fake_platform", 'max_batch_size': 8, 'instance_group': [{'count': 1, 'kind': 'KIND_GPU'}],'dynamic_batching': {}},
            {'platform': "fake_platform", 'max_batch_size': 8, 'instance_group': [{'count': 2, 'kind': 'KIND_GPU'}],'dynamic_batching': {}},
            {'platform': "fake_platform", 'max_batch_size': 8, 'instance_group': [{'count': 3, 'kind': 'KIND_GPU'}],'dynamic_batching': {}},
            {'platform': "fake_platform", 'max_batch_size': 8, 'instance_group': [{'count': 4, 'kind': 'KIND_GPU'}],'dynamic_batching': {}}
        ]
        # yapf: enable

        self._run_and_test_model_config_generator(yaml_str, expected_configs, protobuf)

    def test_remote_yes_params_specified(self):
        """
        Test remote mode with model_config_parameters specified
        """

        # yapf: disable
        yaml_str = ("""
            triton_launch_mode: remote
            run_config_search_max_instance_count: 16
            profile_models:
                test_model:
                    model_config_parameters:
                        max_batch_size: [1,4,16]
                        instance_group:
                        -
                            kind: KIND_GPU
                            count: [1,2]
            """)

        expected_configs = [
            {'instance_group': [{'count': 1, 'kind': 'KIND_GPU'}], 'max_batch_size': 1},
            {'instance_group': [{'count': 1, 'kind': 'KIND_GPU'}], 'max_batch_size': 4},
            {'instance_group': [{'count': 1, 'kind': 'KIND_GPU'}], 'max_batch_size': 16},
            {'instance_group': [{'count': 2, 'kind': 'KIND_GPU'}], 'max_batch_size': 1},
            {'instance_group': [{'count': 2, 'kind': 'KIND_GPU'}], 'max_batch_size': 4},
            {'instance_group': [{'count': 2, 'kind': 'KIND_GPU'}], 'max_batch_size': 16}
        ]
        # yapf: enable

        self._run_and_test_model_config_generator(yaml_str, expected_configs)

    def test_remote_no_params_specified(self):
        """
        Test remote mode with no model_config_parameters specified
        """

        # yapf: disable
        yaml_str = ("""
            triton_launch_mode: remote
            run_config_search_max_instance_count: 4
            run_config_search_min_model_batch_size: 8
            run_config_search_max_model_batch_size: 8
            profile_models:
                - my-model
            """)

        expected_configs = [
            {'max_batch_size': 8, 'instance_group': [{'count': 1, 'kind': 'KIND_GPU'}],'dynamic_batching': {}},
            {'max_batch_size': 8, 'instance_group': [{'count': 2, 'kind': 'KIND_GPU'}],'dynamic_batching': {}},
            {'max_batch_size': 8, 'instance_group': [{'count': 3, 'kind': 'KIND_GPU'}],'dynamic_batching': {}},
            {'max_batch_size': 8, 'instance_group': [{'count': 4, 'kind': 'KIND_GPU'}],'dynamic_batching': {}}
        ]
        # yapf: enable

        self._run_and_test_model_config_generator(yaml_str, expected_configs)

    def test_search_subparameter(self):
        """
        Test that if a subparameter is swept, that it will not overwrite other subparameters

        Param2 should exist in all results despite the fact that param1 is the one being swept.
        Also, gpu_execution_accelerator (a nested subproperty) should still exist despite a
        sibling property (cpu_execution_accelerator) being overwritten
        """

        # yapf: disable
        protobuf = """
            max_batch_size: 4
            instance_group [
            {
                kind: KIND_GPU
                count: 1
            }
            ]
            parameters [
                {
                    key: "param1"
                    value: { string_value: "param1_value" }
                },
                {
                    key: "param2"
                    value: { string_value: "param2_value" }
                }
            ]
            optimization { execution_accelerators {
                cpu_execution_accelerator : [ {
                    name : "fake_cpu_accelerator"
                }]
                gpu_execution_accelerator : [ {
                    name : "fake_gpu_accelerator"
                }]
            }}
            """

        yaml_str = ("""
            profile_models:
                my-model:
                    model_config_parameters:
                        optimization:
                            execution_accelerators:
                                cpu_execution_accelerator:
                                - name: "new_cpu_accelerator"
                        parameters:
                            param1:
                                string_value: ["foo", "bar"]
            """)

        expected_configs = [
            {
                'optimization':
                {
                    'execution_accelerators':
                    {
                        'cpu_execution_accelerator': [{'name': 'new_cpu_accelerator'}],
                        'gpu_execution_accelerator': [{'name': 'fake_gpu_accelerator'}]
                    }
                },
                'parameters':
                {
                    'param1': {'string_value': 'foo'},
                    'param2': {'string_value': 'param2_value'}
                },
                'max_batch_size': 4,
                'instance_group': [{'count': 1, 'kind': 'KIND_GPU'}]
            },
            {
                'optimization':
                {
                    'execution_accelerators':
                    {
                        'cpu_execution_accelerator': [{'name': 'new_cpu_accelerator'}],
                        'gpu_execution_accelerator': [{'name': 'fake_gpu_accelerator'}]
                    }
                },
                'parameters':
                {
                    'param1': {'string_value': 'bar'},
                    'param2': {'string_value': 'param2_value'}
                },
                'max_batch_size': 4,
                'instance_group': [{'count': 1, 'kind': 'KIND_GPU'}]
            },
        ]
        # yapf: enable

        self._run_and_test_model_config_generator(yaml_str, expected_configs, protobuf)

    def test_search_dynamic_batching_subparameter(self):
        """
        Test that if dynamic batching was already on with subparameters, they will not be overwritten in automatic search

        Normally automatic search just turns on dynamic_batching, but in this case dynamic_batching is already
        on in the user's default model config with a value for max_queue_delay_microseconds. That value should remain
        unchanged in all model configs generated by the manual search
        """

        # yapf: disable
        protobuf = """
            max_batch_size: 4
            instance_group [
            {
                kind: KIND_GPU
                count: 1
            }
            ]
            dynamic_batching {
                max_queue_delay_microseconds: 100
            }
            """

        yaml_str = ("""
            run_config_search_max_instance_count: 4
            run_config_search_min_model_batch_size: 8
            run_config_search_max_model_batch_size: 8
            profile_models:
                - my-model
            """)

        expected_configs = [
            {'max_batch_size': 8, 'instance_group': [{'count': 1, 'kind': 'KIND_GPU'}],'dynamic_batching': { 'max_queue_delay_microseconds': '100'}},
            {'max_batch_size': 8, 'instance_group': [{'count': 2, 'kind': 'KIND_GPU'}],'dynamic_batching': { 'max_queue_delay_microseconds': '100'}},
            {'max_batch_size': 8, 'instance_group': [{'count': 3, 'kind': 'KIND_GPU'}],'dynamic_batching': { 'max_queue_delay_microseconds': '100'}},
            {'max_batch_size': 8, 'instance_group': [{'count': 4, 'kind': 'KIND_GPU'}],'dynamic_batching': { 'max_queue_delay_microseconds': '100'}}
        ]
        # yapf: enable

        self._run_and_test_model_config_generator(yaml_str, expected_configs, protobuf)

    def test_apply_value_to_dict(self):
        """
        Test different combinations of input and existing value types for apply_value_to_dict()
        """
        # Both input and existing are scalar value in a dict
        existing_dict = {"a": 1, "b": 2}
        expected_dict = {"a": 3, "b": 2}
        BaseModelConfigGenerator._apply_value_to_dict("a", 3, existing_dict)
        self.assertEqual(existing_dict, expected_dict)

        # Input is scalar, existing is dict
        existing_dict = {"a": 1, "b": {"c": 5, "d": 6}}
        expected_dict = {"a": 1, "b": 2}
        BaseModelConfigGenerator._apply_value_to_dict("b", 2, existing_dict)
        self.assertEqual(existing_dict, expected_dict)

        # Input is dict, existing is scalar
        existing_dict = {"a": 1, "b": 3}
        expected_dict = {"a": 1, "b": {"c": 7, "d": 8}}
        BaseModelConfigGenerator._apply_value_to_dict(
            "b", {"c": 7, "d": 8}, existing_dict
        )
        self.assertEqual(existing_dict, expected_dict)

        # Input and dict are both dicts
        existing_dict = {"a": 1, "b": {"c": {"e": 9, "f": 10}, "d": 6}}
        expected_dict = {"a": 1, "b": {"c": {"e": 11, "f": 10}, "d": 6}}
        BaseModelConfigGenerator._apply_value_to_dict(
            "b", {"c": {"e": 11}}, existing_dict
        )
        self.assertEqual(existing_dict, expected_dict)

    def test_early_exit_off_automatic_asserts(self):
        """
        Test that passing early_exit=False for automatic search raises an assert
        """

        # yapf: disable
        yaml_str = ("""
            profile_models:
                - my-model
            """)

        expected_configs = []
        # yapf: enable

        with self.assertRaises(TritonModelAnalyzerException):
            self._run_and_test_model_config_generator(
                yaml_str, expected_configs, early_exit_enable=False
            )

    def test_early_exit_on_automatic(self):
        """
        Test that automatic mode will early exit max_batch_size when throughput plateaus
        """
        # yapf: disable
        yaml_str = ("""
            profile_models:
                - my-model
            run_config_search_max_instance_count: 3
            run_config_search_max_model_batch_size: 8
            """)

        expected_configs = [
            {'max_batch_size': 1, 'instance_group': [{'count': 1, 'kind': 'KIND_GPU'}],'dynamic_batching': {}},
            {'max_batch_size': 2, 'instance_group': [{'count': 1, 'kind': 'KIND_GPU'}],'dynamic_batching': {}},
            {'max_batch_size': 4, 'instance_group': [{'count': 1, 'kind': 'KIND_GPU'}],'dynamic_batching': {}},
            {'max_batch_size': 8, 'instance_group': [{'count': 1, 'kind': 'KIND_GPU'}],'dynamic_batching': {}},
            {'max_batch_size': 1, 'instance_group': [{'count': 2, 'kind': 'KIND_GPU'}],'dynamic_batching': {}},
            {'max_batch_size': 2, 'instance_group': [{'count': 2, 'kind': 'KIND_GPU'}],'dynamic_batching': {}},
            {'max_batch_size': 1, 'instance_group': [{'count': 3, 'kind': 'KIND_GPU'}],'dynamic_batching': {}},
            {'max_batch_size': 2, 'instance_group': [{'count': 3, 'kind': 'KIND_GPU'}],'dynamic_batching': {}},
            {'max_batch_size': 4, 'instance_group': [{'count': 3, 'kind': 'KIND_GPU'}],'dynamic_batching': {}},
            {'max_batch_size': 8, 'instance_group': [{'count': 3, 'kind': 'KIND_GPU'}],'dynamic_batching': {}},
        ]

        with patch.object(TestModelConfigGenerator,
                          "_get_next_fake_throughput") as mock_method:
            mock_method.side_effect = [
                1, 2, 4, 8,  # 1 instance
                1, 1,        # 2 instances. Yes backoff
                1, 2, 4, 8   # 3 instances
            ]
            self._run_and_test_model_config_generator(yaml_str,
                                                      expected_configs,
                                                      early_exit_enable=True)
        # yapf: enable

    def test_early_exit_off_manual(self):
        """
        Test that manual mode will not early exit despite throughput plateauing despite because early_exit_enable=False
        """

        # yapf: disable
        yaml_str = ("""
            profile_models:
                test_model:
                    model_config_parameters:
                        max_batch_size: [1,2,4,8]
                        instance_group:
                        -
                            kind: KIND_GPU
                            count: [1,2]
            """)

        expected_configs = [
            {'max_batch_size': 1, 'instance_group': [{'count': 1, 'kind': 'KIND_GPU'}]},
            {'max_batch_size': 2, 'instance_group': [{'count': 1, 'kind': 'KIND_GPU'}]},
            {'max_batch_size': 4, 'instance_group': [{'count': 1, 'kind': 'KIND_GPU'}]},
            {'max_batch_size': 8, 'instance_group': [{'count': 1, 'kind': 'KIND_GPU'}]},
            {'max_batch_size': 1, 'instance_group': [{'count': 2, 'kind': 'KIND_GPU'}]},
            {'max_batch_size': 2, 'instance_group': [{'count': 2, 'kind': 'KIND_GPU'}]},
            {'max_batch_size': 4, 'instance_group': [{'count': 2, 'kind': 'KIND_GPU'}]},
            {'max_batch_size': 8, 'instance_group': [{'count': 2, 'kind': 'KIND_GPU'}]},
        ]
        # yapf: enable

        with patch.object(
            TestModelConfigGenerator, "_get_next_fake_throughput"
        ) as mock_method:
            mock_method.return_value = 1
            self._run_and_test_model_config_generator(
                yaml_str, expected_configs, early_exit_enable=False
            )

    def test_early_exit_on_manual(self):
        """
        Test that manual mode will early exit when throughput plateaus when early_exit_enable=True
        """

        # yapf: disable
        yaml_str = ("""
            profile_models:
                test_model:
                    model_config_parameters:
                        max_batch_size: [1,2,3,4]
                        instance_group:
                        -
                            kind: KIND_GPU
                            count: [1,2,3]
            """)

        expected_configs = [
            {'max_batch_size': 1, 'instance_group': [{'count': 1, 'kind': 'KIND_GPU'}]},
            {'max_batch_size': 2, 'instance_group': [{'count': 1, 'kind': 'KIND_GPU'}]},
            {'max_batch_size': 3, 'instance_group': [{'count': 1, 'kind': 'KIND_GPU'}]},
            {'max_batch_size': 4, 'instance_group': [{'count': 1, 'kind': 'KIND_GPU'}]},
            {'max_batch_size': 1, 'instance_group': [{'count': 2, 'kind': 'KIND_GPU'}]},
            {'max_batch_size': 2, 'instance_group': [{'count': 2, 'kind': 'KIND_GPU'}]},
            {'max_batch_size': 1, 'instance_group': [{'count': 3, 'kind': 'KIND_GPU'}]},
            {'max_batch_size': 2, 'instance_group': [{'count': 3, 'kind': 'KIND_GPU'}]},
            {'max_batch_size': 3, 'instance_group': [{'count': 3, 'kind': 'KIND_GPU'}]},
            {'max_batch_size': 4, 'instance_group': [{'count': 3, 'kind': 'KIND_GPU'}]},
        ]

        with patch.object(TestModelConfigGenerator,
                          "_get_next_fake_throughput") as mock_method:
            mock_method.side_effect = [
                1, 2, 4, 8,  # 1 instance
                1, 1,        # 2 instances. Yes backoff
                1, 2, 4, 8   # 3 instances
            ]
            self._run_and_test_model_config_generator(yaml_str,
                                                      expected_configs,
                                                      early_exit_enable=True)
        # yapf: enable

    def test_extract_model_name_from_variant_name(self):
        input_output_pairs = {}
        input_output_pairs[
            "onnx_int32_int32_int32_config_default"
        ] = "onnx_int32_int32_int32"
        input_output_pairs["onnx_int32_int32_int32_config_2"] = "onnx_int32_int32_int32"
        input_output_pairs["onnx_int32_int32_int32"] = "onnx_int32_int32_int32"

        for variant_name, expected_model_name in input_output_pairs.items():
            model_name = BaseModelConfigGenerator.extract_model_name_from_variant_name(
                variant_name
            )
            self.assertEqual(model_name, expected_model_name)

    def _run_and_test_model_config_generator(
        self,
        yaml_str,
        expected_configs,
        protobuf="max_batch_size: 8",
        default_only=False,
        early_exit_enable=True,
    ):
        """
        Main function that creates a config from the yaml_str, runs it through
        ModelConfigGenerator, and compares the resulting model_configs vs the expected_configs
        """
        args = [
            "model-analyzer",
            "profile",
            "--model-repository",
            "cli_repository",
            "-f",
            "path-to-config-file",
        ]

        # Use mock model config or else TritonModelAnalyzerException will be thrown as it tries to read from disk
        self.mock_model_config = MockModelConfig(protobuf)
        self.mock_model_config.start()
        config = evaluate_mock_config(args, yaml_str, subcommand="profile")

        profile_models = []
        for model in config.profile_models:
            profile_models.append(
                ModelProfileSpec(model, config, MagicMock(), MagicMock())
            )

        # Fake out a client that can return a 'model_config' dict with
        # a valid name (only used by remote mode)
        #
        fake_client = MagicMock()
        fake_client.get_model_config = lambda name, retry_count: {"name": name}

        try:
            mcg = ModelConfigGeneratorFactory.create_model_config_generator(
                config,
                MagicMock(),
                profile_models[0],
                fake_client,
                ModelVariantNameManager(),
                default_only=default_only,
                early_exit_enable=early_exit_enable,
            )

            model_configs = []
            for model_config_variant in mcg.get_configs():
                mcg.set_last_results(self._get_next_fake_results())
                model_config_dict = model_config_variant.model_config.get_config()
                model_configs.append(model_config_dict)

            self.assertEqual(len(expected_configs), len(model_configs))

            # Rip out the model name (so the order doesn't have to exactly match),
            # but verify that it exists and is not none
            #
            for config in model_configs:
                name = config.pop("name", None)
                self.assertIsNotNone(name)

            # Confirm the configs match
            #
            for config in expected_configs:
                self.assertIn(config, model_configs)

        finally:
            self.mock_model_config.stop()

    def _get_next_fake_results(self):
        throughput = self._get_next_fake_throughput()

        measurement = construct_run_config_measurement(
            model_name=MagicMock(),
            model_config_names=["test_model_config_name"],
            model_specific_pa_params=MagicMock(),
            gpu_metric_values=MagicMock(),
            non_gpu_metric_values=[{"perf_throughput": throughput}],
        )

        return [measurement]

    def _get_next_fake_throughput(self):
        self._fake_throughput += 1
        return self._fake_throughput

    def setUp(self):
        # Mock path validation
        self.mock_os = MockOSMethods(
            mock_paths=["model_analyzer.config.input.config_utils"]
        )
        self.mock_os.start()

    def tearDown(self):
        self.mock_os.stop()
        patch.stopall()
        ModelConfig._default_config_dict = {}


if __name__ == "__main__":
    unittest.main()
