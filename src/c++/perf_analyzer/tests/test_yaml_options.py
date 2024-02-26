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

import yaml
from model_analyzer.config.input.yaml_config_validator import YamlConfigValidator
from model_analyzer.model_analyzer_exceptions import TritonModelAnalyzerException

from .common import test_result_collector as trc


class TestYamlOptions(trc.TestResultCollector):
    def test_correct_option(self):
        correct_option = "client_max_retries"
        YamlConfigValidator._create_valid_option_set()
        self.assertTrue(YamlConfigValidator._is_valid_option(correct_option))

    def test_misspelled_option(self):
        misspelled_option = "profile_model"
        YamlConfigValidator._create_valid_option_set()
        self.assertFalse(YamlConfigValidator._is_valid_option(misspelled_option))

    def test_using_hyphens_not_underscores(self):
        hyphen_option = "triton-server-flags"
        YamlConfigValidator._create_valid_option_set()
        self.assertFalse(YamlConfigValidator._is_valid_option(hyphen_option))

    def test_multiple_options(self):
        """
        Tests multiple options.
        The following are incorrect:
            "profile_model",
            "triton-server-flags",
            "DURATION_seconds",
            ""
        """
        options = {
            "client_max_retries",
            "profile_model",
            "triton-server-flags",
            "DURATION_seconds",
            "",
        }
        count = 0
        YamlConfigValidator._create_valid_option_set()
        for entry in options:
            if not YamlConfigValidator._is_valid_option(entry):
                count += 1
        self.assertEqual(
            count,
            4,
            f"{count} incorrect yaml options are present in the yaml configuration file",
        )

    def test_valid_yaml_file(self):
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
        # yapf: enable

        yaml_config = self._load_config_file(yaml_str)
        YamlConfigValidator.validate(yaml_config)

    def test_invalid_yaml_file(self):
        """
        Raises an exception because run-config-search-max-instance-count: 16 uses
        hyphens instead of the required underscores
        """
        # yapf: disable
        yaml_str = ("""
            run-config-search-max-instance-count: 16
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
        # yapf: enable

        yaml_config = self._load_config_file(yaml_str)
        with self.assertRaises(TritonModelAnalyzerException):
            YamlConfigValidator.validate(yaml_config)

    def _load_config_file(self, yaml_str):
        """
        Load YAML config
        """

        config = yaml.safe_load(yaml_str)
        return config
