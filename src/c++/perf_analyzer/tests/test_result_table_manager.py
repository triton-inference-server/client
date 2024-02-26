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
from filecmp import cmp
from shutil import rmtree
from unittest.mock import patch

from google.protobuf import text_format
from tritonclient.grpc import model_config_pb2

from model_analyzer.result.constraint_manager import ConstraintManager
from model_analyzer.result.result_manager import ResultManager
from model_analyzer.result.result_table_manager import ResultTableManager
from model_analyzer.state.analyzer_state_manager import AnalyzerStateManager

from .common import test_result_collector as trc
from .common.test_utils import (
    ROOT_DIR,
    evaluate_mock_config,
    load_bls_result_manager,
    load_ensemble_result_manager,
    load_multi_model_result_manager,
    load_request_rate_result_manager,
    load_single_model_result_manager,
)


class TestResultTableManager(trc.TestResultCollector):
    def test_single_model_csv_against_golden(self):
        """
        Match the csvs against the golden versions in
        tests/common/single-model-ckpt
        """
        table_manager = self._create_single_model_result_table_manager()

        table_manager.create_tables()
        table_manager.tabulate_results()
        table_manager.export_results()

        self.assertTrue(
            cmp(
                f"{ROOT_DIR}/single-model-ckpt/results/metrics-model-gpu.csv",
                f"{ROOT_DIR}/single-model-ckpt/golden-metrics-model-gpu.csv",
            )
        )

        self.assertTrue(
            cmp(
                f"{ROOT_DIR}/single-model-ckpt/results/metrics-model-inference.csv",
                f"{ROOT_DIR}/single-model-ckpt/golden-metrics-model-inference.csv",
            )
        )

        self.assertTrue(
            cmp(
                f"{ROOT_DIR}/single-model-ckpt/results/metrics-server-only.csv",
                f"{ROOT_DIR}/single-model-ckpt/golden-metrics-server-only.csv",
            )
        )

        rmtree(f"{ROOT_DIR}/single-model-ckpt/results/")

    def test_multi_model_csv_against_golden(self):
        """
        Match the csvs against the golden versions in
        tests/common/multi-model-ckpt
        """
        table_manager = self._create_multi_model_result_table_manager()

        table_manager.create_tables()
        table_manager.tabulate_results()
        table_manager.export_results()

        self.assertTrue(
            cmp(
                f"{ROOT_DIR}/multi-model-ckpt/results/metrics-model-gpu.csv",
                f"{ROOT_DIR}/multi-model-ckpt/golden-metrics-model-gpu.csv",
            )
        )

        self.assertTrue(
            cmp(
                f"{ROOT_DIR}/multi-model-ckpt/results/metrics-model-inference.csv",
                f"{ROOT_DIR}/multi-model-ckpt/golden-metrics-model-inference.csv",
            )
        )

        self.assertTrue(
            cmp(
                f"{ROOT_DIR}/multi-model-ckpt/results/metrics-server-only.csv",
                f"{ROOT_DIR}/multi-model-ckpt/golden-metrics-server-only.csv",
            )
        )

        rmtree(f"{ROOT_DIR}/multi-model-ckpt/results/")

    def test_ensemble_csv_against_golden(self):
        """
        Match the csvs against the golden versions in
        tests/common/ensemble-ckpt
        """
        table_manager = self._create_ensemble_result_table_manager()

        table_manager.create_tables()
        table_manager.tabulate_results()
        table_manager.export_results()

        self.assertTrue(
            cmp(
                f"{ROOT_DIR}/ensemble-ckpt/results/metrics-model-gpu.csv",
                f"{ROOT_DIR}/ensemble-ckpt/golden-metrics-model-gpu.csv",
            )
        )

        self.assertTrue(
            cmp(
                f"{ROOT_DIR}/ensemble-ckpt/results/metrics-model-inference.csv",
                f"{ROOT_DIR}/ensemble-ckpt/golden-metrics-model-inference.csv",
            )
        )

        self.assertTrue(
            cmp(
                f"{ROOT_DIR}/ensemble-ckpt/results/metrics-server-only.csv",
                f"{ROOT_DIR}/ensemble-ckpt/golden-metrics-server-only.csv",
            )
        )

        rmtree(f"{ROOT_DIR}/ensemble-ckpt/results/")

    def test_request_rate_csv_against_golden(self):
        """
        Match the csvs against the golden versions in
        tests/common/request-rate-ckpt
        """
        table_manager = self._create_request_rate_result_table_manager()

        table_manager.create_tables()
        table_manager.tabulate_results()
        table_manager.export_results()

        self.assertTrue(
            cmp(
                f"{ROOT_DIR}/request-rate-ckpt/results/metrics-model-gpu.csv",
                f"{ROOT_DIR}/request-rate-ckpt/golden-metrics-model-gpu.csv",
            )
        )

        self.assertTrue(
            cmp(
                f"{ROOT_DIR}/request-rate-ckpt/results/metrics-model-inference.csv",
                f"{ROOT_DIR}/request-rate-ckpt/golden-metrics-model-inference.csv",
            )
        )

        self.assertTrue(
            cmp(
                f"{ROOT_DIR}/request-rate-ckpt/results/metrics-server-only.csv",
                f"{ROOT_DIR}/request-rate-ckpt/golden-metrics-server-only.csv",
            )
        )

        rmtree(f"{ROOT_DIR}/request-rate-ckpt/results/")

    def test_create_inference_table_with_backend_parameters(self):
        args = [
            "model-analyzer",
            "profile",
            "-f",
            "config.yml",
            "--model-repository",
            ".",
        ]
        yaml_str = """
            profile_models: profile_models
            inference_output_fields: model_name,batch_size,backend_parameter/parameter_1,backend_parameter/parameter_2
        """
        config = evaluate_mock_config(args, yaml_str, subcommand="profile")
        state_manager = AnalyzerStateManager(config=config, server=None)
        result_manager = ResultManager(
            config=config,
            state_manager=state_manager,
            constraint_manager=ConstraintManager(config),
        )
        result_table_manager = ResultTableManager(
            config=config, result_manager=result_manager
        )

        result_table_manager._create_inference_table()
        self.assertEqual(
            result_table_manager._inference_output_fields,
            [
                "model_name",
                "batch_size",
                "backend_parameter/parameter_1",
                "backend_parameter/parameter_2",
            ],
        )

    def test_get_common_row_items_with_backend_parameters(self):
        """
        This tests that a metrics model inference table row can be created with
        backend parameters included. Each backend parameter gets its own column.
        The column name is the backend parameter key (prepended with a prefix
        to avoid potentially overlapping with an existing column). The column
        value is the backend parameter value.

        Here is an example table:

        Models (Inference):
        Model     Model Config Path   backend_parameter/add_sub_key_1   backend_parameter/add_sub_key_2
        add_sub   add_sub_config_2    add_sub_value_1                   add_sub_value_2
        add_sub   add_sub_config_0    add_sub_value_1                   add_sub_value_2
        add_sub   add_sub_config_1    add_sub_value_1                   add_sub_value_2

        Each row of the metrics model inference table corresponds to one model
        config variant.

        It is possible for a user to run the profile command with multiple
        models config variants from different models with potentially different
        backend parameters. This test includes backend parameters from two
        separate models, showing that for one particular row (for a 'model A'
        config variant), it only populates the backend parameter cells for
        'model A', and the backend parameter cells for 'model B' are empty
        (None).

        Here is an example table with backend parameters from different models:

        Models (Inference):
        Model       Model Config Path   backend_parameter/add_sub_key_1   backend_parameter/add_sub_key_2   backend_parameter/add_sub_2_key_1   backend_parameter/add_sub_2_key_2
        add_sub     add_sub_config_2    add_sub_value_1                   add_sub_value_2                   None                                None
        add_sub     add_sub_config_0    add_sub_value_1                   add_sub_value_2                   None                                None
        add_sub     add_sub_config_1    add_sub_value_1                   add_sub_value_2                   None                                None
        add_sub_2   add_sub_2_config_2  None                              None                              add_sub_2_value_1                   add_sub_2_value_2
        add_sub_2   add_sub_2_config_1  None                              None                              add_sub_2_value_1                   add_sub_2_value_2
        add_sub_2   add_sub_2_config_0  None                              None                              add_sub_2_value_1                   add_sub_2_value_2
        """

        args = [
            "model-analyzer",
            "profile",
            "-f",
            "config.yml",
            "--model-repository",
            ".",
        ]
        yaml_str = """
            profile_models: profile_models
            inference_output_fields: model_name,batch_size,backend_parameter/model_1_key_1,backend_parameter/model_1_key_2,backend_parameter/model_2_key_1
        """
        config = evaluate_mock_config(args, yaml_str, subcommand="profile")
        state_manager = AnalyzerStateManager(config=config, server=None)
        result_manager = ResultManager(
            config=config,
            state_manager=state_manager,
            constraint_manager=ConstraintManager(config),
        )
        table_manager = ResultTableManager(config=config, result_manager=result_manager)
        model_config_str = """
            parameters: {
            key: "model_1_key_1"
                value: {
                string_value:"model_1_value_1"
                }
            }
            parameters: {
            key:"model_1_key_2"
                value: {
                string_value:"model_1_value_2"
                }
            }
            """
        backend_parameters = text_format.Parse(
            model_config_str, model_config_pb2.ModelConfig()
        ).parameters
        row = table_manager._get_common_row_items(
            fields=[
                "model_name",
                "batch_size",
                "backend_parameter/model_1_key_1",
                "backend_parameter/model_1_key_2",
                "backend_parameter/model_2_key_1",
            ],
            batch_sizes="batch_size",
            concurrencies=None,
            request_rates=None,
            satisfies=None,
            model_name="model_name",
            model_config_path=None,
            dynamic_batchings=None,
            instance_groups=None,
            max_batch_sizes=None,
            backend_parameters=backend_parameters,
        )
        self.assertEqual(
            row,
            ["model_name", "batch_size", "model_1_value_1", "model_1_value_2", None],
        )

    def _create_single_model_result_table_manager(self):
        result_manager, config = load_single_model_result_manager()

        return ResultTableManager(config, result_manager)

    def _create_multi_model_result_table_manager(self):
        result_manager, config = load_multi_model_result_manager()

        return ResultTableManager(config, result_manager)

    def _create_ensemble_result_table_manager(self):
        result_manager, config = load_ensemble_result_manager()

        return ResultTableManager(config, result_manager)

    def _create_bls_result_table_manager(self):
        result_manager, config = load_bls_result_manager()

        return ResultTableManager(config, result_manager)

    def _create_request_rate_result_table_manager(self):
        result_manager, config = load_request_rate_result_manager()

        return ResultTableManager(config, result_manager)

    def tearDown(self):
        patch.stopall()


if __name__ == "__main__":
    unittest.main()
