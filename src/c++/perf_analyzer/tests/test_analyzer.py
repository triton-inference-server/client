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

import os
import unittest
from unittest.mock import MagicMock, patch

from model_analyzer.analyzer import Analyzer
from model_analyzer.config.input.config_status import ConfigStatus
from model_analyzer.config.run.model_run_config import ModelRunConfig
from model_analyzer.config.run.run_config import RunConfig
from model_analyzer.constants import CONFIG_PARSER_SUCCESS
from model_analyzer.result.results import Results
from model_analyzer.result.run_config_result import RunConfigResult
from model_analyzer.result.sorted_results import SortedResults
from model_analyzer.state.analyzer_state_manager import AnalyzerStateManager
from model_analyzer.triton.model.model_config import ModelConfig
from model_analyzer.triton.model.model_config_variant import ModelConfigVariant
from tests.common.test_utils import evaluate_mock_config

from .common import test_result_collector as trc


class TestAnalyzer(trc.TestResultCollector):
    """
    Tests the methods of the Analyzer class
    """

    def tearDown(self):
        patch.stopall()

    def mock_get_state_variable(self, name):
        if name == "ResultManager.results":
            return Results()
        else:
            return {
                "model1": {
                    "config1": None,
                    "config2": None,
                    "config3": None,
                    "config4": None,
                }
            }

    def mock_get_list_of_models(self):
        return ["model1"]

    @patch.multiple(
        f"{AnalyzerStateManager.__module__}.AnalyzerStateManager",
        get_state_variable=mock_get_state_variable,
        exiting=lambda _: False,
    )
    @patch.multiple(
        f"{Analyzer.__module__}.Analyzer",
        _create_metrics_manager=MagicMock(),
        _create_model_manager=MagicMock(),
        _get_server_only_metrics=MagicMock(),
        _profile_models=MagicMock(),
        _check_for_perf_analyzer_errors=MagicMock(),
    )
    def test_profile_skip_summary_reports(self, **mocks):
        """
        Tests when the skip_summary_reports config option is turned on,
        the profile stage does not create any summary reports.

        NOTE: this test only ensures that the reports are not created with
        the default export-path.
        """
        args = [
            "model-analyzer",
            "profile",
            "--model-repository",
            "/tmp",
            "--profile-models",
            "model1",
            "--config-file",
            "/tmp/my_config.yml",
            "--checkpoint-directory",
            "/tmp/my_checkpoints",
            "--skip-summary-reports",
        ]
        config = evaluate_mock_config(args, "", subcommand="profile")
        state_manager = AnalyzerStateManager(config, None)
        analyzer = Analyzer(config, None, state_manager, checkpoint_required=False)
        analyzer.profile(client=None, gpus=None, mode=None, verbose=False)

        path = os.getcwd()
        self.assertFalse(os.path.exists(os.path.join(path, "plots")))
        self.assertFalse(os.path.exists(os.path.join(path, "results")))
        self.assertFalse(os.path.exists(os.path.join(path, "reports")))

    def mock_top_n_results(self, model_name=None, n=SortedResults.GET_ALL_RESULTS):
        rc1 = RunConfig({})
        rc1.add_model_run_config(
            ModelRunConfig(
                "fake_model_name",
                ModelConfigVariant(
                    ModelConfig.create_from_dictionary({"name": "config1"}),
                    "config1",
                ),
                MagicMock(),
            )
        )
        rc2 = RunConfig({})
        rc2.add_model_run_config(
            ModelRunConfig(
                "fake_model_name",
                ModelConfigVariant(
                    ModelConfig.create_from_dictionary({"name": "config3"}),
                    "config3",
                ),
                MagicMock(),
            )
        )
        rc3 = RunConfig({})
        rc3.add_model_run_config(
            ModelRunConfig(
                "fake_model_name",
                ModelConfigVariant(
                    ModelConfig.create_from_dictionary({"name": "config4"}),
                    "config4",
                ),
                MagicMock(),
            )
        )

        return [
            RunConfigResult("fake_model_name", rc1, MagicMock(), MagicMock()),
            RunConfigResult("fake_model_name", rc2, MagicMock(), MagicMock()),
            RunConfigResult("fake_model_name", rc3, MagicMock(), MagicMock()),
        ]

    @patch(
        "model_analyzer.config.input.config_command_profile.file_path_validator",
        lambda _: ConfigStatus(status=CONFIG_PARSER_SUCCESS),
    )
    @patch(
        "model_analyzer.config.input.config_command_profile.ConfigCommandProfile._preprocess_and_verify_arguments",
        lambda _: None,
    )
    @patch(
        "model_analyzer.result.result_manager.ResultManager.top_n_results",
        mock_top_n_results,
    )
    def test_get_report_command_help_string(self):
        """
        Tests that the member function returning the report command help string
        works correctly.
        """

        args = [
            "model-analyzer",
            "profile",
            "--profile-models",
            "model1",
            "--config-file",
            "/tmp/my_config.yml",
            "--checkpoint-directory",
            "/tmp/my_checkpoints",
            "--export-path",
            "/tmp/my_export_path",
            "--model-repository",
            "cli_repository",
        ]
        config = evaluate_mock_config(args, "", subcommand="profile")
        state_manager = AnalyzerStateManager(config, None)
        analyzer = Analyzer(config, None, state_manager, checkpoint_required=False)
        self.assertEqual(
            analyzer._get_report_command_help_string(model_name="model1"),
            "To generate detailed reports for the 3 best model1 configurations, run "
            "`model-analyzer report --report-model-configs "
            "config1,config3,config4 --export-path /tmp/my_export_path "
            "--config-file /tmp/my_config.yml --checkpoint-directory "
            "/tmp/my_checkpoints`",
        )

    def test_multiple_models_in_report_model_config(self):
        """
        Test that multiple different models in the report are detected
        """
        # Single model config
        args = [
            "model-analyzer",
            "report",
            "--report-model-configs",
            "modelA_config_0",
            "--checkpoint-directory",
            "/tmp/my_checkpoints",
        ]
        config = evaluate_mock_config(args, "", subcommand="report")
        state_manager = AnalyzerStateManager(config, None)
        analyzer = Analyzer(config, None, state_manager, checkpoint_required=False)

        self.assertFalse(analyzer._multiple_models_in_report_model_config())

        # Multiple different models
        args = [
            "model-analyzer",
            "report",
            "--report-model-configs",
            "modelA_config_0,modelB_config_1",
            "--checkpoint-directory",
            "/tmp/my_checkpoints",
        ]
        config = evaluate_mock_config(args, "", subcommand="report")
        state_manager = AnalyzerStateManager(config, None)
        analyzer = Analyzer(config, None, state_manager, checkpoint_required=False)

        self.assertTrue(analyzer._multiple_models_in_report_model_config())

        # Single model - multiple configs
        args = [
            "model-analyzer",
            "report",
            "--report-model-configs",
            "modelA_config_0,modelA_config_1",
            "--checkpoint-directory",
            "/tmp/my_checkpoints",
        ]
        config = evaluate_mock_config(args, "", subcommand="report")
        state_manager = AnalyzerStateManager(config, None)
        analyzer = Analyzer(config, None, state_manager, checkpoint_required=False)

        self.assertFalse(analyzer._multiple_models_in_report_model_config())


if __name__ == "__main__":
    unittest.main()
