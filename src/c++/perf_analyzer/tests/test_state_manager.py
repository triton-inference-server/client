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

from model_analyzer.model_analyzer_exceptions import TritonModelAnalyzerException
from model_analyzer.state.analyzer_state_manager import AnalyzerStateManager

from .common import test_result_collector as trc
from .common.test_utils import evaluate_mock_config
from .mocks.mock_glob import MockGlobMethods
from .mocks.mock_io import MockIOMethods
from .mocks.mock_json import MockJSONMethods
from .mocks.mock_os import MockOSMethods


class TestAnalyzerStateManagerMethods(trc.TestResultCollector):
    @patch(
        "model_analyzer.config.input.config_command_profile.ConfigCommandProfile._preprocess_and_verify_arguments",
        MagicMock(),
    )
    def setUp(self):
        args = [
            "model-analyzer",
            "profile",
            "--model-repository",
            "cli_repository",
            "-f",
            "path-to-config-file",
            "--profile-models",
            "test_model",
        ]
        yaml_str = """
            export_path: /test_export_path/
        """

        # start mocks
        self.mock_io = MockIOMethods(
            mock_paths=["model_analyzer.state.analyzer_state_manager"]
        )
        self.mock_json = MockJSONMethods()
        self.mock_os = MockOSMethods(
            mock_paths=[
                "model_analyzer.state.analyzer_state_manager",
                "model_analyzer.config.input.config_utils",
            ]
        )
        self.mock_glob = MockGlobMethods()

        self.mock_io.start()
        self.mock_json.start()
        self.mock_os.start()
        self.mock_glob.start()

        config = evaluate_mock_config(args, yaml_str, subcommand="profile")

        # state manager
        self.state_manager = AnalyzerStateManager(config=config, server=None)

    def tearDown(self):
        patch.stopall()

    def test_set_get_state_variables(self):
        self.mock_os.set_os_path_exists_return_value(False)
        self.state_manager.load_checkpoint(checkpoint_required=False)

        vars = [f"test_var{j}" for j in range(10)]
        for i, name in enumerate(vars):
            self.state_manager.set_state_variable(name, i)

        for i, name in enumerate(vars):
            self.assertEqual(self.state_manager.get_state_variable(name), i)

        for i, name in enumerate(vars):
            self.state_manager.set_state_variable(name, 9 - i)

        for i, name in enumerate(vars):
            self.assertEqual(self.state_manager.get_state_variable(name), 9 - i)

    def test_load_checkpoint(self):
        # Load checkpoint without ckpt files
        self.mock_os.set_os_path_exists_return_value(False)
        self.state_manager.load_checkpoint(checkpoint_required=False)
        self.assertTrue(self.state_manager.starting_fresh_run())

        # Load checkpoint files with ckpt files
        self.mock_os.set_os_path_exists_return_value(True)
        self.mock_os.set_os_path_join_return_value("0.ckpt")
        self.state_manager.load_checkpoint(checkpoint_required=False)
        self.assertFalse(self.state_manager.starting_fresh_run())

        # Load checkpoint throws error
        self.mock_json.set_json_load_side_effect(EOFError)
        with self.assertRaises(
            TritonModelAnalyzerException,
            msg="Checkpoint file 0.ckpt is"
            " empty or corrupted. Remove it from checkpoint"
            " directory.",
        ):
            self.mock_os.set_os_path_exists_return_value(True)
            self.mock_os.set_os_path_join_return_value("0.ckpt")
            self.state_manager.load_checkpoint(checkpoint_required=False)
            self.assertFalse(self.state_manager.starting_fresh_run())

    def test_latest_checkpoint(self):
        # No checkpoints
        self.mock_glob.set_glob_return_value([])
        self.assertEqual(self.state_manager._latest_checkpoint(), -1)

        # single checkpoint file
        for i in range(5):
            self.mock_glob.set_glob_return_value([f"{i}.ckpt"])
            self.assertEqual(self.state_manager._latest_checkpoint(), i)

        # Multiple checkpoint files consecutive, sorted
        self.mock_glob.set_glob_return_value([f"{i}.ckpt" for i in range(5)])
        self.assertEqual(self.state_manager._latest_checkpoint(), 4)

        # Multiple checkpoint files consecutive, unsorted
        self.mock_glob.set_glob_return_value([f"{i}.ckpt" for i in range(5, 1, -1)])
        self.assertEqual(self.state_manager._latest_checkpoint(), 5)

        # Multiple files nonconsecutive unsorted
        self.mock_glob.set_glob_return_value([f"{i}.ckpt" for i in [1, 3, 5, 2, 0, 4]])
        self.assertEqual(self.state_manager._latest_checkpoint(), 5)

        # Malformed checkpoint filename
        self.mock_glob.set_glob_return_value(["XYZ.ckpt"])
        with self.assertRaises(TritonModelAnalyzerException):
            self.state_manager._latest_checkpoint()


if __name__ == "__main__":
    unittest.main()
