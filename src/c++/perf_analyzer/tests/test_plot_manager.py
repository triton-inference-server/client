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

from model_analyzer.plots.plot_manager import PlotManager
from model_analyzer.result.constraint_manager import ConstraintManager
from model_analyzer.result.result_manager import ResultManager
from model_analyzer.state.analyzer_state_manager import AnalyzerStateManager

from .common import test_result_collector as trc
from .common.test_utils import (
    ROOT_DIR,
    convert_to_bytes,
    default_encode,
    evaluate_mock_config,
)


class TestPlotManager(trc.TestResultCollector):
    def setUp(self):
        self.create_new_dump = False
        self._create_single_model_result_manager()
        self._create_multi_model_result_manager()

    def tearDown(self):
        patch.stopall()

    def test_single_model_summary_plots_against_golden(self):
        """
        Match the summary plots against the golden versions in
        tests/common/single-model-ckpt
        """
        plot_manager = PlotManager(
            config=self._single_model_config,
            result_manager=self._single_model_result_manager,
            constraint_manager=MagicMock(),
        )

        plot_manager.create_summary_plots()

        # Set self.create_new_dump to True for creating a new output dump to compare against.
        # Default value is False.
        if self.create_new_dump:
            with open(f"{ROOT_DIR}/single-model-ckpt/plot_manager.json", "wb") as f:
                f.write(
                    convert_to_bytes(
                        json.dumps(
                            self._plot_manager_to_dict(plot_manager),
                            default=default_encode,
                        )
                    )
                )

        with open(f"{ROOT_DIR}/single-model-ckpt/plot_manager.json", "rb") as f:
            golden_plot_manager_dict = json.load(f)

        plot_manager_dict = self._plot_manager_to_dict(plot_manager)

        self.assertEqual(golden_plot_manager_dict, plot_manager_dict)

    def test_multi_model_summary_plots_against_golden(self):
        """
        Match the summary plots against the golden versions in
        tests/common/multi-model-ckpt
        """
        plot_manager = PlotManager(
            config=self._multi_model_config,
            result_manager=self._multi_model_result_manager,
            constraint_manager=MagicMock(),
        )

        plot_manager.create_summary_plots()

        # Set self.create_new_dump to True for creating a new output dump to compare against.
        # Default value is False.
        if self.create_new_dump:
            with open(f"{ROOT_DIR}/multi-model-ckpt/plot_manager.json", "wb") as f:
                f.write(
                    convert_to_bytes(
                        json.dumps(
                            self._plot_manager_to_dict(plot_manager),
                            default=default_encode,
                        )
                    )
                )

        with open(f"{ROOT_DIR}/multi-model-ckpt/plot_manager.json", "rb") as f:
            golden_plot_manager_dict = json.load(f)

        plot_manager_dict = self._plot_manager_to_dict(plot_manager)

        self.assertEqual(golden_plot_manager_dict, plot_manager_dict)

    def _create_single_model_result_manager(self):
        args = [
            "model-analyzer",
            "profile",
            "-f",
            "config.yml",
            "--checkpoint-directory",
            f"{ROOT_DIR}/single-model-ckpt/",
            "--export-path",
            f"{ROOT_DIR}/single-model-ckpt/",
            "--model-repository",
            ".",
        ]
        yaml_str = """
            profile_models: add_sub
        """
        config = evaluate_mock_config(args, yaml_str, subcommand="profile")
        state_manager = AnalyzerStateManager(config=config, server=None)
        state_manager.load_checkpoint(checkpoint_required=True)

        constraint_manager = ConstraintManager(config)
        self._single_model_result_manager = ResultManager(
            config=config,
            state_manager=state_manager,
            constraint_manager=constraint_manager,
        )

        self._single_model_config = config

    def _create_multi_model_result_manager(self):
        args = [
            "model-analyzer",
            "profile",
            "-f",
            "config.yml",
            "--checkpoint-directory",
            f"{ROOT_DIR}/multi-model-ckpt/",
            "--export-path",
            f"{ROOT_DIR}/multi-model-ckpt/",
            "--model-repository",
            ".",
            "--run-config-profile-models-concurrently-enable",
            "--run-config-search-mode",
            "quick",
        ]
        yaml_str = """
            profile_models: resnet50_libtorch,vgg19_libtorch
        """
        config = evaluate_mock_config(args, yaml_str, subcommand="profile")
        state_manager = AnalyzerStateManager(config=config, server=None)
        state_manager.load_checkpoint(checkpoint_required=True)

        constraint_manager = ConstraintManager(config)
        self._multi_model_result_manager = ResultManager(
            config=config,
            state_manager=state_manager,
            constraint_manager=constraint_manager,
        )

        self._multi_model_config = config

    def _plot_manager_to_dict(self, plot_manager):
        plot_manager_dict = {}
        plot_manager_dict["_simple_plots"] = {}
        for spd_key, simple_plot_dict in plot_manager._simple_plots.items():
            plot_manager_dict["_simple_plots"][spd_key] = {}
            for sp_key, simple_plot in simple_plot_dict.items():
                plot_manager_dict["_simple_plots"][spd_key][
                    sp_key
                ] = self._simple_plot_to_dict(simple_plot)

        return plot_manager_dict

    def _simple_plot_to_dict(self, simple_plot):
        simple_plot_dict = {}
        simple_plot_dict["_name"] = simple_plot._name
        simple_plot_dict["_title"] = simple_plot._title
        simple_plot_dict["_x_axis"] = simple_plot._x_axis
        simple_plot_dict["_y_axis"] = simple_plot._y_axis
        simple_plot_dict["_monotonic"] = simple_plot._monotonic
        simple_plot_dict["_data"] = simple_plot._data

        return simple_plot_dict


if __name__ == "__main__":
    unittest.main()
