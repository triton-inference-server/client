#!/usr/bin/env python3

# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import shutil
import unittest
from unittest.mock import MagicMock, patch

from model_analyzer.reports.report_manager import ReportManager
from model_analyzer.result.result_manager import ResultManager
from model_analyzer.state.analyzer_state_manager import AnalyzerStateManager

from .common import test_result_collector as trc
from .common.test_utils import ROOT_DIR, evaluate_mock_config


class TestBLSReportManager(trc.TestResultCollector):
    def _init_managers(
        self,
        models="test_model",
        num_configs_per_model=10,
        mode="online",
        subcommand="profile",
    ):
        args = ["model-analyzer", subcommand, "-f", "path-to-config-file"]
        args.extend(["--checkpoint-directory", f"{ROOT_DIR}/bls-ckpt"])

        if subcommand == "profile":
            args.extend(["--profile-models", models])
            args.extend(["--model-repository", "/tmp"])
        else:
            args.extend(["--report-model-configs", models])

        yaml_str = f"""
            client_protocol: grpc
            export_path: {ROOT_DIR}
        """
        config = evaluate_mock_config(args, yaml_str, subcommand=subcommand)
        state_manager = AnalyzerStateManager(config=config, server=None)

        state_manager.load_checkpoint(checkpoint_required=True)
        gpu_info = {"gpu_uuid": {"name": "fake_gpu_name", "total_memory": 1024000000}}
        self.result_manager = ResultManager(
            config=config, state_manager=state_manager, constraint_manager=MagicMock()
        )
        self.report_manager = ReportManager(
            mode=mode,
            config=config,
            gpu_info=gpu_info,
            result_manager=self.result_manager,
            constraint_manager=MagicMock(),
        )

    def test_bls_summary(self):
        """
        Ensures the summary report sentence and table are accurate for a basic bls model (loaded from a checkpoint)
        """
        self._init_managers(models="bls", subcommand="profile")

        self.report_manager.create_summaries()

        expected_summary_sentence = (
            "In 54 measurements across 9 configurations, <strong>bls_config_6</strong> "
            "is <strong>1111%</strong> better than the default configuration at maximizing throughput, "
            "under the given constraints, on GPU(s) TITAN RTX.<UL><LI> <strong>bls_config_6</strong>: 8 GPU instances with a max batch size of 0 on platform python "
            "</LI><BR>Which is comprised of the following composing models: "
            "<UL><LI> <strong>add_config_4</strong>: 6 GPU instances with a max batch size of 0 on platform python "
            "</LI><LI> <strong>sub_config_0</strong>: 1 GPU instance with a max batch size of 0 on platform python </LI> </UL>"
        )

        summary_table, summary_sentence = self.report_manager._build_summary_table(
            report_key="bls",
            num_measurements=54,
            num_configurations=9,
            gpu_name="TITAN RTX",
            report_gpu_metrics=True,
        )

        self.assertEqual(summary_sentence, expected_summary_sentence)

        # Check the first row of the table
        self.assertEqual(summary_table._rows[0][0], "bls_config_6")  # model config name
        self.assertEqual(summary_table._rows[0][1], "(0, 0)")  # max batch size
        self.assertEqual(
            summary_table._rows[0][2], "(Disabled, Disabled)"
        )  # dynamic batching
        self.assertEqual(summary_table._rows[0][3], "(6:GPU, 1:GPU)")  # instance count
        self.assertEqual(summary_table._rows[0][4], "2.215")  # p99 latency
        self.assertEqual(summary_table._rows[0][5], "10077.3")  # throughput
        self.assertEqual(summary_table._rows[0][6], 874)  # max gpu memory
        self.assertEqual(summary_table._rows[0][7], 0.0)  # GPU utilization

    def test_bls_summary_cpu_only(self):
        """
        Ensures the summary report sentence and table are accurate for a basic BLS model (loaded from a checkpoint)
        when the cpu only flag is set
        """

        self._init_managers(models="bls", subcommand="profile")

        self.report_manager.create_summaries()

        expected_summary_sentence = (
            "In 54 measurements across 9 configurations, <strong>bls_config_6</strong> "
            "is <strong>1111%</strong> better than the default configuration at maximizing throughput, "
            "under the given constraints.<UL><LI> <strong>bls_config_6</strong>: 8 GPU instances with a max batch size of 0 on platform python "
            "</LI><BR>Which is comprised of the following composing models: "
            "<UL><LI> <strong>add_config_4</strong>: 6 GPU instances with a max batch size of 0 on platform python "
            "</LI><LI> <strong>sub_config_0</strong>: 1 GPU instance with a max batch size of 0 on platform python </LI> </UL>"
        )

        summary_table, summary_sentence = self.report_manager._build_summary_table(
            report_key="bls",
            num_measurements=54,
            num_configurations=9,
            gpu_name="TITAN RTX",
            report_gpu_metrics=False,
        )

        self.assertEqual(summary_sentence, expected_summary_sentence)

        # Check the first row of the table
        self.assertEqual(summary_table._rows[0][0], "bls_config_6")  # model config name
        self.assertEqual(summary_table._rows[0][1], "0, 0")  # max batch size
        self.assertEqual(
            summary_table._rows[0][2], "(Disabled, Disabled)"
        )  # dynamic batching
        self.assertEqual(summary_table._rows[0][3], "6:GPU, 1:GPU")  # instance count
        self.assertEqual(summary_table._rows[0][4], "2.215")  # p99 latency
        self.assertEqual(summary_table._rows[0][5], "10077.3")  # throughput

    def test_bls_detailed(self):
        """
        Ensures the detailed report sentence is accurate for a BLS model (loaded from a checkpoint)
        """
        self._init_managers(models="bls_config_6", subcommand="report")

        self.report_manager._add_detailed_report_data()
        self.report_manager._build_detailed_table("bls_config_6")
        detailed_sentence = self.report_manager._build_detailed_info("bls_config_6")

        expected_detailed_sentence = (
            "<strong>bls_config_6</strong> is comprised of the "
            "following composing models:<LI> <strong>add_config_4</strong>: "
            "6 GPU instances with a max batch size of 0 on platform python </LI><LI> "
            "<strong>sub_config_0</strong>: 1 GPU instance with a max batch size of 0 "
            "on platform python </LI><br>12 measurement(s) "
            "were obtained for the model config on GPU(s)  with total memory 0 GB."
        )

        self.assertEqual(detailed_sentence, expected_detailed_sentence)

    def tearDown(self):
        patch.stopall()
        shutil.rmtree(f"{ROOT_DIR}/reports")
        shutil.rmtree(f"{ROOT_DIR}/plots")


if __name__ == "__main__":
    unittest.main()
