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

from model_analyzer.config.run.model_run_config import ModelRunConfig
from model_analyzer.config.run.run_config import RunConfig
from model_analyzer.perf_analyzer.perf_config import PerfAnalyzerConfig
from model_analyzer.reports.report_manager import ReportManager
from model_analyzer.result.constraint_manager import ConstraintManager
from model_analyzer.result.result_manager import ResultManager
from model_analyzer.result.run_config_result_comparator import RunConfigResultComparator
from model_analyzer.state.analyzer_state_manager import AnalyzerStateManager
from model_analyzer.triton.model.model_config import ModelConfig
from model_analyzer.triton.model.model_config_variant import ModelConfigVariant

from .common import test_result_collector as trc
from .common.test_utils import (
    construct_constraint_manager,
    construct_run_config_measurement,
    evaluate_mock_config,
)
from .mocks.mock_io import MockIOMethods
from .mocks.mock_json import MockJSONMethods
from .mocks.mock_matplotlib import MockMatplotlibMethods
from .mocks.mock_os import MockOSMethods


class TestReportManager(trc.TestResultCollector):
    def _init_managers(
        self,
        models="test_model",
        num_configs_per_model=10,
        mode="online",
        subcommand="profile",
        report_gpu_metrics=False,
    ):
        args = ["model-analyzer", subcommand, "-f", "path-to-config-file"]
        if subcommand == "profile":
            args.extend(["--profile-models", models])
            args.extend(["--model-repository", "/tmp"])
        else:
            args.extend(["--report-model-configs", models])

        if report_gpu_metrics:
            args.extend(["--always-report-gpu-metrics"])

        yaml_str = (
            """
            num_configs_per_model: """
            + str(num_configs_per_model)
            + """
            client_protocol: grpc
            export_path: /test/export/path
            constraints:
              perf_latency_p99:
                max: 100
              gpu_used_memory:
                max: 10000
        """
        )
        config = evaluate_mock_config(args, yaml_str, subcommand=subcommand)
        state_manager = AnalyzerStateManager(config=config, server=None)

        gpu_info = {"gpu_uuid": {"name": "fake_gpu_name", "total_memory": 1024000000}}
        constraint_manager = ConstraintManager(config=config)

        self.result_manager = ResultManager(
            config=config,
            state_manager=state_manager,
            constraint_manager=constraint_manager,
        )
        self.report_manager = ReportManager(
            mode=mode,
            config=config,
            gpu_info=gpu_info,
            result_manager=self.result_manager,
            constraint_manager=constraint_manager,
        )

    def _add_result_measurement(
        self,
        model_config_name,
        model_name,
        avg_gpu_metrics,
        avg_non_gpu_metrics,
        result_comparator,
        cpu_only=False,
        add_to_results_only=False,
    ):
        config_pb = self.model_config.copy()
        config_pb["name"] = model_config_name
        model_config = ModelConfig.create_from_dictionary(config_pb)
        model_config_variant = ModelConfigVariant(
            model_config, model_config_name, cpu_only
        )

        measurement = construct_run_config_measurement(
            model_name=model_name,
            model_config_names=[model_config_name],
            model_specific_pa_params=MagicMock(),
            gpu_metric_values=avg_gpu_metrics,
            non_gpu_metric_values=[avg_non_gpu_metrics],
            metric_objectives=result_comparator._metric_weights,
            model_config_weights=[1],
        )

        perf_config = PerfAnalyzerConfig()
        perf_config.update_config({"model-name": model_config_name})
        mrc = ModelRunConfig(model_name, model_config_variant, perf_config)
        run_config = RunConfig({})
        run_config.add_model_run_config(mrc)

        if add_to_results_only:
            self.result_manager._add_rcm_to_results(run_config, measurement)
        else:
            self.result_manager.add_run_config_measurement(run_config, measurement)

    def setUp(self):
        self.model_config = {
            "platform": "tensorflow_graphdef",
            "instance_group": [{"count": 1, "kind": "KIND_GPU"}],
            "max_batch_size": 8,
            "dynamic_batching": {},
        }

        self.os_mock = MockOSMethods(
            mock_paths=[
                "model_analyzer.reports.report_manager",
                "model_analyzer.plots.plot_manager",
                "model_analyzer.state.analyzer_state_manager",
                "model_analyzer.config.input.config_utils",
                "model_analyzer.config.input.config_command_profile",
                "model_analyzer.config.input.config_command_report",
            ]
        )
        self.os_mock.start()
        self.os_mock.set_os_path_isfile_return_value(False)
        # Required patch ordering here
        # html_report must be patched before pdf_report
        # Likely due to patching dealing with parent + child classes
        self.io_mock = MockIOMethods(
            mock_paths=[
                "model_analyzer.reports.html_report",
                "model_analyzer.reports.pdf_report",
                "model_analyzer.state.analyzer_state_manager",
            ],
            read_data=[bytes(">:(".encode("ascii"))],
        )
        self.io_mock.start()
        self.matplotlib_mock = MockMatplotlibMethods()
        self.matplotlib_mock.start()
        self.json_mock = MockJSONMethods()
        self.json_mock.start()

    def test_add_results(self, *args):
        for mode in ["online", "offline"]:
            self._init_managers(
                models="test_model1,test_model2", mode=mode, subcommand="profile"
            )
            result_comparator = RunConfigResultComparator(
                metric_objectives_list=[{"perf_throughput": 10}], model_weights=[1]
            )

            avg_gpu_metrics = {0: {"gpu_used_memory": 6000, "gpu_utilization": 60}}

            for i in range(10):
                avg_non_gpu_metrics = {
                    "perf_throughput": 100 + 10 * i,
                    "perf_latency_p99": 4000,
                    "cpu_used_ram": 1000,
                }
                self._add_result_measurement(
                    f"test_model1_config_report_{i}",
                    "test_model1",
                    avg_gpu_metrics,
                    avg_non_gpu_metrics,
                    result_comparator,
                    add_to_results_only=False,
                )

            for i in range(5):
                avg_non_gpu_metrics = {
                    "perf_throughput": 200 + 10 * i,
                    "perf_latency_p99": 4000,
                    "cpu_used_ram": 1000,
                }
                self._add_result_measurement(
                    f"test_model2_config_report_{i}",
                    "test_model2",
                    avg_gpu_metrics,
                    avg_non_gpu_metrics,
                    result_comparator,
                    add_to_results_only=False,
                )

            self.report_manager.create_summaries()
            self.assertEqual(
                self.report_manager.report_keys(), ["test_model1", "test_model2"]
            )

            report1_data = self.report_manager.data("test_model1")
            report2_data = self.report_manager.data("test_model2")

            self.assertEqual(len(report1_data), 10)
            self.assertEqual(len(report2_data), 5)

    @patch(
        "model_analyzer.reports.report_manager.ReportManager._find_default_run_config_measurement",
        return_value=100,
    )
    @patch(
        "model_analyzer.reports.report_manager.ReportManager._get_objective_gain",
        return_value=100,
    )
    def test_build_summary_table(self, *args):
        for mode in ["offline", "online"]:
            for cpu_only in [True, False]:
                for report_gpu_metrics in [True, False]:
                    self.subtest_build_summary_table(mode, cpu_only, report_gpu_metrics)

    def subtest_build_summary_table(self, mode, cpu_only, report_gpu_metrics):
        self._init_managers(
            models="test_model",
            mode=mode,
            subcommand="profile",
            report_gpu_metrics=report_gpu_metrics,
        )
        result_comparator = RunConfigResultComparator(
            metric_objectives_list=[{"perf_throughput": 10}], model_weights=[1]
        )

        avg_gpu_metrics = {0: {"gpu_used_memory": 6000, "gpu_utilization": 60}}

        gpu_metrics = report_gpu_metrics or not cpu_only

        for i in range(10, 0, -1):
            avg_non_gpu_metrics = {
                "perf_throughput": 100 + 10 * i,
                "perf_latency_p99": 4000,
                "cpu_used_ram": 1000,
            }
            self._add_result_measurement(
                f"test_model_config_{i}",
                "test_model",
                avg_gpu_metrics,
                avg_non_gpu_metrics,
                result_comparator,
                cpu_only=not gpu_metrics,
            )

        self.report_manager.create_summaries()

        summary_table, summary_sentence = self.report_manager._build_summary_table(
            report_key="test_model",
            num_measurements=10,
            num_configurations=3,
            gpu_name="TITAN RTX",
            report_gpu_metrics=gpu_metrics,
        )

        if mode == "online":
            objective = "maximizing throughput"
        else:
            objective = "minimizing latency"

        if gpu_metrics:
            expected_summary_sentence = (
                "In 10 measurements across 3 configurations, "
                "<strong>test_model_config_10</strong> is <strong>100%</strong> better than the default configuration "
                f"at {objective}, under the given constraints, on GPU(s) TITAN RTX.<UL><LI> "
                "<strong>test_model_config_10</strong>: 1 GPU instance with a max batch size of 8 on platform tensorflow_graphdef "
                "</LI> </UL>"
            )
        else:
            expected_summary_sentence = (
                "In 10 measurements across 3 configurations, "
                "<strong>test_model_config_10</strong> is <strong>100%</strong> better than the default configuration "
                f"at {objective}, under the given constraints.<UL><LI> "
                "<strong>test_model_config_10</strong>: 1 GPU instance with a max batch size of 8 on platform tensorflow_graphdef "
                "</LI> </UL>"
            )

        self.assertEqual(len(expected_summary_sentence), len(summary_sentence))
        self.assertEqual(expected_summary_sentence, summary_sentence)

        # Get throughput index and make sure results are sorted
        throughput_index = summary_table.headers().index("Throughput (infer/sec)")
        model_name_index = summary_table.headers().index("Model Config Name")
        for i in range(9):
            current_row = summary_table.get_row_by_index(i)
            next_row = summary_table.get_row_by_index(i + 1)
            self.assertEqual(current_row[model_name_index], f"test_model_config_{10-i}")
            self.assertGreaterEqual(
                current_row[throughput_index], next_row[throughput_index]
            )

    def test_build_detailed_info(self):
        for cpu_only in [True, False]:
            for report_gpu_metrics in [True, False]:
                self._subtest_build_detailed_info(cpu_only, report_gpu_metrics)

    def _subtest_build_detailed_info(self, cpu_only, report_gpu_metrics):
        self._init_managers(models="test_model_config_10", subcommand="report")

        result_comparator = RunConfigResultComparator(
            metric_objectives_list=[{"perf_throughput": 10}], model_weights=[1]
        )

        gpu_metrics = report_gpu_metrics or not cpu_only

        avg_gpu_metrics = {"gpu_uuid": {"gpu_used_memory": 6000, "gpu_utilization": 60}}

        for i in range(10, 0, -1):
            avg_non_gpu_metrics = {
                "perf_throughput": 100 + 10 * i,
                "perf_latency_p99": 4000,
                "cpu_used_ram": 1000,
            }
            self._add_result_measurement(
                f"test_model_config_{i}",
                "test_model",
                avg_gpu_metrics,
                avg_non_gpu_metrics,
                result_comparator,
                cpu_only=not gpu_metrics,
                add_to_results_only=True,
            )

        self.report_manager._add_detailed_report_data()
        self.report_manager._build_detailed_table("test_model_config_10")
        sentence = self.report_manager._build_detailed_info("test_model_config_10")

        if gpu_metrics:
            expected_sentence = (
                f"The model config <strong>test_model_config_10</strong> uses 1 GPU instance with "
                f"a max batch size of 8 and has dynamic batching enabled. 1 measurement(s) "
                f"were obtained for the model config on GPU(s) 1 x fake_gpu_name with total memory 1.0 GB. "
                f"This model uses the platform tensorflow_graphdef."
            )
        else:
            expected_sentence = (
                f"The model config <strong>test_model_config_10</strong> uses 1 GPU instance with "
                f"a max batch size of 8 and has dynamic batching enabled. 1 measurement(s) "
                f"were obtained for the model config on CPU. "
                f"This model uses the platform tensorflow_graphdef."
            )

        self.assertEqual(expected_sentence, sentence)

    @patch("model_analyzer.plots.plot_manager.PlotManager._create_update_simple_plot")
    @patch("model_analyzer.result.result_table.ResultTable.insert_row_by_index")
    def test_summary_default_within_top(self, add_table_fn, add_plot_fn):
        """
        Test summary report generation when default is in the top n configs

        Creates some results where the default config is within the top n configs,
        and then confirms that the number of entries added to plots and tables
        is correct
        """

        default_within_top = True
        top_n = 3
        self._test_summary_counts(add_table_fn, add_plot_fn, default_within_top, top_n)

    @patch("model_analyzer.plots.plot_manager.PlotManager._create_update_simple_plot")
    @patch("model_analyzer.result.result_table.ResultTable.insert_row_by_index")
    def test_summary_default_not_within_top(self, add_table_fn, add_plot_fn):
        """
        Test summary report generation when default is not in the top n configs

        Creates some results where the default config is not within the top n configs,
        and then confirms that the number of entries added to plots and tables
        is correct such that it includes the default config data
        """
        default_within_top = False
        top_n = 3
        self._test_summary_counts(add_table_fn, add_plot_fn, default_within_top, top_n)

    def _test_summary_counts(
        self, add_table_fn, add_plot_fn, default_within_top, top_n
    ):
        """
        Helper function to test creating summary reports and confirming that the number
        of entries added to plots and tables is as expected
        """
        num_plots_in_summary_report = 2
        num_tables_in_summary_report = 1
        expected_config_count = top_n + 1 if not default_within_top else top_n
        expected_plot_count = num_plots_in_summary_report * expected_config_count
        expected_table_count = num_tables_in_summary_report * expected_config_count

        self._init_managers(
            models="test_model1", num_configs_per_model=top_n, subcommand="profile"
        )
        result_comparator = RunConfigResultComparator(
            metric_objectives_list=[{"perf_throughput": 10}], model_weights=[1]
        )
        avg_gpu_metrics = {0: {"gpu_used_memory": 6000, "gpu_utilization": 60}}
        for i in range(10):
            # Create a bunch of fake measurement results
            #
            # For the 'default_within_top' case, have the throughput
            # be decreasing for each config (so the highest throughput is the
            # first, default config)
            #
            # For the default not within top case, have the throughput be
            # increasing for each config (so the highest throughput is the last
            # config, and the default config is the worst)
            #
            p99 = 20 + i
            throughput = 100 - 10 * i if default_within_top else 200 + 10 * i
            avg_non_gpu_metrics = {
                "perf_throughput": throughput,
                "perf_latency_p99": p99,
                "cpu_used_ram": 1000,
            }
            name = f"test_model1_config_{i}"
            if not i:
                name = f"test_model1_config_default"
            self._add_result_measurement(
                name,
                "test_model1",
                avg_gpu_metrics,
                avg_non_gpu_metrics,
                result_comparator,
            )
        self.report_manager.create_summaries()

        self.assertEqual(expected_plot_count, add_plot_fn.call_count)
        self.assertEqual(expected_table_count, add_table_fn.call_count)

    def test_create_instance_group_phrase(self):
        """Test all corner cases of _create_instance_group_phrase()"""
        self._init_managers(models="test_model", subcommand="profile")

        model_config_dict = {
            "instance_group": [
                {
                    "count": 1,
                    "kind": "KIND_GPU",
                }
            ]
        }
        self._test_instance_group_helper(model_config_dict, "1 GPU instance")

        model_config_dict = {
            "instance_group": [
                {
                    "count": 2,
                    "kind": "KIND_GPU",
                }
            ]
        }
        self._test_instance_group_helper(model_config_dict, "2 GPU instances")

        model_config_dict = {
            "instance_group": [
                {
                    "count": 3,
                    "kind": "KIND_GPU",
                },
                {
                    "count": 2,
                    "kind": "KIND_CPU",
                },
            ]
        }
        self._test_instance_group_helper(
            model_config_dict, "3 GPU instances and 2 CPU instances"
        )

    def _test_instance_group_helper(self, model_config_dict, expected_output):
        model_config = ModelConfig.create_from_dictionary(model_config_dict)
        output = self.report_manager._create_instance_group_phrase(model_config)
        self.assertEqual(output, expected_output)

    def test_gpu_name_and_memory(self):
        """
        Test return value of _get_gpu_stats()

        Creates a scenario where 4 GPUs are visible, but only 3 are
        used (1 measurement uses 1 and 2, other uses 1 and 4)

        The function should note that 1 and 2 are the same GPU type and combine
        them in the name string, and it should combine the total memory of the used
        GPUs
        """
        gpu_info = {
            "gpu_uuid1": {"name": "fake_gpu_name", "total_memory": 2**30 * 2},  # 2GB
            "gpu_uuid2": {"name": "fake_gpu_name", "total_memory": 2**30 * 2},  # 2GB
            "gpu_uuid3": {
                "name": "fake_gpu_name_2",
                "total_memory": 2**30 * 4,  # 4GB
            },
            "gpu_uuid4": {
                "name": "fake_gpu_name_3",
                "total_memory": 2**30 * 8,  # 8GB
            },
        }

        report_manager = ReportManager(
            mode=MagicMock(),
            config=MagicMock(),
            gpu_info=gpu_info,
            result_manager=MagicMock(),
            constraint_manager=MagicMock(),
        )

        avg_gpu_metrics1 = {
            "gpu_uuid1": {"gpu_used_memory": 6000, "gpu_utilization": 60},
            "gpu_uuid2": {"gpu_used_memory": 6000, "gpu_utilization": 60},
        }

        avg_gpu_metrics2 = {
            "gpu_uuid1": {"gpu_used_memory": 6000, "gpu_utilization": 60},
            "gpu_uuid4": {"gpu_used_memory": 6000, "gpu_utilization": 60},
        }

        measurement1 = construct_run_config_measurement(
            model_name=MagicMock(),
            model_config_names=MagicMock(),
            model_specific_pa_params=MagicMock(),
            gpu_metric_values=avg_gpu_metrics1,
            constraint_manager=MagicMock(),
            non_gpu_metric_values=MagicMock(),
            metric_objectives=MagicMock(),
            model_config_weights=MagicMock(),
        )

        measurement2 = construct_run_config_measurement(
            model_name=MagicMock(),
            model_config_names=MagicMock(),
            model_specific_pa_params=MagicMock(),
            gpu_metric_values=avg_gpu_metrics2,
            constraint_manager=MagicMock(),
            non_gpu_metric_values=MagicMock(),
            metric_objectives=MagicMock(),
            model_config_weights=MagicMock(),
        )

        measurements = [measurement1, measurement2]
        names, max_mem = report_manager._get_gpu_stats(measurements)
        self.assertEqual(names, "2 x fake_gpu_name, 1 x fake_gpu_name_3")
        self.assertEqual(max_mem, "12.0 GB")

    @patch(
        "model_analyzer.reports.report_manager.ReportManager._build_constraint_strings",
        return_value={"modelA": "Max p99 latency: 100 ms"},
    )
    def test_constraint_string_single_model(self, *args):
        constraint_manager = construct_constraint_manager(
            """
            profile_models:
              modelA:
                constraints:
                  perf_latency_p99:
                    max: 100
            """
        )

        report_manager = ReportManager(
            mode=MagicMock(),
            config=MagicMock(),
            gpu_info=MagicMock(),
            result_manager=MagicMock(),
            constraint_manager=constraint_manager,
        )
        expected_constraint_str = "Max p99 latency: 100 ms"
        actual_constraint_str = report_manager._create_constraint_string(
            report_key="modelA"
        )

        self.assertEqual(actual_constraint_str, expected_constraint_str)

    @patch(
        "model_analyzer.reports.report_manager.ReportManager._build_constraint_strings",
        return_value={
            "modelA": "Max p99 latency: 100 ms",
            "modelB": "Max p99 latency: 200 ms",
        },
    )
    def test_constraint_string_multi_model(self, *args):
        constraint_manager = construct_constraint_manager(
            """
            profile_models:
              modelA:
                constraints:
                  perf_latency_p99:
                    max: 100
              modelB:
                constraints:
                  perf_latency_p99:
                    max: 200
            """
        )

        report_manager = ReportManager(
            mode=MagicMock(),
            config=MagicMock(),
            gpu_info=MagicMock(),
            result_manager=MagicMock(),
            constraint_manager=constraint_manager,
        )
        expected_constraint_str = "<strong>modelA</strong>: Max p99 latency: 100 ms"
        expected_constraint_str += "<br>"

        for i in range(len("Constraint targets: ")):
            expected_constraint_str += "&ensp;"

        expected_constraint_str += "<strong>modelB</strong>: Max p99 latency: 200 ms"

        actual_constraint_str = report_manager._create_constraint_string(
            report_key="modelA,modelB"
        )

        self.assertEqual(actual_constraint_str, expected_constraint_str)

    def tearDown(self):
        self.matplotlib_mock.stop()
        self.io_mock.stop()
        self.os_mock.stop()
        self.json_mock.stop()
        patch.stopall()


if __name__ == "__main__":
    unittest.main()
