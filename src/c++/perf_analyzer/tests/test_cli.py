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

# This test exercises all of the cli options.
# It does a basic test of the cli and help message.
# Then the rest of the testing uses an OptionStruct, which holds all
# of the data necessary to test a command, and feeds that to the
# CLI parser. The result of the CLI parsing is compared against the
# expected value for the CLI. Default values are also verified as well as
# values that are expected to cause failures.

import copy
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import psutil

from model_analyzer.cli.cli import CLI
from model_analyzer.config.input.config_command_profile import ConfigCommandProfile
from model_analyzer.config.input.config_command_report import ConfigCommandReport
from model_analyzer.config.input.config_defaults import DEFAULT_TRITON_DOCKER_IMAGE
from model_analyzer.config.input.config_status import ConfigStatus
from model_analyzer.constants import CONFIG_PARSER_SUCCESS
from model_analyzer.model_analyzer_exceptions import TritonModelAnalyzerException

from .common import test_result_collector as trc


def tearDown(self):
    patch.stopall()


def get_test_options():
    """
    Returns the list of OptionStructs that are used for testing.
    """
    # yapf: disable
    options = [
        #Boolean options
        # Options format:
        #   (bool, MA step, long_option)
        OptionStruct("bool", "profile","--override-output-model-repository"),
        OptionStruct("bool", "profile","--collect-cpu-metrics"),
        OptionStruct("bool", "profile","--perf-output"),
        OptionStruct("bool", "profile","--run-config-search-disable"),
        OptionStruct("bool", "profile","--run-config-profile-models-concurrently-enable"),
        OptionStruct("bool", "profile","--request-rate-search-enable"),
        OptionStruct("bool", "profile","--reload-model-disable"),
        OptionStruct("bool", "profile","--early-exit-enable"),
        OptionStruct("bool", "profile","--skip-summary-reports"),
        OptionStruct("bool", "profile","--skip-detailed-reports"),
        OptionStruct("bool", "profile","--always-report-gpu-metrics"),
        #Int/Float options
        # Options format:
        #   (int/float, MA step, long_option, short_option, test_value, expected_default_value)
        # The following options can be None:
        #   short_option
        #   expected_default_value
        OptionStruct("int", "profile", "--client-max-retries", "-r", "125", "50"),
        OptionStruct("int", "profile", "--duration-seconds", "-d", "10", "3"),
        OptionStruct("int", "profile", "--perf-analyzer-timeout", None, "100", "600"),
        OptionStruct("int", "profile", "--perf-analyzer-max-auto-adjusts", None, "100", "10"),
        OptionStruct("int", "profile", "--run-config-search-min-concurrency", None, "2", "1"),
        OptionStruct("int", "profile", "--run-config-search-max-concurrency", None, "100", "1024"),
        OptionStruct("int", "profile", "--run-config-search-min-request-rate", None, "2", "16"),
        OptionStruct("int", "profile", "--run-config-search-max-request-rate", None, "100", "8192"),
        OptionStruct("int", "profile", "--run-config-search-min-model-batch-size", None, "100", "1"),
        OptionStruct("int", "profile", "--run-config-search-max-model-batch-size", None, "100", "128"),
        OptionStruct("int", "profile", "--run-config-search-min-instance-count", None, "2", "1"),
        OptionStruct("int", "profile", "--run-config-search-max-instance-count", None, "10", "5"),
        OptionStruct("int", "profile", "--run-config-search-max-binary-search-steps", None, "10", "5"),
        OptionStruct("float", "profile", "--monitoring-interval", "-i", "10.0", "1.0"),
        OptionStruct("float", "profile", "--perf-analyzer-cpu-util", None, "10.0", str(psutil.cpu_count() * 80.0)),
        OptionStruct("int", "profile", "--num-configs-per-model", None, "10", "3"),
        OptionStruct("int", "profile", "--num-top-model-configs", None, "10", "0"),
        OptionStruct("int", "profile", "--latency-budget", None, "200", None),
        OptionStruct("int", "profile", "--min-throughput", None, "300", None),

        #String options
        # Options format:
        #   (string, MA step, long_flag, short_flag, test_value, expected_default_value, expected_failing_value, extra_commands)
        # The following options can be None:
        #   short_flag
        #   expected_default_value
        #   expected_failing_value
        # For options with choices, list the test_values in a list of strings
        OptionStruct("string", "profile", "--config-file", "-f", "baz", None, None),
        OptionStruct("string", "profile", "--checkpoint-directory", "-s", "./test_dir", os.path.join(os.getcwd(), "checkpoints"), None),
        OptionStruct("string", "profile", "--output-model-repository-path", None, "./test_dir", os.path.join(os.getcwd(), "output_model_repository"), None),
        OptionStruct("string", "profile", "--client-protocol", None, ["http", "grpc"], "grpc", "SHOULD_FAIL"),
        OptionStruct("string", "profile", "--perf-analyzer-path", None, ".", "perf_analyzer", None),
        OptionStruct("string", "profile", "--perf-output-path", None, ".", None, None),
        OptionStruct("string", "profile", "--triton-docker-image", None, "test_image", DEFAULT_TRITON_DOCKER_IMAGE, None),
        OptionStruct("string", "profile", "--triton-http-endpoint", None, "localhost:4000", "localhost:8000", None),
        OptionStruct("string", "profile", "--triton-grpc-endpoint", None, "localhost:4001", "localhost:8001", None),
        OptionStruct("string", "profile", "--triton-metrics-url", None, "localhost:4002", "http://localhost:8002/metrics", None),
        OptionStruct("string", "profile", "--triton-server-path", None, "test_path", "tritonserver", None),
        OptionStruct("string", "profile", "--triton-output-path", None, "test_path", None, None),
        OptionStruct("string", "profile", "--triton-launch-mode", None, ["local", "docker", "remote","c_api"], "local", "SHOULD_FAIL"),
        OptionStruct("string", "profile", "--triton-install-path", None, "test_path", "/opt/tritonserver", None),
        OptionStruct("string", "profile", "--checkpoint-directory", "-s", "./test_dir", os.path.join(os.getcwd(), "checkpoints"), None),
        OptionStruct("string", "profile", "--export-path", "-e", "./test_dir", os.getcwd(), None),
        OptionStruct("string", "profile", "--filename-model-inference", None, "foo", "metrics-model-inference.csv", None),
        OptionStruct("string", "profile", "--filename-model-gpu", None, "foo", "metrics-model-gpu.csv", None),
        OptionStruct("string", "profile", "--filename-server-only", None, "foo", "metrics-server-only.csv", None),
        OptionStruct("string", "profile", "--config-file", "-f", "baz", None, None),

        OptionStruct("string", "report", "--checkpoint-directory", "-s", "./test_dir", os.path.join(os.getcwd(), "checkpoints"), None),
        OptionStruct("string", "report", "--export-path", "-e", "./test_dir", os.getcwd(), None),
        OptionStruct("string", "report", "--config-file", "-f", "baz", None, None),
        OptionStruct("string", "profile", "--triton-docker-shm-size", None, "1G", None, extra_commands=["--triton-launch-mode", "docker"]),
        OptionStruct("string", "profile","--run-config-search-mode", None, ["quick", "brute"], "brute", "SHOULD_FAIL"),

        #List Options:
        # Options format:
        #   (intlist/stringlist, MA step, long_flag, short_flag, test_value, expected_default_value, extra_commands)
        # The following options can be None:
        #   short_flag
        #   expected_default_value
        OptionStruct("intlist", "profile", "--batch-sizes", "-b", "2, 4, 6", "1"),
        OptionStruct("intlist", "profile", "--concurrency", "-c", "1, 2, 3", None),
        OptionStruct("intlist", "profile", "--request-rate", None, "1, 2, 3", None),
        OptionStruct("stringlist", "profile", "--triton-docker-mounts", None, "a:b:c, d:e:f", None, extra_commands=["--triton-launch-mode", "docker"]),
        OptionStruct("stringlist", "profile", "--gpus", None, "a, b, c", "all"),
        OptionStruct("stringlist", "profile", "--inference-output-fields", None, "a, b, c",
            "model_name,batch_size,concurrency,model_config_path,instance_group,max_batch_size,satisfies_constraints,perf_throughput,perf_latency_p99"),
        OptionStruct("stringlist", "profile", "--gpu-output-fields", None, "a, b, c",
            "model_name,gpu_uuid,batch_size,concurrency,model_config_path,instance_group,satisfies_constraints,gpu_used_memory,gpu_utilization,gpu_power_usage"),
        OptionStruct("stringlist", "profile", "--server-output-fields", None, "a, b, c",
            "model_name,gpu_uuid,gpu_used_memory,gpu_utilization,gpu_power_usage"),

        # No OP Options:
        # Option format:
        # (noop, any MA step, long_flag)
        # These commands aren't tested directly but are here to ensure that
        # the count is correct for all options in the config.
        # Some of these are required to run the subcommand
        # Others are yaml only options
        OptionStruct("noop", "profile", "--model-repository"),
        OptionStruct("noop", "profile", "--profile-models"),
        OptionStruct("noop", "profile", "--bls-composing-models"),
        OptionStruct("noop", "profile", "--cpu-only-composing-models"),

        OptionStruct("noop", "report", "--report-model-configs"),
        OptionStruct("noop", "report", "--output-formats", "-o", ["pdf", "csv", "png"], "pdf", "SHOULD_FAIL"),
        OptionStruct("noop", "yaml_profile", "constraints"),
        OptionStruct("noop", "yaml_profile", "objectives"),
        OptionStruct("noop", "yaml_profile", "weighting"),
        OptionStruct("noop", "yaml_profile", "triton_server_flags"),
        OptionStruct("noop", "yaml_profile", "perf_analyzer_flags"),
        OptionStruct("noop", "yaml_profile", "triton_docker_labels"),
        OptionStruct("noop", "yaml_profile", "triton_server_environment"),
        OptionStruct("noop", "yaml_profile", "triton_docker_args"),
        OptionStruct("noop", "yaml_profile", "plots")
    ]
    # yapf: enable
    return options


class CLISubclass(CLI):
    """
    Subclass of CLI to overwrite the parse method.
    Parse takes a list of arguments instead of getting the args
    from sys.argv
    """

    def __init__(self):
        super().__init__()

    def parse(self, parsed_commands=None):
        args = self._parser.parse_args(parsed_commands[1:])
        if args.subcommand is None:
            self._parser.print_help()
            self._parser.exit()
        config = self._subcommand_configs[args.subcommand]
        config.set_config_values(args)
        return args, config


class CLIConfigProfileStruct:
    """
    Struct class to hold the common variables shared between profile tests
    """

    def __init__(self):
        # yapf: disable
        self.args = [
            '/usr/local/bin/model-analyzer',
            'profile',
            '--model-repository',
            'foo',
            '--profile-models',
            'bar'
        ]
        # yapf: enable
        config_profile = ConfigCommandProfile()
        self.cli = CLISubclass()
        self.cli.add_subcommand(cmd="profile", help="", config=config_profile)

    def parse(self):
        return self.cli.parse(self.args)


class CLIConfigReportStruct:
    """
    Struct class to hold the common variables shared between report tests
    """

    def __init__(self):
        # yapf: disable
        self.args = [
            '/usr/local/bin/model-analyzer',
            'report',
            '--report-model-configs',
            'a, b, c'
        ]
        # yapf: enable
        config_report = ConfigCommandReport()
        self.cli = CLISubclass()
        self.cli.add_subcommand(cmd="report", help="", config=config_report)

    def parse(self):
        return self.cli.parse(self.args)


class OptionStruct:
    """
    Struct that holds all of the data necessary to test a single command
    """

    def __init__(
        self,
        type,
        stage,
        long_flag,
        short_flag=None,
        expected_value=None,
        expected_default_value=None,
        expected_failing_value=None,
        extra_commands=None,
    ):
        self.long_flag = long_flag
        self.short_flag = short_flag
        self.expected_value = expected_value
        self.expected_default_value = expected_default_value
        self.expected_failing_value = expected_failing_value
        self.type = type
        self.extra_commands = extra_commands

        if stage == "profile":
            self.cli_subcommand = CLIConfigProfileStruct
        elif stage == "report":
            self.cli_subcommand = CLIConfigReportStruct


@patch(
    "model_analyzer.config.input.config_command_profile.file_path_validator",
    lambda _: ConfigStatus(status=CONFIG_PARSER_SUCCESS),
)
@patch(
    "model_analyzer.config.input.config_command_profile.binary_path_validator",
    lambda _: ConfigStatus(status=CONFIG_PARSER_SUCCESS),
)
@patch(
    "model_analyzer.config.input.config_command_report.file_path_validator",
    lambda _: ConfigStatus(status=CONFIG_PARSER_SUCCESS),
)
class TestCLI(trc.TestResultCollector):
    """
    Tests the methods of the CLI class
    """

    @patch("model_analyzer.cli.cli.ArgumentParser.print_help")
    def test_help_message_no_args(self, mock_print_help):
        """
        Tests that model-analyzer prints the help message when no arguments are
        given
        """

        sys.argv = ["/usr/local/bin/model-analyzer"]

        cli = CLI()

        self.assertRaises(SystemExit, cli.parse)
        mock_print_help.assert_called()

    def test_basic_cli_config_profile_options(self):
        """
        Test the minimal set of cli commands necessary to run Model Analyzer profile
        """
        cli = CLIConfigProfileStruct()
        _, config = cli.parse()
        model_repo = config.model_repository
        profile_model = config.profile_models[0].model_name()
        self.assertEqual("foo", model_repo)
        self.assertEqual("bar", profile_model)

    @patch(
        "model_analyzer.config.input.config_command_report.ConfigCommandReport._load_config_file",
        MagicMock(),
    )
    @patch(
        "model_analyzer.config.input.config_command_report.ConfigCommandReport._preprocess_and_verify_arguments",
        MagicMock(),
    )
    @patch(
        "model_analyzer.config.input.config_command_profile.ConfigCommandProfile._preprocess_and_verify_arguments",
        MagicMock(),
    )
    @patch(
        "model_analyzer.config.input.config_command_profile.ConfigCommandProfile._load_config_file",
        MagicMock(),
    )
    @patch(
        "model_analyzer.config.input.config_command.ConfigCommand._check_for_illegal_config_settings",
        MagicMock(),
    )
    def test_all_options(self):
        options = get_test_options()
        all_tested_options_set = set()

        for option in options:
            all_tested_options_set.add(
                self._convert_flag_to_use_underscores(option.long_flag)
            )

            if option.type in ["bool"]:
                self._test_boolean_option(option)
            elif option.type in ["int", "float"]:
                self._test_numeric_option(option)
            elif option.type in ["string"]:
                self._test_string_option(option)
            elif option.type in ["intlist", "stringlist"]:
                self._test_list_option(option)
            elif option.type in ["noop"]:
                pass
            else:
                raise (TritonModelAnalyzerException("Invalid option type"))

        self._verify_all_options_tested(all_tested_options_set)

    def _verify_all_options_tested(self, all_tested_options_set):
        cli_option_set = set()

        # Get all of the options in the CLI Configs
        structs = [CLIConfigProfileStruct, CLIConfigReportStruct]
        for struct in structs:
            cli = struct()
            _, config = cli.parse()
            for key in config.get_config().keys():
                cli_option_set.add(key)

        self.assertEqual(
            cli_option_set,
            all_tested_options_set,
            "The available options for the CLI does not match the available options tested. "
            "If you recently added or removed a CLI option, please update the OptionStruct list above.",
        )

    def _test_boolean_option(self, option_struct):
        option = option_struct.long_flag
        option_with_underscores = self._convert_flag_to_use_underscores(option)
        cli = option_struct.cli_subcommand()
        _, config = cli.parse()
        option_value = config.get_config().get(option_with_underscores).value()
        self.assertEqual(option_value, False)

        # Test boolean option
        cli = option_struct.cli_subcommand()
        cli.args.extend([option])
        _, config = cli.parse()
        option_value = config.get_config().get(option_with_underscores).value()
        self.assertEqual(option_value, True)

        # Test boolean option followed by value fails
        cli = option_struct.cli_subcommand()
        cli.args.extend([option, "SHOULD_FAIL"])
        with self.assertRaises(SystemExit):
            _, config = cli.parse()

    def _test_numeric_option(self, option_struct):
        long_option = option_struct.long_flag
        short_option = option_struct.short_flag
        expected_value_string = option_struct.expected_value
        expected_value = self._convert_string_to_numeric(option_struct.expected_value)
        expected_default_value = (
            None
            if option_struct.expected_default_value is None
            else self._convert_string_to_numeric(option_struct.expected_default_value)
        )

        long_option_with_underscores = self._convert_flag_to_use_underscores(
            long_option
        )

        self._test_long_flag(
            long_option,
            option_struct.cli_subcommand,
            expected_value_string,
            long_option_with_underscores,
            expected_value,
        )

        self._test_short_flag(
            short_option,
            option_struct.cli_subcommand,
            expected_value_string,
            long_option_with_underscores,
            expected_value,
        )

        if expected_default_value is not None:
            self._test_expected_default_value(
                option_struct.cli_subcommand,
                long_option_with_underscores,
                expected_default_value,
            )

    def _test_string_option(self, option_struct):
        long_option = option_struct.long_flag
        short_option = option_struct.short_flag
        expected_value = option_struct.expected_value
        expected_default_value = option_struct.expected_default_value
        expected_failing_value = option_struct.expected_failing_value
        extra_commands = option_struct.extra_commands

        # This covers strings that have choices
        # Recursively call this method with choices
        if type(expected_value) is list:
            for value in expected_value:
                new_struct = copy.deepcopy(option_struct)
                new_struct.expected_value = value
                self._test_string_option(new_struct)
        else:
            long_option_with_underscores = self._convert_flag_to_use_underscores(
                long_option
            )

            self._test_long_flag(
                long_option,
                option_struct.cli_subcommand,
                expected_value,
                long_option_with_underscores,
                expected_value,
                extra_commands,
            )

            self._test_short_flag(
                short_option,
                option_struct.cli_subcommand,
                expected_value,
                long_option_with_underscores,
                expected_value,
            )

            if expected_default_value is not None:
                self._test_expected_default_value(
                    option_struct.cli_subcommand,
                    long_option_with_underscores,
                    expected_default_value,
                )

            # Verify that a incorrect value causes a failure
            if expected_failing_value is not None:
                cli = option_struct.cli_subcommand()
                cli.args.extend([long_option, expected_failing_value])
                with self.assertRaises(SystemExit):
                    _, config = cli.parse()

    def _test_list_option(self, option_struct):
        long_option = option_struct.long_flag
        short_option = option_struct.short_flag
        expected_value = option_struct.expected_value
        expected_default_value = option_struct.expected_default_value
        extra_commands = option_struct.extra_commands

        # Convert expected and default values to proper types for comparison
        if option_struct.type == "intlist":
            expected_value_converted = self._convert_string_to_int_list(expected_value)
            if expected_default_value is not None:
                expected_default_value_converted = self._convert_string_to_int_list(
                    expected_default_value
                )
            else:
                expected_default_value_converted = None
        else:
            expected_value_converted = self._convert_string_to_string_list(
                expected_value
            )
            if expected_default_value is not None:
                expected_default_value_converted = self._convert_string_to_string_list(
                    expected_default_value
                )
            else:
                expected_default_value_converted = None

        long_option_with_underscores = self._convert_flag_to_use_underscores(
            long_option
        )

        self._test_long_flag(
            long_option,
            option_struct.cli_subcommand,
            expected_value,
            long_option_with_underscores,
            expected_value_converted,
            extra_commands,
        )

        self._test_short_flag(
            short_option,
            option_struct.cli_subcommand,
            expected_value,
            long_option_with_underscores,
            expected_value_converted,
        )

        if expected_default_value is not None:
            self._test_expected_default_value(
                option_struct.cli_subcommand,
                long_option_with_underscores,
                expected_default_value_converted,
            )

    # Helper methods

    def _test_long_flag(
        self,
        long_option,
        cli_subcommand,
        expected_value_string,
        long_option_with_underscores,
        expected_value,
        extra_commands=None,
    ):
        cli = cli_subcommand()
        cli.args.extend([long_option, expected_value_string])
        if extra_commands is not None:
            cli.args.extend(extra_commands)
        _, config = cli.parse()
        option_value = config.get_config().get(long_option_with_underscores).value()
        self.assertEqual(option_value, expected_value)

    def _test_short_flag(
        self,
        short_option,
        cli_subcommand,
        expected_value_string,
        long_option_with_underscores,
        expected_value,
    ):
        if short_option is not None:
            cli = cli_subcommand()
            cli.args.extend([short_option, expected_value_string])
            _, config = cli.parse()
            option_value = config.get_config().get(long_option_with_underscores).value()
            self.assertEqual(option_value, expected_value)

    def _test_expected_default_value(
        self, cli_subcommand, long_option_with_underscores, expected_default_value
    ):
        cli = cli_subcommand()
        _, config = cli.parse()
        option_value = (
            config.get_config().get(long_option_with_underscores).default_value()
        )
        self.assertEqual(option_value, expected_default_value)

    def _convert_flag_to_use_underscores(self, option):
        return option.lstrip("-").replace("-", "_")

    def _convert_string_to_numeric(self, number):
        return float(number) if "." in number else int(number)

    def _convert_string_to_int_list(self, list_values):
        ret_val = [int(x) for x in list_values.split(",")]
        if len(ret_val) == 1:
            return ret_val[0]
        return ret_val

    def _convert_string_to_string_list(self, list_values):
        ret_val = [x for x in list_values.split(",")]
        if len(ret_val) == 1:
            return ret_val[0]
        return ret_val


if __name__ == "__main__":
    unittest.main()
