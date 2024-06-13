# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
from pathlib import Path

import genai_perf.logging as logging
import pytest
from genai_perf import __version__, parser
from genai_perf.llm_inputs.llm_inputs import (
    ImageFormat,
    ModelSelectionStrategy,
    OutputFormat,
    PromptSource,
)
from genai_perf.llm_inputs.synthetic_image_generator import ImageFormat
from genai_perf.parser import PathType


class TestCLIArguments:
    # ================================================
    # PROFILE COMMAND
    # ================================================
    expected_help_output = (
        "CLI to profile LLMs and Generative AI models with Perf Analyzer"
    )
    expected_version_output = f"genai-perf {__version__}"

    @pytest.mark.parametrize(
        "args, expected_output",
        [
            (["-h"], expected_help_output),
            (["--help"], expected_help_output),
            (["--version"], expected_version_output),
        ],
    )
    def test_help_version_arguments_output_and_exit(
        self, monkeypatch, args, expected_output, capsys
    ):
        monkeypatch.setattr("sys.argv", ["genai-perf"] + args)

        with pytest.raises(SystemExit) as excinfo:
            _ = parser.parse_args()

        # Check that the exit was successful
        assert excinfo.value.code == 0

        # Capture that the correct message was displayed
        captured = capsys.readouterr()
        assert expected_output in captured.out

    @pytest.mark.parametrize(
        "arg, expected_attributes",
        [
            (
                ["--artifact-dir", "test_artifact_dir"],
                {"artifact_dir": Path("test_artifact_dir")},
            ),
            (
                [
                    "--batch-size",
                    "5",
                    "--endpoint-type",
                    "embeddings",
                    "--service-kind",
                    "openai",
                ],
                {"batch_size": 5},
            ),
            (
                [
                    "-b",
                    "5",
                    "--endpoint-type",
                    "embeddings",
                    "--service-kind",
                    "openai",
                ],
                {"batch_size": 5},
            ),
            (["--concurrency", "3"], {"concurrency": 3}),
            (
                ["--endpoint-type", "completions", "--service-kind", "openai"],
                {"endpoint": "v1/completions"},
            ),
            (
                ["--endpoint-type", "chat", "--service-kind", "openai"],
                {"endpoint": "v1/chat/completions"},
            ),
            (
                ["--endpoint-type", "rankings", "--service-kind", "openai"],
                {"endpoint": "v1/ranking"},
            ),
            (
                [
                    "--endpoint-type",
                    "chat",
                    "--service-kind",
                    "openai",
                    "--endpoint",
                    "custom/address",
                ],
                {"endpoint": "custom/address"},
            ),
            (
                [
                    "--endpoint-type",
                    "chat",
                    "--service-kind",
                    "openai",
                    "--endpoint",
                    "   /custom/address",
                ],
                {"endpoint": "custom/address"},
            ),
            (
                [
                    "--endpoint-type",
                    "completions",
                    "--service-kind",
                    "openai",
                    "--endpoint",
                    "custom/address",
                ],
                {"endpoint": "custom/address"},
            ),
            (
                ["--extra-inputs", "test_key:test_value"],
                {"extra_inputs": ["test_key:test_value"]},
            ),
            (
                [
                    "--extra-inputs",
                    "test_key:5",
                    "--extra-inputs",
                    "another_test_key:6",
                ],
                {"extra_inputs": ["test_key:5", "another_test_key:6"]},
            ),
            (
                [
                    "--extra-inputs",
                    '{"name": "Wolverine","hobbies": ["hacking", "slashing"],"address": {"street": "1407 Graymalkin Lane, Salem Center","city": "NY"}}',
                ],
                {
                    "extra_inputs": [
                        '{"name": "Wolverine","hobbies": ["hacking", "slashing"],"address": {"street": "1407 Graymalkin Lane, Salem Center","city": "NY"}}'
                    ]
                },
            ),
            (["--input-dataset", "openorca"], {"input_dataset": "openorca"}),
            (["--measurement-interval", "100"], {"measurement_interval": 100}),
            (
                ["--model-selection-strategy", "random"],
                {"model_selection_strategy": ModelSelectionStrategy.RANDOM},
            ),
            (["--num-prompts", "101"], {"num_prompts": 101}),
            (
                ["--output-tokens-mean", "6"],
                {"output_tokens_mean": 6},
            ),
            (
                ["--output-tokens-mean", "6", "--output-tokens-stddev", "7"],
                {"output_tokens_stddev": 7},
            ),
            (
                ["--output-tokens-mean", "6", "--output-tokens-mean-deterministic"],
                {"output_tokens_mean_deterministic": True},
            ),
            (["-p", "100"], {"measurement_interval": 100}),
            (
                ["--profile-export-file", "test.json"],
                {
                    "profile_export_file": Path(
                        "artifacts/test_model-triton-tensorrtllm-concurrency1/test.json"
                    )
                },
            ),
            (["--random-seed", "8"], {"random_seed": 8}),
            (["--request-rate", "9.0"], {"request_rate": 9.0}),
            (["-s", "99.5"], {"stability_percentage": 99.5}),
            (["--service-kind", "triton"], {"service_kind": "triton"}),
            (
                ["--service-kind", "tensorrtllm_engine"],
                {"service_kind": "tensorrtllm_engine"},
            ),
            (
                ["--service-kind", "openai", "--endpoint-type", "chat"],
                {"service_kind": "openai", "endpoint": "v1/chat/completions"},
            ),
            (["--stability-percentage", "99.5"], {"stability_percentage": 99.5}),
            (["--streaming"], {"streaming": True}),
            (
                ["--synthetic-input-tokens-mean", "6"],
                {"synthetic_input_tokens_mean": 6},
            ),
            (
                ["--synthetic-input-tokens-stddev", "7"],
                {"synthetic_input_tokens_stddev": 7},
            ),
            (
                ["--image-width-mean", "123"],
                {"image_width_mean": 123},
            ),
            (
                ["--image-width-stddev", "123"],
                {"image_width_stddev": 123},
            ),
            (
                ["--image-height-mean", "456"],
                {"image_height_mean": 456},
            ),
            (
                ["--image-height-stddev", "456"],
                {"image_height_stddev": 456},
            ),
            (["--image-format", "png"], {"image_format": ImageFormat.PNG}),
            (["-v"], {"verbose": True}),
            (["--verbose"], {"verbose": True}),
            (["-u", "test_url"], {"u": "test_url"}),
            (["--url", "test_url"], {"u": "test_url"}),
        ],
    )
    def test_non_file_flags_parsed(self, monkeypatch, arg, expected_attributes, capsys):
        logging.init_logging()
        combined_args = ["genai-perf", "profile", "--model", "test_model"] + arg
        monkeypatch.setattr("sys.argv", combined_args)
        args, _ = parser.parse_args()

        # Check that the attributes are set correctly
        for key, value in expected_attributes.items():
            assert getattr(args, key) == value

        # Check that nothing was printed as a byproduct of parsing the arguments
        captured = capsys.readouterr()
        assert captured.out == ""

    @pytest.mark.parametrize(
        "models, expected_model_list, formatted_name",
        [
            (
                ["--model", "test_model_A"],
                {"model": ["test_model_A"]},
                {"formatted_model_name": "test_model_A"},
            ),
            (
                ["--model", "test_model_A", "test_model_B"],
                {"model": ["test_model_A", "test_model_B"]},
                {"formatted_model_name": "test_model_A_multi"},
            ),
            (
                ["--model", "test_model_A", "test_model_B", "test_model_C"],
                {"model": ["test_model_A", "test_model_B", "test_model_C"]},
                {"formatted_model_name": "test_model_A_multi"},
            ),
            (
                ["--model", "test_model_A:math", "test_model_B:embedding"],
                {"model": ["test_model_A:math", "test_model_B:embedding"]},
                {"formatted_model_name": "test_model_A:math_multi"},
            ),
        ],
    )
    def test_multiple_model_args(
        self, monkeypatch, models, expected_model_list, formatted_name, capsys
    ):
        logging.init_logging()
        combined_args = ["genai-perf", "profile"] + models
        monkeypatch.setattr("sys.argv", combined_args)
        args, _ = parser.parse_args()

        # Check that models are handled correctly
        for key, value in expected_model_list.items():
            assert getattr(args, key) == value

        # Check that the formatted_model_name is correctly generated
        for key, value in formatted_name.items():
            assert getattr(args, key) == value

        # Check that nothing was printed as a byproduct of parsing the arguments
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_file_flags_parsed(self, monkeypatch, mocker):
        _ = mocker.patch("os.path.isfile", return_value=True)
        combined_args = [
            "genai-perf",
            "profile",
            "--model",
            "test_model",
            "--input-file",
            "fakefile.txt",
        ]
        monkeypatch.setattr("sys.argv", combined_args)
        args, _ = parser.parse_args()
        filepath, pathtype = args.input_file
        assert filepath == Path(
            "fakefile.txt"
        ), "The file argument should be the path to the file"
        assert pathtype == PathType.FILE

    @pytest.mark.parametrize(
        "arg, expected_path",
        [
            (
                ["--service-kind", "openai", "--endpoint-type", "chat"],
                "artifacts/test_model-openai-chat-concurrency1",
            ),
            (
                ["--service-kind", "openai", "--endpoint-type", "completions"],
                "artifacts/test_model-openai-completions-concurrency1",
            ),
            (
                ["--service-kind", "openai", "--endpoint-type", "rankings"],
                "artifacts/test_model-openai-rankings-concurrency1",
            ),
            (
                ["--service-kind", "triton", "--backend", "tensorrtllm"],
                "artifacts/test_model-triton-tensorrtllm-concurrency1",
            ),
            (
                ["--service-kind", "triton", "--backend", "vllm"],
                "artifacts/test_model-triton-vllm-concurrency1",
            ),
            (
                [
                    "--service-kind",
                    "triton",
                    "--backend",
                    "vllm",
                    "--concurrency",
                    "32",
                ],
                "artifacts/test_model-triton-vllm-concurrency32",
            ),
        ],
    )
    def test_default_profile_export_filepath(
        self, monkeypatch, arg, expected_path, capsys
    ):
        logging.init_logging()
        combined_args = ["genai-perf", "profile", "--model", "test_model"] + arg
        monkeypatch.setattr("sys.argv", combined_args)
        args, _ = parser.parse_args()

        assert args.artifact_dir == Path(expected_path)
        captured = capsys.readouterr()
        assert captured.out == ""

    @pytest.mark.parametrize(
        "arg, expected_path, expected_output",
        [
            (
                ["--model", "strange/test_model"],
                "artifacts/strange_test_model-triton-tensorrtllm-concurrency1",
                (
                    "Model name 'strange/test_model' cannot be used to create "
                    "artifact directory. Instead, 'strange_test_model' will be used"
                ),
            ),
            (
                [
                    "--model",
                    "hello/world/test_model",
                    "--service-kind",
                    "openai",
                    "--endpoint-type",
                    "chat",
                ],
                "artifacts/hello_world_test_model-openai-chat-concurrency1",
                (
                    "Model name 'hello/world/test_model' cannot be used to create "
                    "artifact directory. Instead, 'hello_world_test_model' will be used"
                ),
            ),
        ],
    )
    def test_model_name_artifact_path(
        self, monkeypatch, arg, expected_path, expected_output, capsys
    ):
        logging.init_logging()
        combined_args = ["genai-perf", "profile"] + arg
        monkeypatch.setattr("sys.argv", combined_args)
        args, _ = parser.parse_args()

        assert args.artifact_dir == Path(expected_path)
        captured = capsys.readouterr()
        assert expected_output in captured.out

    def test_default_load_level(self, monkeypatch, capsys):
        logging.init_logging()
        monkeypatch.setattr(
            "sys.argv", ["genai-perf", "profile", "--model", "test_model"]
        )
        args, _ = parser.parse_args()
        assert args.concurrency == 1
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_load_level_mutually_exclusive(self, monkeypatch, capsys):
        monkeypatch.setattr(
            "sys.argv",
            ["genai-perf", "profile", "--concurrency", "3", "--request-rate", "9.0"],
        )
        expected_output = (
            "argument --request-rate: not allowed with argument --concurrency"
        )

        with pytest.raises(SystemExit) as excinfo:
            parser.parse_args()

        assert excinfo.value.code != 0
        captured = capsys.readouterr()
        assert expected_output in captured.err

    def test_model_not_provided(self, monkeypatch, capsys):
        monkeypatch.setattr("sys.argv", ["genai-perf", "profile"])
        expected_output = "The -m/--model option is required and cannot be empty."

        with pytest.raises(SystemExit) as excinfo:
            parser.parse_args()

        assert excinfo.value.code != 0
        captured = capsys.readouterr()
        assert expected_output in captured.err

    def test_pass_through_args(self, monkeypatch):
        args = ["genai-perf", "profile", "-m", "test_model"]
        other_args = ["--", "With", "great", "power"]
        monkeypatch.setattr("sys.argv", args + other_args)
        _, pass_through_args = parser.parse_args()

        assert pass_through_args == other_args[1:]

    def test_unrecognized_arg(self, monkeypatch, capsys):
        monkeypatch.setattr(
            "sys.argv",
            [
                "genai-perf",
                "profile",
                "-m",
                "nonexistent_model",
                "--wrong-arg",
            ],
        )
        expected_output = "unrecognized arguments: --wrong-arg"

        with pytest.raises(SystemExit) as excinfo:
            parser.parse_args()

        assert excinfo.value.code != 0
        captured = capsys.readouterr()
        assert expected_output in captured.err

    @pytest.mark.parametrize(
        "args, expected_output",
        [
            (
                [
                    "genai-perf",
                    "profile",
                    "-m",
                    "test_model",
                    "--service-kind",
                    "openai",
                ],
                "The --endpoint-type option is required when using the 'openai' service-kind.",
            ),
            (
                [
                    "genai-perf",
                    "profile",
                    "-m",
                    "test_model",
                    "--service-kind",
                    "openai",
                    "--endpoint",
                    "custom/address",
                ],
                "The --endpoint-type option is required when using the 'openai' service-kind.",
            ),
            (
                [
                    "genai-perf",
                    "profile",
                    "-m",
                    "test_model",
                    "--output-tokens-stddev",
                    "5",
                ],
                "The --output-tokens-mean option is required when using --output-tokens-stddev.",
            ),
            (
                [
                    "genai-perf",
                    "profile",
                    "-m",
                    "test_model",
                    "--output-tokens-mean-deterministic",
                ],
                "The --output-tokens-mean option is required when using --output-tokens-mean-deterministic.",
            ),
            (
                [
                    "genai-perf",
                    "profile",
                    "-m",
                    "test_model",
                    "--output-tokens-mean-deterministic",
                ],
                "The --output-tokens-mean option is required when using --output-tokens-mean-deterministic.",
            ),
            (
                [
                    "genai-perf",
                    "profile",
                    "-m",
                    "test_model",
                    "--service-kind",
                    "openai",
                    "--endpoint-type",
                    "chat",
                    "--output-tokens-mean",
                    "100",
                    "--output-tokens-mean-deterministic",
                ],
                "The --output-tokens-mean-deterministic option is only supported with the Triton service-kind",
            ),
            (
                [
                    "genai-perf",
                    "profile",
                    "-m",
                    "test_model",
                    "--batch-size",
                    "10",
                ],
                "The --batch-size option is currently only supported with the embeddings and rankings endpoint types",
            ),
            (
                [
                    "genai-perf",
                    "profile",
                    "-m",
                    "test_model",
                    "--service-kind",
                    "openai",
                    "--endpoint-type",
                    "embeddings",
                    "--streaming",
                ],
                "The --streaming option is not supported with the embeddings endpoint type",
            ),
            (
                [
                    "genai-perf",
                    "profile",
                    "-m",
                    "test_model",
                    "--service-kind",
                    "openai",
                    "--endpoint-type",
                    "rankings",
                    "--streaming",
                ],
                "The --streaming option is not supported with the rankings endpoint type",
            ),
            (
                [
                    "genai-perf",
                    "profile",
                    "-m",
                    "test_model",
                    "--service-kind",
                    "openai",
                    "--endpoint-type",
                    "embeddings",
                    "--generate-plots",
                ],
                "The --generate-plots option is not currently supported with the embeddings endpoint type",
            ),
            (
                [
                    "genai-perf",
                    "profile",
                    "-m",
                    "test_model",
                    "--service-kind",
                    "openai",
                    "--endpoint-type",
                    "rankings",
                    "--generate-plots",
                ],
                "The --generate-plots option is not currently supported with the rankings endpoint type",
            ),
        ],
    )
    def test_conditional_errors(self, args, expected_output, monkeypatch, capsys):
        monkeypatch.setattr("sys.argv", args)

        with pytest.raises(SystemExit) as excinfo:
            parser.parse_args()

        assert excinfo.value.code != 0
        captured = capsys.readouterr()
        assert expected_output in captured.err

    @pytest.mark.parametrize(
        "args, expected_format",
        [
            (
                ["--service-kind", "openai", "--endpoint-type", "chat"],
                OutputFormat.OPENAI_CHAT_COMPLETIONS,
            ),
            (
                ["--service-kind", "openai", "--endpoint-type", "completions"],
                OutputFormat.OPENAI_COMPLETIONS,
            ),
            (
                [
                    "--service-kind",
                    "openai",
                    "--endpoint-type",
                    "completions",
                    "--endpoint",
                    "custom/address",
                ],
                OutputFormat.OPENAI_COMPLETIONS,
            ),
            (
                ["--service-kind", "openai", "--endpoint-type", "rankings"],
                OutputFormat.RANKINGS,
            ),
            (
                ["--service-kind", "triton", "--backend", "tensorrtllm"],
                OutputFormat.TENSORRTLLM,
            ),
            (["--service-kind", "triton", "--backend", "vllm"], OutputFormat.VLLM),
            (["--service-kind", "tensorrtllm_engine"], OutputFormat.TENSORRTLLM_ENGINE),
        ],
    )
    def test_inferred_output_format(self, monkeypatch, args, expected_format):
        monkeypatch.setattr(
            "sys.argv", ["genai-perf", "profile", "-m", "test_model"] + args
        )

        parsed_args, _ = parser.parse_args()
        assert parsed_args.output_format == expected_format

    @pytest.mark.parametrize(
        "args, expected_error",
        [
            (
                ["--extra-inputs", "hi:"],
                "Input name or value is empty in --extra-inputs: hi:\nExpected input format: 'input_name:value'",
            ),
            (
                ["--extra-inputs", ":a"],
                "Input name or value is empty in --extra-inputs: :a\nExpected input format: 'input_name:value'",
            ),
            (
                ["--extra-inputs", ":a:"],
                "Invalid input format for --extra-inputs: :a:\nExpected input format: 'input_name:value'",
            ),
            (
                ["--extra-inputs", "unknown"],
                "Invalid input format for --extra-inputs: unknown\nExpected input format: 'input_name:value'",
            ),
            (
                ["--extra-inputs", "test_key:5", "--extra-inputs", "test_key:6"],
                "Input name already exists in request_inputs dictionary: test_key",
            ),
        ],
    )
    def test_repeated_extra_arg_warning(self, monkeypatch, args, expected_error):
        combined_args = ["genai-perf", "profile", "-m", "test_model"] + args
        monkeypatch.setattr("sys.argv", combined_args)

        parsed_args, _ = parser.parse_args()

        with pytest.raises(ValueError) as exc_info:
            _ = parser.get_extra_inputs_as_dict(parsed_args)

        assert str(exc_info.value) == expected_error

    @pytest.mark.parametrize(
        "args, expected_prompt_source",
        [
            ([], PromptSource.SYNTHETIC),
            (["--input-dataset", "openorca"], PromptSource.DATASET),
            (["--input-file", "prompt.txt"], PromptSource.FILE),
            (
                ["--input-file", "prompt.txt", "--synthetic-input-tokens-mean", "10"],
                PromptSource.FILE,
            ),
        ],
    )
    def test_inferred_prompt_source(
        self, monkeypatch, mocker, args, expected_prompt_source
    ):
        _ = mocker.patch("builtins.open", mocker.mock_open(read_data="data"))
        _ = mocker.patch("os.path.isfile", return_value=True)
        _ = mocker.patch("os.path.isdir", return_value=True)
        combined_args = ["genai-perf", "profile", "--model", "test_model"] + args
        monkeypatch.setattr("sys.argv", combined_args)
        args, _ = parser.parse_args()

        assert args.prompt_source == expected_prompt_source

    def test_prompt_source_assertions(self, monkeypatch, mocker, capsys):
        _ = mocker.patch("builtins.open", mocker.mock_open(read_data="data"))
        _ = mocker.patch("os.path.isfile", return_value=True)
        _ = mocker.patch("os.path.isdir", return_value=True)
        args = [
            "genai-perf",
            "profile",
            "--model",
            "test_model",
            "--input-dataset",
            "openorca",
            "--input-file",
            "prompt.txt",
        ]
        monkeypatch.setattr("sys.argv", args)

        expected_output = (
            "argument --input-file: not allowed with argument --input-dataset"
        )

        with pytest.raises(SystemExit) as excinfo:
            parser.parse_args()

        assert excinfo.value.code != 0
        captured = capsys.readouterr()
        assert expected_output in captured.err

    @pytest.mark.parametrize(
        "args",
        [
            # negative numbers
            ["--image-width-mean", "-123"],
            ["--image-width-stddev", "-34"],
            ["--image-height-mean", "-123"],
            ["--image-height-stddev", "-34"],
            # zeros
            ["--image-width-mean", "0"],
            ["--image-height-mean", "0"],
        ],
    )
    def test_positive_image_input_args(self, monkeypatch, args):
        combined_args = ["genai-perf", "profile", "-m", "test_model"] + args
        monkeypatch.setattr("sys.argv", combined_args)

        with pytest.raises(SystemExit) as excinfo:
            parser.parse_args()

    # ================================================
    # COMPARE SUBCOMMAND
    # ================================================
    expected_compare_help_output = (
        "Subcommand to generate plots that compare multiple profile runs."
    )

    @pytest.mark.parametrize(
        "args, expected_output",
        [
            (["-h"], expected_compare_help_output),
            (["--help"], expected_compare_help_output),
        ],
    )
    def test_compare_help_arguments_output_and_exit(
        self, monkeypatch, args, expected_output, capsys
    ):
        logging.init_logging()
        monkeypatch.setattr("sys.argv", ["genai-perf", "compare"] + args)

        with pytest.raises(SystemExit) as excinfo:
            _ = parser.parse_args()

        # Check that the exit was successful
        assert excinfo.value.code == 0

        # Capture that the correct message was displayed
        captured = capsys.readouterr()
        assert expected_output in captured.out

    def test_compare_mutually_exclusive(self, monkeypatch, capsys):
        args = ["genai-perf", "compare", "--config", "hello", "--files", "a", "b", "c"]
        monkeypatch.setattr("sys.argv", args)
        expected_output = "argument -f/--files: not allowed with argument --config"

        with pytest.raises(SystemExit) as excinfo:
            parser.parse_args()

        assert excinfo.value.code != 0
        captured = capsys.readouterr()
        assert expected_output in captured.err

    def test_compare_not_provided(self, monkeypatch, capsys):
        args = ["genai-perf", "compare"]
        monkeypatch.setattr("sys.argv", args)
        expected_output = "Either the --config or --files option must be specified."

        with pytest.raises(SystemExit) as excinfo:
            parser.parse_args()

        assert excinfo.value.code != 0
        captured = capsys.readouterr()
        assert expected_output in captured.err

    @pytest.mark.parametrize(
        "extra_inputs_list, expected_dict",
        [
            (["test_key:test_value"], {"test_key": "test_value"}),
            (
                ["test_key:1", "another_test_key:2"],
                {"test_key": 1, "another_test_key": 2},
            ),
            (
                [
                    '{"name": "Wolverine","hobbies": ["hacking", "slashing"],"address": {"street": "1407 Graymalkin Lane, Salem Center","city": "NY"}}'
                ],
                {
                    "name": "Wolverine",
                    "hobbies": ["hacking", "slashing"],
                    "address": {
                        "street": "1407 Graymalkin Lane, Salem Center",
                        "city": "NY",
                    },
                },
            ),
        ],
    )
    def test_get_extra_inputs_as_dict(self, extra_inputs_list, expected_dict):
        namespace = argparse.Namespace()
        namespace.extra_inputs = extra_inputs_list
        actual_dict = parser.get_extra_inputs_as_dict(namespace)
        assert actual_dict == expected_dict
