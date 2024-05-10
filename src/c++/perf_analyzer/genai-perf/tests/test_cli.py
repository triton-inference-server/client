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

from pathlib import Path

import pytest
from genai_perf import __version__, parser
from genai_perf.llm_inputs.llm_inputs import OutputFormat, PromptSource


class TestCLIArguments:
    # ================================================
    # GENAI-PERF COMMAND
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
            (["-m", "abc", "--help"], expected_help_output),
            (["-m", "abc", "-h"], expected_help_output),
            (["--version"], expected_version_output),
            (["-m", "abc", "--version"], expected_version_output),
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
            (["--input-dataset", "openorca"], {"input_dataset": "openorca"}),
            (
                ["--synthetic-input-tokens-mean", "6"],
                {"synthetic_input_tokens_mean": 6},
            ),
            (
                ["--synthetic-input-tokens-stddev", "7"],
                {"synthetic_input_tokens_stddev": 7},
            ),
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
            (["--measurement-interval", "100"], {"measurement_interval": 100}),
            (["-p", "100"], {"measurement_interval": 100}),
            (["--num-prompts", "101"], {"num_prompts": 101}),
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
            (["--service-kind", "triton"], {"service_kind": "triton"}),
            (
                ["--service-kind", "openai", "--endpoint-type", "chat"],
                {"service_kind": "openai", "endpoint": "v1/chat/completions"},
            ),
            (["--stability-percentage", "99.5"], {"stability_percentage": 99.5}),
            (["-s", "99.5"], {"stability_percentage": 99.5}),
            (["--streaming"], {"streaming": True}),
            (["--verbose"], {"verbose": True}),
            (["-v"], {"verbose": True}),
            (["--url", "test_url"], {"u": "test_url"}),
            (["-u", "test_url"], {"u": "test_url"}),
            (
                ["--artifact-dir", "test_artifact_dir"],
                {"artifact_dir": Path("test_artifact_dir")},
            ),
        ],
    )
    def test_non_file_flags_parsed(self, monkeypatch, arg, expected_attributes, capsys):
        combined_args = ["genai-perf", "--model", "test_model"] + arg
        monkeypatch.setattr("sys.argv", combined_args)
        args, _ = parser.parse_args()

        # Check that the attributes are set correctly
        for key, value in expected_attributes.items():
            assert getattr(args, key) == value

        # Check that nothing was printed as a byproduct of parsing the arguments
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_file_flags_parsed(self, monkeypatch, mocker):
        mocked_open = mocker.patch("builtins.open", mocker.mock_open(read_data="data"))
        combined_args = [
            "genai-perf",
            "--model",
            "test_model",
            "--input-file",
            "fakefile.txt",
        ]
        monkeypatch.setattr("sys.argv", combined_args)
        args, _ = parser.parse_args()
        assert (
            args.input_file == mocked_open.return_value
        ), "The file argument should be the mock object"

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
        combined_args = ["genai-perf", "--model", "test_model"] + arg
        monkeypatch.setattr("sys.argv", combined_args)
        args, extra_args = parser.parse_args()

        assert args.artifact_dir == Path(expected_path)
        captured = capsys.readouterr()
        assert captured.out == ""

    @pytest.mark.parametrize(
        "arg, expected_path",
        [
            (
                ["--model", "strange/test_model"],
                "artifacts/strange_test_model-triton-tensorrtllm-concurrency1",
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
            ),
        ],
    )
    def test_model_name_artifact_path(self, monkeypatch, arg, expected_path, capsys):
        combined_args = ["genai-perf"] + arg
        monkeypatch.setattr("sys.argv", combined_args)
        args, extra_args = parser.parse_args()

        assert args.artifact_dir == Path(expected_path)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_default_load_level(self, monkeypatch, capsys):
        monkeypatch.setattr("sys.argv", ["genai-perf", "--model", "test_model"])
        args, extra_args = parser.parse_args()
        assert args.concurrency == 1
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_load_level_mutually_exclusive(self, monkeypatch, capsys):
        monkeypatch.setattr(
            "sys.argv", ["genai-perf", "--concurrency", "3", "--request-rate", "9.0"]
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
        monkeypatch.setattr("sys.argv", ["genai-perf"])
        expected_output = "The -m/--model option is required and cannot be empty."

        with pytest.raises(SystemExit) as excinfo:
            parser.parse_args()

        assert excinfo.value.code != 0
        captured = capsys.readouterr()
        assert expected_output in captured.err

    def test_pass_through_args(self, monkeypatch):
        args = ["genai-perf", "-m", "test_model"]
        other_args = ["--", "With", "great", "power"]
        monkeypatch.setattr("sys.argv", args + other_args)
        _, pass_through_args = parser.parse_args()

        assert pass_through_args == other_args[1:]

    def test_unrecognized_arg(self, monkeypatch, capsys):
        monkeypatch.setattr(
            "sys.argv",
            [
                "genai-perf",
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
                ["genai-perf", "-m", "test_model", "--service-kind", "openai"],
                "The --endpoint-type option is required when using the 'openai' service-kind.",
            ),
            (
                [
                    "genai-perf",
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
                ["genai-perf", "-m", "test_model", "--output-tokens-stddev", "5"],
                "The --output-tokens-mean option is required when using --output-tokens-stddev.",
            ),
            (
                [
                    "genai-perf",
                    "-m",
                    "test_model",
                    "--output-tokens-mean-deterministic",
                ],
                "The --output-tokens-mean option is required when using --output-tokens-mean-deterministic.",
            ),
            (
                [
                    "genai-perf",
                    "-m",
                    "test_model",
                    "--output-tokens-mean-deterministic",
                ],
                "The --output-tokens-mean option is required when using --output-tokens-mean-deterministic.",
            ),
            (
                [
                    "genai-perf",
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
                ["--service-kind", "triton", "--backend", "tensorrtllm"],
                OutputFormat.TENSORRTLLM,
            ),
            (["--service-kind", "triton", "--backend", "vllm"], OutputFormat.VLLM),
        ],
    )
    def test_inferred_output_format(self, monkeypatch, args, expected_format):
        monkeypatch.setattr("sys.argv", ["genai-perf", "-m", "test_model"] + args)

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
        combined_args = ["genai-perf", "-m", "test_model"] + args
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
        combined_args = ["genai-perf", "--model", "test_model"] + args
        monkeypatch.setattr("sys.argv", combined_args)
        args, _ = parser.parse_args()

        assert args.prompt_source == expected_prompt_source

    def test_prompt_source_assertions(self, monkeypatch, mocker, capsys):
        _ = mocker.patch("builtins.open", mocker.mock_open(read_data="data"))
        args = [
            "genai-perf",
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
        "args, expected_model",
        [
            (["--files", "profile1.json", "profile2.json", "profile3.json"], None),
            (["--config", "config.yaml"], None),
        ],
    )
    def test_compare_model_arg(self, monkeypatch, args, expected_model):
        combined_args = ["genai-perf", "compare"] + args
        monkeypatch.setattr("sys.argv", combined_args)
        args, _ = parser.parse_args()

        assert args.model == expected_model
