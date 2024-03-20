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

import genai_perf.utils as utils
import pytest
from genai_perf import __version__, parser
from genai_perf.exceptions import GenAIPerfException
from genai_perf.llm_inputs.llm_inputs import InputType, OutputFormat
from genai_perf.main import run


class TestCLIArguments:
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
            (["--concurrency", "3"], {"concurrency_range": "3"}),
            (["--endpoint", "v1/completions"], {"endpoint": "v1/completions"}),
            (
                ["--endpoint", "v1/chat/completions"],
                {"endpoint": "v1/chat/completions"},
            ),
            (["--expected-output-tokens", "5"], {"expected_output_tokens": 5}),
            (["--input-dataset", "openorca"], {"input_dataset": "openorca"}),
            (["--input-tokens-mean", "6"], {"input_tokens_mean": 6}),
            (["--input-tokens-stddev", "7"], {"input_tokens_stddev": 7}),
            (
                ["--input-type", "synthetic"],
                {"input_type": utils.get_enum_entry("synthetic", InputType)},
            ),
            (
                ["--input-type", "url"],
                {"input_type": utils.get_enum_entry("url", InputType)},
            ),
            (["--measurement-interval", "100"], {"measurement_interval": 100}),
            (["-p", "100"], {"measurement_interval": 100}),
            (["--num-of-output-prompts", "101"], {"num_of_output_prompts": 101}),
            (
                ["--profile-export-file", "text.txt"],
                {"profile_export_file": Path("text.txt")},
            ),
            (["--random-seed", "8"], {"random_seed": 8}),
            (["--request-rate", "9.0"], {"request_rate_range": "9.0"}),
            (["--service-kind", "triton"], {"service_kind": "triton"}),
            (
                ["--service-kind", "openai", "--endpoint", "v1/chat/completions"],
                {"service_kind": "openai", "endpoint": "v1/chat/completions"},
            ),
            (["--stability-percentage", "99.5"], {"stability_percentage": 99.5}),
            (["-s", "99.5"], {"stability_percentage": 99.5}),
            (["--streaming"], {"streaming": True}),
            (["--verbose"], {"verbose": True}),
            (["-v"], {"verbose": True}),
            (["--url", "test_url"], {"u": "test_url"}),
            (["-u", "test_url"], {"u": "test_url"}),
        ],
    )
    def test_all_flags_parsed(self, monkeypatch, arg, expected_attributes, capsys):
        combined_args = ["genai-perf", "--model", "test_model"] + arg
        monkeypatch.setattr("sys.argv", combined_args)
        args, _ = parser.parse_args()

        # Check that the attributes are set correctly
        for key, value in expected_attributes.items():
            assert getattr(args, key) == value

        # Check that nothing was printed as a byproduct of parsing the arguments
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_default_load_level(self, monkeypatch, capsys):
        monkeypatch.setattr("sys.argv", ["genai-perf", "--model", "test_model"])
        args, extra_args = parser.parse_args()
        assert getattr(args, "concurrency_range") == "1"
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
        expected_output = "the following arguments are required: -m/--model"

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

    def test_service_openai_no_endpoint(self, monkeypatch, capsys):
        args = ["genai-perf", "-m", "test_model", "--service-kind", "openai"]
        monkeypatch.setattr("sys.argv", args)
        expected_output = (
            "The --endpoint option is required when using the 'openai' service-kind."
        )

        with pytest.raises(SystemExit) as excinfo:
            parser.parse_args()

        assert excinfo.value.code != 0
        captured = capsys.readouterr()
        assert expected_output in captured.err

    @pytest.mark.parametrize(
        "args, expected_format",
        [
            (
                ["--service-kind", "openai", "--endpoint", "v1/chat/completions"],
                OutputFormat.OPENAI_CHAT_COMPLETIONS,
            ),
            (
                ["--service-kind", "openai", "--endpoint", "v1/completions"],
                OutputFormat.OPENAI_COMPLETIONS,
            ),
            (["--service-kind", "triton", "--backend", "trtllm"], OutputFormat.TRTLLM),
            (["--service-kind", "triton", "--backend", "vllm"], OutputFormat.VLLM),
        ],
    )
    def test_inferred_output_format(self, monkeypatch, args, expected_format):
        monkeypatch.setattr("sys.argv", ["genai-perf", "-m", "test_model"] + args)

        parsed_args, _ = parser.parse_args()
        assert parsed_args.output_format == expected_format
