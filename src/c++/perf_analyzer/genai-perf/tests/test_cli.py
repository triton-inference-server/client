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
from genai_perf import parser
from genai_perf.exceptions import GenAIPerfException
from genai_perf.main import run


class TestCLIArguments:
    @pytest.mark.parametrize(
        "arg, expected_output",
        [
            (["-h"], "CLI to profile LLMs and Generative AI models with Perf Analyzer"),
            (
                ["--help"],
                "CLI to profile LLMs and Generative AI models with Perf Analyzer",
            ),
        ],
    )
    def test_help_arguments_output_and_exit(self, arg, expected_output, capsys):
        with pytest.raises(SystemExit) as excinfo:
            _ = parser.parse_args(arg)

        # Check that the exit was successful
        assert excinfo.value.code == 0

        # Capture that the correct message was displayed
        captured = capsys.readouterr()
        assert expected_output in captured.out

    @pytest.mark.parametrize(
        "arg, expected_attributes",
        [
            (
                ["--profile-export-file", "text.txt"],
                {"profile_export_file": Path("text.txt")},
            ),
            (["--service-kind", "triton"], {"service_kind": "triton"}),
            (["--service-kind", "openai"], {"service_kind": "openai"}),
            (["--version"], {"version": True}),
            (["-u", "test_url"], {"u": "test_url"}),
            (["--url", "test_url"], {"u": "test_url"}),
        ],
    )
    def test_arguments_output(self, arg, expected_attributes, capsys):
        combined_args = ["--model", "test_model", "--concurrency", "2"] + arg
        args, _ = parser.parse_args(combined_args)

        # Check that the attributes are set correctly
        for key, value in expected_attributes.items():
            assert getattr(args, key) == value

        # Check that nothing was printed as a byproduct of parsing the arguments
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_arguments_model_not_provided(self):
        with pytest.raises(SystemExit) as exc_info:
            _ = parser.parse_args()

        # Check that the exit was unsuccessful
        assert exc_info.value.code != 0

    def test_exception_on_nonzero_exit(self):
        with pytest.raises(GenAIPerfException) as e:
            run(["-m", "nonexistent_model", "--concurrency", "3"])

    def test_pass_through_args(self):
        args = ["-m", "test_model", "--concurrency", "1"]
        other_args = ["--", "With", "great", "power"]
        _, pass_through_args = parser.parse_args(args + other_args)

        assert pass_through_args == other_args[1:]
