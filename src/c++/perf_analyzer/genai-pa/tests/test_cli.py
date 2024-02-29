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

import io
import sys

import pytest
from genai_pa import parser
from genai_pa.main import run


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
        "arg, expected_output",
        [
            (["-b", "2"], "batch_size=2"),
            (["--batch-size", "2"], "batch_size=2"),
            (["--concurrency", "3"], "concurrency_range='3'"),
            (["--max-threads", "4"], "max_threads=4"),
            (
                ["--profile-export-file", "text.txt"],
                "profile_export_file=PosixPath('text.txt')",
            ),
            (["--request-rate", "1.5"], "request_rate_range='1.5'"),
            (["--service-kind", "triton"], "service_kind='triton'"),
            (["--service-kind", "openai"], "service_kind='openai'"),
            # TODO: Remove streaming from implementation. It is invalid with HTTP.
            # (["--streaming"], "Streaming=True"),
            (["--version"], "version=True"),
            (["-u", "test_url"], "u='test_url'"),
            (["--url", "test_url"], "u='test_url'"),
        ],
    )
    def test_arguments_output(self, arg, expected_output, capsys):
        combined_args = ["--model", "test_model"] + arg
        _ = parser.parse_args(combined_args)

        # Capture that the correct message was displayed
        captured = capsys.readouterr()
        assert expected_output in captured.out

    def test_arguments_model_not_provided(self):
        with pytest.raises(SystemExit) as exc_info:
            _ = parser.parse_args()

        # Check that the exit was unsuccessful
        assert exc_info.value.code != 0
