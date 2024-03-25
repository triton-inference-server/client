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

import subprocess
from unittest.mock import ANY, MagicMock, patch

import pytest
from genai_perf import parser
from genai_perf.constants import DEFAULT_GRPC_URL
from genai_perf.wrapper import Profiler


class TestWrapper:
    @pytest.mark.parametrize(
        "arg",
        [
            ([]),
            (["-u", "testurl:1000"]),
            (["--url", "testurl:1000"]),
        ],
    )
    def test_url_exactly_once_triton(self, monkeypatch, arg):
        args = ["genai-perf", "-m", "test_model", "--service-kind", "triton"] + arg
        monkeypatch.setattr("sys.argv", args)
        args, extra_args = parser.parse_args()
        cmd_string = Profiler.build_cmd(args, extra_args)

        number_of_url_args = cmd_string.count(" -u ") + cmd_string.count(" --url ")
        assert number_of_url_args == 1

    @pytest.mark.parametrize(
        "arg",
        [
            (["--backend", "trtllm"]),
            (["--backend", "vllm"]),
        ],
    )
    def test_service_triton(self, monkeypatch, arg):
        args = ["genai-perf", "-m", "test_model", "--service-kind", "triton"] + arg
        monkeypatch.setattr("sys.argv", args)
        args, extra_args = parser.parse_args()
        cmd_string = Profiler.build_cmd(args, extra_args)

        # Ensure the correct arguments are appended.
        assert cmd_string.count(" -i grpc") == 1
        assert cmd_string.count(" --streaming") == 1
        assert cmd_string.count(f"-u {DEFAULT_GRPC_URL}") == 1
        if arg[1] == "trtllm":
            assert cmd_string.count("--shape max_tokens:1") == 1
            assert cmd_string.count("--shape text_input:1") == 1

    @pytest.mark.parametrize(
        "arg",
        [
            (["--endpoint", "v1/completions"]),
            (["--endpoint", "v1/chat/completions"]),
        ],
    )
    def test_service_openai(self, monkeypatch, arg):
        args = [
            "genai-perf",
            "-m",
            "test_model",
            "--service-kind",
            "openai",
        ] + arg
        monkeypatch.setattr("sys.argv", args)
        args, extra_args = parser.parse_args()
        cmd_string = Profiler.build_cmd(args, extra_args)

        # Ensure the correct arguments are appended.
        assert cmd_string.count(" -i http") == 1

    @patch("genai_perf.wrapper.subprocess.run")
    def test_stdout_verbose(self, mock_subprocess_run):
        args = MagicMock()
        args.model = "test_model"
        args.verbose = True
        Profiler.run(args=args, extra_args=None)

        # Check that standard output was not redirected.
        for call_args in mock_subprocess_run.call_args_list:
            _, kwargs = call_args
            assert (
                "stdout" not in kwargs or kwargs["stdout"] is None
            ), "With the verbose flag, stdout should not be redirected."

    @patch("genai_perf.wrapper.subprocess.run")
    def test_stdout_not_verbose(self, mock_subprocess_run):
        args = MagicMock()
        args.model = "test_model"
        args.verbose = False
        Profiler.run(args=args, extra_args=None)

        # Check that standard output was redirected.
        for call_args in mock_subprocess_run.call_args_list:
            _, kwargs = call_args
            assert (
                kwargs["stdout"] is subprocess.DEVNULL
            ), "When the verbose flag is not passed, stdout should be redirected to /dev/null."
