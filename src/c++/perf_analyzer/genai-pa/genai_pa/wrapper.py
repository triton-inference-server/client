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

import logging
import subprocess

import genai_pa.utils as utils
from genai_pa.constants import LOGGER_NAME
from genai_pa.llm_inputs.llm_inputs import OutputFormat

logger = logging.getLogger(LOGGER_NAME)


class Profiler:
    @staticmethod
    def add_protocol_args(args):
        cmd = ""
        if args.service_kind == "triton":
            cmd += f"-i grpc "
            if args.output_format == OutputFormat.TRTLLM:
                cmd += f"--shape max_tokens:1 --shape text_input:1 "
        elif args.service_kind == "openai":
            cmd += f"-i http "
        return cmd

    @staticmethod
    def build_cmd(model, args, extra_args):
        skip_args = [
            "model",
            "func",
            "dataset",
            "input_type",
            "input_format",
            "output_format",
        ]
        # OpenAI service-kind sets streaming via the input file, not via the CLI
        if args.service_kind == "openai":
            skip_args.append("streaming")

        if hasattr(args, "version") and args.version:
            cmd = f"perf_analyzer --version"
        else:
            utils.remove_file(args.profile_export_file)

            cmd = f"perf_analyzer -m {model} --async "
            for arg, value in vars(args).items():
                if arg in skip_args:
                    pass
                elif value is False:
                    pass
                elif value is True:
                    cmd += f"--{arg} "
                elif arg == "batch_size":
                    cmd += f"-b {value} "
                else:
                    if len(arg) == 1:
                        cmd += f"-{arg} {value} "
                    else:
                        arg = utils.convert_option_name(arg)
                        cmd += f"--{arg} {value} "

            cmd += Profiler.add_protocol_args(args)

            if extra_args is not None:
                for arg in extra_args:
                    cmd += f"{arg} "
        return cmd

    @staticmethod
    def run(model, args=None, extra_args=None):
        cmd = Profiler.build_cmd(model, args, extra_args)
        logger.info(f"Running Perf Analyzer : '{cmd}'")
        subprocess.run(cmd, shell=True, check=True)
