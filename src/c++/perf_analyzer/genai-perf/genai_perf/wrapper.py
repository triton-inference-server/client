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
from argparse import Namespace

import genai_perf.logging as logging
import genai_perf.utils as utils
from genai_perf.constants import DEFAULT_GRPC_URL, DEFAULT_INPUT_DATA_JSON
from genai_perf.llm_inputs.llm_inputs import OutputFormat

logger = logging.getLogger(__name__)


class Profiler:
    @staticmethod
    def add_protocol_args(args: Namespace):
        cmd = [""]
        if args.service_kind == "triton":
            cmd += ["-i", "grpc", "--streaming"]
            if "u" not in vars(args).keys():
                cmd += ["-u", f"{DEFAULT_GRPC_URL}"]
            if args.output_format == OutputFormat.TENSORRTLLM:
                cmd += ["--shape", "max_tokens:1", "--shape", "text_input:1"]
        elif args.service_kind == "openai":
            cmd += ["-i", "http"]
        return cmd

    @staticmethod
    def build_cmd(args: Namespace, extra_args: list[str] | None = None) -> list[str]:
        skip_args = [
            "func",
            "input_dataset",
            "prompt_source",
            "input_format",
            "model",
            "backend",
            "extra_inputs",
            "output_format",
            # The 'streaming' passed in to this script is to determine if the
            # LLM response should be streaming. That is different than the
            # 'streaming' that PA takes, which means something else (and is
            # required for decoupled models into triton).
            "streaming",
            "synthetic_input_tokens_mean",
            "synthetic_input_tokens_stddev",
            "output_tokens_mean",
            "output_tokens_stddev",
            "output_tokens_mean_deterministic",
            "num_prompts",
            "random_seed",
            "tokenizer",
            "endpoint_type",
            "generate_plots",
        ]

        utils.remove_file(args.profile_export_file)

        cmd = [
            f"perf_analyzer",
            f"-m",
            f"{args.model}",
            f"--async",
            f"--input-data",
            f"{DEFAULT_INPUT_DATA_JSON}",
        ]
        for arg, value in vars(args).items():
            if arg in skip_args:
                pass
            elif value is False:
                pass
            elif value is True:
                if len(arg) == 1:
                    cmd += [f"-{arg}"]
                else:
                    cmd += [f"--{arg}"]
            else:
                if len(arg) == 1:
                    cmd += [f"-{arg}", f"{value}"]
                else:
                    arg = utils.convert_option_name(arg)
                    cmd += [f"--{arg}", f"{value}"]

        cmd += Profiler.add_protocol_args(args)

        if extra_args is not None:
            for arg in extra_args:
                cmd += [f"{arg}"]
        return cmd

    @staticmethod
    def run(args: Namespace, extra_args: list[str] | None) -> None:
        cmd = Profiler.build_cmd(args, extra_args)
        logger.info(f"Running Perf Analyzer : '{' '.join(cmd)}'")
        if args and args.verbose:
            subprocess.run(cmd, check=True, stdout=None)
        else:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
