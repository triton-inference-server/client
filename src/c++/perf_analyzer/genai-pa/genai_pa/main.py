#!/usr/bin/env python3
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

import contextlib
import io
import logging
import sys

from genai_pa import parser
from genai_pa.constants import LOGGER_NAME
from genai_pa.exceptions import GenAiPAException
from genai_pa.llm_inputs.llm_inputs import LlmInputs

# Silence tokenizer warning on import
with contextlib.redirect_stdout(io.StringIO()) as stdout, contextlib.redirect_stderr(
    io.StringIO()
) as stderr:
    from genai_pa.llm_metrics import LLMProfileData
    from transformers import AutoTokenizer as tokenizer
    from transformers import logging as token_logger

    token_logger.set_verbosity_error()


logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(LOGGER_NAME)


def generate_inputs(args):
    # TODO: remove once fale support is in
    input_file_name = ""
    # TODO: review if always true
    add_model_name = True
    LlmInputs.create_llm_inputs(
        args.input_type,
        args.output_format,
        args.dataset,
        args.model,
        input_file_name,
        LlmInputs.DEFAULT_STARTING_INDEX,
        LlmInputs.DEFAULT_LENGTH,
        add_model_name,
        args.streaming,
    )


def calculate_metrics(file: str) -> LLMProfileData:
    t = tokenizer.from_pretrained("gpt2")
    return LLMProfileData(file, t)


def report_output(metrics: LLMProfileData, args):
    if "concurrency_range" in args:
        infer_mode = "concurrency"
        load_level = args.concurrency_range
    elif "request_rate_range" in args:
        infer_mode = "request_rate"
        load_level = args.request_rate_range
    else:
        raise GenAiPAException(
            "Neither concurrency_range nor request_rate_range was found in args when reporting metrics"
        )
    stats = metrics.get_statistics(infer_mode, int(load_level))
    export_csv_name = args.profile_export_file.with_name(
        args.profile_export_file.stem + "_genai_pa" + args.profile_export_file.suffix
    )
    stats.export_to_csv(export_csv_name)
    stats.pretty_print()


# Separate function that can raise exceptions used for testing
# to assert correct errors and messages.
# Optional argv used for testing - will default to sys.argv if None.
def run(argv=None):
    try:
        args, extra_args = parser.parse_args(argv)
        generate_inputs(args)
        args.func(args, extra_args)
        metrics = calculate_metrics(args.profile_export_file)
        report_output(metrics, args)
    except Exception as e:
        raise GenAiPAException(e)


def main():
    # Interactive use will catch exceptions and log formatted errors rather than tracebacks.
    try:
        run()
    except Exception as e:
        logger.error(f"{e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
