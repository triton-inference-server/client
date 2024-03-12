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

from genai_perf import parser
from genai_perf.constants import LOGGER_NAME
from genai_perf.exceptions import GenAIPerfException
from genai_perf.llm_inputs.llm_inputs import LlmInputs

# Silence tokenizer warning on import
with contextlib.redirect_stdout(io.StringIO()) as stdout, contextlib.redirect_stderr(
    io.StringIO()
) as stderr:
    from genai_perf.llm_metrics import LLMProfileDataParser
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
        input_type=args.input_type,
        output_format=args.output_format,
        dataset_name=args.dataset,
        model_name=args.model,
        input_filename=input_file_name,
        starting_index=LlmInputs.DEFAULT_STARTING_INDEX,
        length=LlmInputs.DEFAULT_LENGTH,
        prompt_tokens_mean=args.mean_input_tokens,
        prompt_tokens_stddev=args.stddev_input_tokens,
        add_model_name=add_model_name,
        add_stream=args.streaming,
    )


def calculate_metrics(file: str, service_kind: str) -> LLMProfileDataParser:
    t = tokenizer.from_pretrained("gpt2")
    return LLMProfileDataParser(file, service_kind, t)


def report_output(metrics: LLMProfileDataParser, args):
    if "concurrency_range" in args:
        infer_mode = "concurrency"
        load_level = args.concurrency_range
    elif "request_rate_range" in args:
        infer_mode = "request_rate"
        load_level = args.request_rate_range
    else:
        raise GenAIPerfException(
            "Neither concurrency_range nor request_rate_range was found in args when reporting metrics"
        )
    stats = metrics.get_statistics(infer_mode, load_level)
    export_csv_name = args.profile_export_file.with_name(
        args.profile_export_file.stem + "_genai_perf.csv"
    )
    stats.export_to_csv(export_csv_name)
    stats.pretty_print()


# Separate function that can raise exceptions used for testing
# to assert correct errors and messages.
def run():
    try:
        args, extra_args = parser.parse_args()
        generate_inputs(args)
        args.func(args, extra_args)
        metrics = calculate_metrics(args.profile_export_file, args.service_kind)
        report_output(metrics, args)
    except Exception as e:
        raise GenAIPerfException(e)


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
