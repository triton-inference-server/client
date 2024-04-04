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

import logging
import sys
from argparse import ArgumentParser

from genai_perf import parser
from genai_perf.constants import LOGGER_NAME
from genai_perf.exceptions import GenAIPerfException
from genai_perf.llm_inputs.llm_inputs import LlmInputs
from genai_perf.llm_metrics import LLMProfileDataParser
from genai_perf.tokenizer import AutoTokenizer, get_tokenizer

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(LOGGER_NAME)


def generate_inputs(args: ArgumentParser, tokenizer: AutoTokenizer) -> None:
    # TODO (TMA-1758): remove once file support is implemented
    input_file_name = ""
    # TODO (TMA-1759): review if add_model_name is always true
    add_model_name = True
    try:
        extra_input_dict = parser.get_extra_inputs_as_dict(args)
    except ValueError as e:
        raise GenAIPerfException(e)

    LlmInputs.create_llm_inputs(
        input_type=args.prompt_source,
        output_format=args.output_format,
        dataset_name=args.input_dataset,
        model_name=args.model,
        input_filename=input_file_name,
        starting_index=LlmInputs.DEFAULT_STARTING_INDEX,
        length=args.num_prompts,
        prompt_tokens_mean=args.synthetic_tokens_mean,
        prompt_tokens_stddev=args.synthetic_tokens_stddev,
        expected_output_tokens=args.synthetic_requested_output_tokens,
        random_seed=args.random_seed,
        num_of_output_prompts=args.num_prompts,
        add_model_name=add_model_name,
        add_stream=args.streaming,
        tokenizer=tokenizer,
        extra_inputs=extra_input_dict,
    )


def calculate_metrics(
    args: ArgumentParser, tokenizer: AutoTokenizer
) -> LLMProfileDataParser:
    return LLMProfileDataParser(
        filename=args.profile_export_file,
        service_kind=args.service_kind,
        output_format=args.output_format,
        tokenizer=tokenizer,
    )


def report_output(metrics: LLMProfileDataParser, args):
    if "concurrency_range" in args:
        infer_mode = "concurrency"
        load_level = args.concurrency_range
    elif "request_rate_range" in args:
        infer_mode = "request_rate"
        load_level = args.request_rate_range
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
        tokenizer = get_tokenizer(args.tokenizer)
        generate_inputs(args, tokenizer)
        args.func(args, extra_args)
        metrics = calculate_metrics(args, tokenizer)
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
