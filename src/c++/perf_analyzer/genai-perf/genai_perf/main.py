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

import os
import shutil
import sys
import traceback
from argparse import Namespace
from pathlib import Path

import genai_perf.logging as logging
from genai_perf import parser
from genai_perf.constants import DEFAULT_ARTIFACT_DIR, DEFAULT_PARQUET_FILE
from genai_perf.exceptions import GenAIPerfException
from genai_perf.llm_inputs.llm_inputs import LlmInputs
from genai_perf.llm_metrics import LLMProfileDataParser, Statistics
from genai_perf.plots.plot_manager import PlotManager
from genai_perf.tokenizer import Tokenizer, get_tokenizer


def init_logging() -> None:
    logging.init_logging()


def create_artifacts_dirs(generate_plots: bool) -> None:
    if not os.path.exists(f"{DEFAULT_ARTIFACT_DIR}"):
        os.mkdir(f"{DEFAULT_ARTIFACT_DIR}")
        os.mkdir(f"{DEFAULT_ARTIFACT_DIR}/data")
        os.mkdir(f"{DEFAULT_ARTIFACT_DIR}/plots")


def generate_inputs(args: Namespace, tokenizer: Tokenizer) -> None:
    # TODO (TMA-1759): review if add_model_name is always true
    add_model_name = True
    try:
        extra_input_dict = (
            parser.get_extra_inputs_as_dict(args)
            if args.extra_inputs is not None
            else None
        )
    except ValueError as e:
        raise GenAIPerfException(e)

    LlmInputs.create_llm_inputs(
        input_type=args.prompt_source,
        output_format=args.output_format,
        dataset_name=args.input_dataset if args.input_dataset is not None else "",
        model_name=args.model,
        input_filename=args.input_file if args.input_file is not None else "",
        starting_index=LlmInputs.DEFAULT_STARTING_INDEX,
        length=args.num_prompts,
        prompt_tokens_mean=args.synthetic_input_tokens_mean,
        prompt_tokens_stddev=args.synthetic_input_tokens_stddev,
        output_tokens_mean=args.output_tokens_mean,
        output_tokens_stddev=args.output_tokens_stddev,
        output_tokens_deterministic=args.output_tokens_mean_deterministic,
        random_seed=args.random_seed,
        num_of_output_prompts=args.num_prompts,
        add_model_name=add_model_name,
        add_stream=args.streaming,
        tokenizer=tokenizer,
        extra_inputs=extra_input_dict,
    )


def calculate_metrics(args: Namespace, tokenizer: Tokenizer) -> LLMProfileDataParser:
    return LLMProfileDataParser(
        filename=args.profile_export_file,
        service_kind=args.service_kind,
        output_format=args.output_format,
        tokenizer=tokenizer,
    )


def report_output(data_parser: LLMProfileDataParser, args: Namespace) -> None:
    if "concurrency_range" in args:
        infer_mode = "concurrency"
        load_level = args.concurrency_range
    elif "request_rate_range" in args:
        infer_mode = "request_rate"
        load_level = args.request_rate_range
    else:
        raise GenAIPerfException("No valid infer mode specified")

    stats = data_parser.get_statistics(infer_mode, load_level)
    export_csv_name = args.profile_export_file.with_name(
        args.profile_export_file.stem + "_genai_perf.csv"
    )
    stats.export_to_csv(export_csv_name)
    stats.export_parquet(DEFAULT_PARQUET_FILE)
    stats.pretty_print()
    if args.generate_plots:
        create_plots(stats)


def create_plots(stats: Statistics) -> None:
    plot_manager = PlotManager(stats)
    plot_manager.create_default_plots()


def finalize(profile_export_file: Path):
    shutil.move("llm_inputs.json", f"{DEFAULT_ARTIFACT_DIR}/data/llm_inputs.json")
    shutil.move(
        profile_export_file, f"{DEFAULT_ARTIFACT_DIR}/data/{profile_export_file}"
    )
    profile_export_file_csv = profile_export_file.stem + "_genai_perf.csv"
    shutil.move(
        profile_export_file_csv,
        f"{DEFAULT_ARTIFACT_DIR}/data/{profile_export_file_csv}",
    )


# Separate function that can raise exceptions used for testing
# to assert correct errors and messages.
def run():
    try:
        init_logging()
        args, extra_args = parser.parse_args()
        create_artifacts_dirs(args.generate_plots)
        tokenizer = get_tokenizer(args.tokenizer)
        generate_inputs(args, tokenizer)
        args.func(args, extra_args)
        data_parser = calculate_metrics(args, tokenizer)
        report_output(data_parser, args)
        finalize(args.profile_export_file)
    except Exception as e:
        raise GenAIPerfException(e)


def main():
    # Interactive use will catch exceptions and log formatted errors rather than tracebacks.
    try:
        run()
    except Exception as e:
        traceback.print_exc()
        logger = logging.getLogger(__name__)
        logger.error(e)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
