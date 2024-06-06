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
import sys
import traceback
from argparse import Namespace
from pathlib import Path

import genai_perf.logging as logging
from genai_perf import parser
from genai_perf.constants import DEFAULT_PARQUET_FILE
from genai_perf.exceptions import GenAIPerfException
from genai_perf.export_data.output_reporter import OutputReporter
from genai_perf.llm_inputs.llm_inputs import LlmInputs
from genai_perf.llm_metrics import LLMProfileDataParser
from genai_perf.plots.plot_config_parser import PlotConfigParser
from genai_perf.plots.plot_manager import PlotManager
from genai_perf.tokenizer import Tokenizer, get_tokenizer


def create_artifacts_dirs(args: Namespace) -> None:
    # TMA-1911: support plots CLI option
    plot_dir = args.artifact_dir / "plots"
    os.makedirs(args.artifact_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)


def generate_inputs(args: Namespace, tokenizer: Tokenizer) -> None:
    # TODO (TMA-1759): review if add_model_name is always true
    input_filename = Path(args.input_file.name) if args.input_file else None
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
        model_selection_strategy=args.model_selection_strategy,
        input_filename=input_filename,
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
        output_dir=args.artifact_dir,
    )


def calculate_metrics(args: Namespace, tokenizer: Tokenizer) -> LLMProfileDataParser:
    return LLMProfileDataParser(
        filename=args.profile_export_file,
        tokenizer=tokenizer,
    )


def report_output(data_parser: LLMProfileDataParser, args: Namespace) -> None:
    reporter = OutputReporter(data_parser, args)
    reporter.report_output()
    if args.generate_plots:
        create_plots(args)


def create_plots(args: Namespace) -> None:
    # TMA-1911: support plots CLI option
    plot_dir = args.artifact_dir / "plots"
    PlotConfigParser.create_init_yaml_config(
        filenames=[args.profile_export_file],  # single run
        output_dir=plot_dir,
    )
    config_parser = PlotConfigParser(plot_dir / "config.yaml")
    plot_configs = config_parser.generate_configs()
    plot_manager = PlotManager(plot_configs)
    plot_manager.generate_plots()


# Separate function that can raise exceptions used for testing
# to assert correct errors and messages.
def run():
    try:
        # TMA-1900: refactor CLI handler
        logging.init_logging()
        args, extra_args = parser.parse_args()
        if args.subcommand == "compare":
            args.func(args)
        else:
            create_artifacts_dirs(args)
            tokenizer = get_tokenizer(args.tokenizer)
            generate_inputs(args, tokenizer)
            args.func(args, extra_args)
            data_parser = calculate_metrics(args, tokenizer)
            report_output(data_parser, args)
    except Exception as e:
        raise GenAIPerfException(e)


def main():
    # Interactive use will catch exceptions and log formatted errors rather than
    # tracebacks.
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
