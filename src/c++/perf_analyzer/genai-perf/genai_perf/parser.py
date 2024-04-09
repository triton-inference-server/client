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

import argparse
import logging
import sys
from pathlib import Path

import genai_perf.utils as utils
from genai_perf.constants import CNN_DAILY_MAIL, LOGGER_NAME, OPEN_ORCA
from genai_perf.llm_inputs.llm_inputs import LlmInputs, OutputFormat, PromptSource
from genai_perf.tokenizer import DEFAULT_TOKENIZER

from . import __version__

logger = logging.getLogger(LOGGER_NAME)


def _check_conditional_args(
    parser: argparse.ArgumentParser, args: argparse.ArgumentParser
) -> argparse.ArgumentParser:
    """
    Check for conditional args and raise an error if they are not set.
    """
    if args.service_kind == "openai":
        if args.endpoint is None:
            parser.error(
                "The --endpoint option is required when using the 'openai' service-kind."
            )
        if args.endpoint == "v1/chat/completions":
            args.output_format = OutputFormat.OPENAI_CHAT_COMPLETIONS
        elif args.endpoint == "v1/completions":
            args.output_format = OutputFormat.OPENAI_COMPLETIONS
    elif args.endpoint is not None:
        parser.error(
            "The --endpoint option should only be used when using the 'openai' service-kind."
        )
    if args.service_kind == "triton":
        args = _convert_str_to_enum_entry(args, "backend", OutputFormat)
        args.output_format = args.backend
    if (
        args.output_tokens_mean == LlmInputs.DEFAULT_OUTPUT_TOKENS_MEAN
        and args.output_tokens_stddev != LlmInputs.DEFAULT_OUTPUT_TOKENS_STDDEV
    ):
        parser.error(
            "The --output-tokens-mean option is required when using --output-tokens-stddev."
        )

    return args


def _prune_args(args: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Prune the parsed arguments to remove args with None.
    """
    return argparse.Namespace(**{k: v for k, v in vars(args).items() if v is not None})


def _update_load_manager_args(args: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Update genai-perf load manager attributes to PA format
    """
    for attr_key in ["concurrency", "request_rate"]:
        attr_val = getattr(args, attr_key)
        if attr_val is not None:
            setattr(args, f"{attr_key}_range", f"{attr_val}")
            delattr(args, attr_key)
            return args

    # If no concurrency or request rate is set, default to 1
    setattr(args, "concurrency_range", "1")
    return args


def _convert_str_to_enum_entry(args, option, enum):
    """
    Convert string option to corresponding enum entry
    """
    attr_val = getattr(args, option)
    if attr_val is not None:
        setattr(args, f"{option}", utils.get_enum_entry(attr_val, enum))
    return args


### Handlers ###


def handler(args, extra_args):
    from genai_perf.wrapper import Profiler

    Profiler.run(args=args, extra_args=extra_args)


### Parsers ###


def _add_input_args(parser):
    input_group = parser.add_argument_group("Input")

    input_group.add_argument(
        "--extra-inputs",
        action="append",
        help="Provide additional inputs to include with every request. "
        "You can repeat this flag for multiple inputs. Inputs should be in an input_name:value format.",
    )

    input_group.add_argument(
        "--input-dataset",
        type=str.lower,
        default=OPEN_ORCA,
        choices=[OPEN_ORCA, CNN_DAILY_MAIL],
        required=False,
        help="The HuggingFace dataset to use for prompts when prompt-source is dataset.",
    )

    input_group.add_argument(
        "--num-prompts",
        type=int,
        default=LlmInputs.DEFAULT_NUM_PROMPTS,
        required=False,
        help=f"The number of unique prompts to generate as stimulus.",
    )

    input_group.add_argument(
        "--output-tokens-mean",
        type=int,
        default=LlmInputs.DEFAULT_OUTPUT_TOKENS_MEAN,
        required=False,
        help=f"The mean number of tokens in each output. "
        "Ensure the --tokenizer value is set correctly. "
        "Note that there is still some variability in the requested number "
        "of output tokens, but GenAi-Perf attempts its best effort with your "
        "model to get the right number of output tokens. ",
    )

    input_group.add_argument(
        "--output-tokens-stddev",
        type=int,
        default=LlmInputs.DEFAULT_OUTPUT_TOKENS_STDDEV,
        required=False,
        help=f"The standard deviation of the number of tokens in each output. "
        "This is only used when output-tokens-mean is provided.",
    )

    input_group.add_argument(
        "--prompt-source",
        type=str,
        choices=utils.get_enum_names(PromptSource),
        default="synthetic",
        required=False,
        help=f"The source of the input prompts.",
    )

    input_group.add_argument(
        "--random-seed",
        type=int,
        default=LlmInputs.DEFAULT_RANDOM_SEED,
        required=False,
        help="The seed used to generate random values.",
    )

    input_group.add_argument(
        "--synthetic-input-tokens-mean",
        type=int,
        default=LlmInputs.DEFAULT_PROMPT_TOKENS_MEAN,
        required=False,
        help=f"The mean of number of tokens in the generated prompts when prompt-source is synthetic.",
    )

    input_group.add_argument(
        "--synthetic-input-tokens-stddev",
        type=int,
        default=LlmInputs.DEFAULT_PROMPT_TOKENS_STDDEV,
        required=False,
        help=f"The standard deviation of number of tokens in the generated prompts when prompt-source is synthetic.",
    )


def _add_profile_args(parser):
    profile_group = parser.add_argument_group("Profiling")
    load_management_group = profile_group.add_mutually_exclusive_group(required=False)

    load_management_group.add_argument(
        "--concurrency",
        type=int,
        required=False,
        help="The concurrency value to benchmark.",
    )

    profile_group.add_argument(
        "--measurement-interval",
        "-p",
        type=int,
        default="10000",
        required=False,
        help="The time interval used "
        "for each measurement in milliseconds. The perf analyzer will "
        "sample a time interval specified and take measurement over "
        "the requests completed within that time interval.",
    )

    load_management_group.add_argument(
        "--request-rate",
        type=float,
        required=False,
        help="Sets the request rate for the load generated by PA. ",
    )

    profile_group.add_argument(
        "-s",
        "--stability-percentage",
        type=float,
        default=999,
        required=False,
        help="The allowed variation in "
        "latency measurements when determining if a result is stable. The "
        "measurement is considered as stable if the ratio of max / min "
        "from the recent 3 measurements is within (stability percentage) "
        "in terms of both infer per second and latency.",
    )


def _add_endpoint_args(parser):
    endpoint_group = parser.add_argument_group("Endpoint")

    endpoint_group.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help=f"The name of the model to benchmark.",
    )

    endpoint_group.add_argument(
        "--backend",
        type=str,
        choices=utils.get_enum_names(OutputFormat)[2:],
        default="trtllm",
        required=False,
        help=f'When using the "triton" service-kind, '
        "this is the backend of the model. "
        "For the TRT-LLM backend, you currently must set 'exclude_input_in_output' to true in the model config to "
        "not echo the input tokens in the output.",
    )

    endpoint_group.add_argument(
        "--endpoint",
        type=str,
        choices=["v1/chat/completions", "v1/completions"],
        required=False,
        help=f"The endpoint to send requests to on the "
        'server. This is only used with the "openai" service-kind. ',
    )

    endpoint_group.add_argument(
        "--service-kind",
        type=str,
        choices=["triton", "openai"],
        default="triton",
        required=False,
        help="The kind of service perf_analyzer will "
        'generate load for. In order to use "openai", '
        "you must specify an endpoint via --endpoint.",
    )

    endpoint_group.add_argument(
        "--streaming",
        action="store_true",
        required=False,
        help=f"An option to enable the use of the streaming API.",
    )

    endpoint_group.add_argument(
        "-u",
        "--url",
        type=str,
        required=False,
        dest="u",
        metavar="URL",
        help="URL of the endpoint to target for benchmarking.",
    )


def _add_output_args(parser):
    output_group = parser.add_argument_group("Output")

    output_group.add_argument(
        "--profile-export-file",
        type=Path,
        default="profile_export.json",
        help="The path where the perf_analyzer profile export will be "
        "generated. By default, the profile export will be to profile_export.json. "
        "The genai-perf file will be exported to <profile_export_file>_genai_perf.csv. "
        "For example, if the profile export file is profile_export.json, the genai-perf file will be "
        "exported to profile_export_genai_perf.csv.",
    )


def _add_other_args(parser):
    output_group = parser.add_argument_group("Other")

    output_group.add_argument(
        "--tokenizer",
        type=str,
        default=DEFAULT_TOKENIZER,
        required=False,
        help="The HuggingFace tokenizer to use to interpret token metrics from prompts and responses",
    )

    output_group.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        required=False,
        help="An option to enable verbose mode.",
    )

    output_group.add_argument(
        "--version",
        action="version",
        version="%(prog)s " + __version__,
        help=f"An option to print the version and exit.",
    )


def get_extra_inputs_as_dict(args: argparse.ArgumentParser) -> dict:
    request_inputs = {}
    if hasattr(args, "extra_inputs"):
        for input_str in args.extra_inputs:
            semicolon_count = input_str.count(":")
            if semicolon_count != 1:
                raise ValueError(
                    f"Invalid input format for --extra-inputs: {input_str}\n"
                    "Expected input format: 'input_name:value'"
                )
            input_name, value = input_str.split(":", 1)

            if not input_name or not value:
                raise ValueError(
                    f"Input name or value is empty in --extra-inputs: {input_str}\n"
                    "Expected input format: 'input_name:value'"
                )

            is_bool = value.lower() in ["true", "false"]
            is_int = value.isdigit()
            is_float = value.count(".") == 1 and (
                value[0] == "." or value.replace(".", "").isdigit()
            )

            if is_bool:
                value = value.lower() == "true"
            elif is_int:
                value = int(value)
            elif is_float:
                value = float(value)

            if input_name in request_inputs:
                raise ValueError(
                    f"Input name already exists in request_inputs dictionary: {input_name}"
                )
            request_inputs[input_name] = value

    return request_inputs


### Entrypoint ###


def parse_args():
    argv = sys.argv

    parser = argparse.ArgumentParser(
        prog="genai-perf",
        description="CLI to profile LLMs and Generative AI models with Perf Analyzer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.set_defaults(func=handler)

    # Conceptually group args for easier visualization
    _add_endpoint_args(parser)
    _add_input_args(parser)
    _add_profile_args(parser)
    _add_output_args(parser)
    _add_other_args(parser)

    # Check for passthrough args
    if "--" in argv:
        passthrough_index = argv.index("--")
        logger.info(f"Detected passthrough args: {argv[passthrough_index + 1:]}")
    else:
        passthrough_index = len(argv)

    args = parser.parse_args(argv[1:passthrough_index])
    args = _check_conditional_args(parser, args)
    args = _update_load_manager_args(args)
    args = _convert_str_to_enum_entry(args, "prompt_source", PromptSource)
    args = _prune_args(args)

    return args, argv[passthrough_index + 1 :]
