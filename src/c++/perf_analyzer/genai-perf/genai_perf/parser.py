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
import os
import sys
from pathlib import Path

import genai_perf.logging as logging
import genai_perf.utils as utils
from genai_perf.constants import (
    CNN_DAILY_MAIL,
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_COMPARE_DIR,
    OPEN_ORCA,
)
from genai_perf.llm_inputs.llm_inputs import (
    LlmInputs,
    ModelSelectionStrategy,
    OutputFormat,
    PromptSource,
)
from genai_perf.plots.plot_config_parser import PlotConfigParser
from genai_perf.plots.plot_manager import PlotManager
from genai_perf.tokenizer import DEFAULT_TOKENIZER

from . import __version__

logger = logging.getLogger(__name__)

_endpoint_type_map = {
    "chat": "v1/chat/completions",
    "completions": "v1/completions",
    "generate": "v2/models/{MODEL_NAME}/generate",
}


def _check_model_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> argparse.Namespace:
    """
    Check if model name is provided.
    """
    if not args.subcommand and not args.model:
        parser.error("The -m/--model option is required and cannot be empty.")
    args = _convert_str_to_enum_entry(
        args, "model_selection_strategy", ModelSelectionStrategy
    )
    _generate_formatted_model_name(args)
    return args


def _generate_formatted_model_name(args: argparse.Namespace) -> None:
    if len(args.model) == 1:
        args.formatted_model_name = args.model[0]
    elif len(args.model) == 0:
        args.model = None
        args.formatted_model_name = None
    else:
        args.formatted_model_name = args.model[0] + "_multi"


def _check_compare_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> argparse.Namespace:
    """
    Check compare subcommand args
    """
    if args.subcommand == "compare":
        if not args.config and not args.files:
            parser.error("Either the --config or --files option must be specified.")
    return args


def _check_conditional_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> argparse.Namespace:
    """
    Check for conditional args and raise an error if they are not set.
    """

    if args.endpoint_type == "chat":
        args.output_format = OutputFormat.OPENAI_CHAT_COMPLETIONS
        args.service_kind = "openai"
    elif args.endpoint_type == "completions":
        args.output_format = OutputFormat.OPENAI_COMPLETIONS
        args.service_kind = "openai"
    elif args.endpoint_type == "generate":
        args.output_format = OutputFormat.TRITON_GENERATE
        args.service_kind = "openai"
    elif args.endpoint_type == "kserve":
        args.service_kind = "triton"
        args = _convert_str_to_enum_entry(args, "backend", OutputFormat)
        args.output_format = args.backend

    if args.endpoint is not None:
        args.endpoint = args.endpoint.lstrip(" /")
    else:
        args.endpoint = _endpoint_type_map[args.endpoint_type].format(
            MODEL_NAME=args.model
        )

    # Output token distribution checks
    if args.output_tokens_mean == LlmInputs.DEFAULT_OUTPUT_TOKENS_MEAN:
        if args.output_tokens_stddev != LlmInputs.DEFAULT_OUTPUT_TOKENS_STDDEV:
            parser.error(
                "The --output-tokens-mean option is required when using --output-tokens-stddev."
            )
        if args.output_tokens_mean_deterministic:
            parser.error(
                "The --output-tokens-mean option is required when using --output-tokens-mean-deterministic."
            )

    if args.service_kind != "triton":
        if args.output_tokens_mean_deterministic:
            parser.error(
                "The --output-tokens-mean-deterministic option is only supported with the Triton service-kind."
            )

    return args


def _check_load_manager_args(args: argparse.Namespace) -> argparse.Namespace:
    """
    Check inference load args
    """
    # If no concurrency or request rate is set, default to 1
    if not args.concurrency and not args.request_rate:
        args.concurrency = 1
    return args


def _set_artifact_paths(args: argparse.Namespace) -> argparse.Namespace:
    """
    Set paths for all the artifacts.
    """
    if args.artifact_dir == Path(DEFAULT_ARTIFACT_DIR):
        # Preprocess Huggingface model names that include '/' in their model name.
        if (args.formatted_model_name is not None) and (
            "/" in args.formatted_model_name
        ):
            filtered_name = "_".join(args.formatted_model_name.split("/"))
            logger.info(
                f"Model name '{args.formatted_model_name}' cannot be used to create artifact "
                f"directory. Instead, '{filtered_name}' will be used."
            )
            name = [f"{filtered_name}"]
        else:
            name = [f"{args.formatted_model_name}"]

        if args.service_kind == "openai":
            name += [f"{args.service_kind}-{args.endpoint_type}"]
        elif args.service_kind == "triton":
            name += [f"{args.service_kind}-{args.backend.to_lowercase()}"]
        else:
            raise ValueError(f"Unknown service kind '{args.service_kind}'.")

        if args.concurrency:
            name += [f"concurrency{args.concurrency}"]
        elif args.request_rate:
            name += [f"request_rate{args.request_rate}"]
        args.artifact_dir = args.artifact_dir / Path("-".join(name))

    if args.profile_export_file.parent != Path(""):
        raise ValueError(
            "Please use --artifact-dir option to define intermediary paths to "
            "the profile export file."
        )

    args.profile_export_file = args.artifact_dir / args.profile_export_file
    return args


def _infer_prompt_source(args: argparse.Namespace) -> argparse.Namespace:
    if args.input_dataset:
        args.prompt_source = PromptSource.DATASET
        logger.debug(f"Input source is the following dataset: {args.input_dataset}")
    elif args.input_file:
        args.prompt_source = PromptSource.FILE
        logger.debug(f"Input source is the following file: {args.input_file.name}")
    else:
        args.prompt_source = PromptSource.SYNTHETIC
        logger.debug("Input source is synthetic data")
    return args


def _convert_str_to_enum_entry(args, option, enum):
    """
    Convert string option to corresponding enum entry
    """
    attr_val = getattr(args, option)
    if attr_val is not None:
        setattr(args, f"{option}", utils.get_enum_entry(attr_val, enum))
    return args


### Parsers ###


def _add_input_args(parser):
    input_group = parser.add_argument_group("Input")

    input_group.add_argument(
        "--extra-inputs",
        action="append",
        help="Provide additional inputs to include with every request. "
        "You can repeat this flag for multiple inputs. Inputs should be in an input_name:value format.",
    )

    prompt_source_group = input_group.add_mutually_exclusive_group(required=False)
    prompt_source_group.add_argument(
        "--input-dataset",
        type=str.lower,
        default=None,
        choices=[OPEN_ORCA, CNN_DAILY_MAIL],
        required=False,
        help="The HuggingFace dataset to use for prompts.",
    )

    prompt_source_group.add_argument(
        "--input-file",
        type=argparse.FileType("r"),
        default=None,
        required=False,
        help="The input file containing the single prompt to use for profiling.",
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
        "Ensure the --tokenizer value is set correctly. ",
    )

    input_group.add_argument(
        "--output-tokens-mean-deterministic",
        action="store_true",
        required=False,
        help=f"When using --output-tokens-mean, this flag can be set to "
        "improve precision by setting the minimum number of tokens "
        "equal to the requested number of tokens. This is currently "
        "supported with the Triton service-kind. "
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
        "This is only used when --output-tokens-mean is provided.",
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
        help=f"The mean of number of tokens in the generated prompts when using synthetic data.",
    )

    input_group.add_argument(
        "--synthetic-input-tokens-stddev",
        type=int,
        default=LlmInputs.DEFAULT_PROMPT_TOKENS_STDDEV,
        required=False,
        help=f"The standard deviation of number of tokens in the generated prompts when using synthetic data.",
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
        help="The time interval used for each measurement in milliseconds. "
        "Perf Analyzer will sample a time interval specified and take "
        "measurement over the requests completed within that time interval.",
    )

    load_management_group.add_argument(
        "--request-rate",
        type=float,
        required=False,
        help="Sets the request rate for the load generated by PA.",
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
        nargs="+",
        default=[],
        help=f"The name of the model(s) to benchmark.",
    )
    endpoint_group.add_argument(
        "--model-selection-strategy",
        type=str,
        choices=utils.get_enum_names(ModelSelectionStrategy),
        default="round_robin",
        required=False,
        help=f"When multiple model are specified, this is how a specific model "
        "should be assigned to a prompt.  round_robin means that ith prompt in the "
        "list gets assigned to i mod len(models).  random means that assignment is "
        "uniformly random",
    )

    endpoint_group.add_argument(
        "--backend",
        type=str,
        choices=utils.get_enum_names(OutputFormat)[2:],
        default="tensorrtllm",
        required=False,
        help=f'When using the "triton" service-kind, '
        "this is the backend of the model. "
        "For the TENSORRT-LLM backend, you currently must set "
        "'exclude_input_in_output' to true in the model config to "
        "not echo the input tokens in the output.",
    )

    endpoint_group.add_argument(
        "--endpoint",
        type=str,
        required=False,
        help=f"Set a custom endpoint that differs from the OpenAI defaults.",
    )

    endpoint_group.add_argument(
        "--endpoint-type",
        type=str,
        choices=["chat", "completions", "generate", "kserve"],
        default="kserve",
        required=False,
        help=f"The endpoint-type for requests. Inputs will be formatted according to endpoint-type.",
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
        "--generate-plots",
        action="store_true",
        required=False,
        help="An option to enable the generation of plots.",
    )
    output_group.add_argument(
        "--profile-export-file",
        type=Path,
        default=Path("profile_export.json"),
        help="The path where the perf_analyzer profile export will be "
        "generated. By default, the profile export will be to profile_export.json. "
        "The genai-perf file will be exported to <profile_export_file>_genai_perf.csv. "
        "For example, if the profile export file is profile_export.json, the genai-perf file will be "
        "exported to profile_export_genai_perf.csv.",
    )
    output_group.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path(DEFAULT_ARTIFACT_DIR),
        help="The directory to store all the (output) artifacts generated by "
        "GenAI-Perf and Perf Analyzer.",
    )


def _add_other_args(parser):
    other_group = parser.add_argument_group("Other")

    other_group.add_argument(
        "--tokenizer",
        type=str,
        default=DEFAULT_TOKENIZER,
        required=False,
        help="The HuggingFace tokenizer to use to interpret token metrics from prompts and responses.",
    )

    other_group.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        required=False,
        help="An option to enable verbose mode.",
    )

    other_group.add_argument(
        "--version",
        action="version",
        version="%(prog)s " + __version__,
        help=f"An option to print the version and exit.",
    )


def get_extra_inputs_as_dict(args: argparse.Namespace) -> dict:
    request_inputs = {}
    if args.extra_inputs:
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


def _parse_compare_args(subparsers) -> argparse.ArgumentParser:
    compare = subparsers.add_parser(
        "compare",
        description="Subcommand to generate plots that compare multiple profile runs.",
    )
    compare_group = compare.add_argument_group("Compare")
    mx_group = compare_group.add_mutually_exclusive_group(required=False)
    mx_group.add_argument(
        "--config",
        type=Path,
        default=None,
        help="The path to the YAML file that specifies plot configurations for "
        "comparing multiple runs.",
    )
    mx_group.add_argument(
        "-f",
        "--files",
        nargs="+",
        default=[],
        help="List of paths to the profile export JSON files. Users can specify "
        "this option instead of the `--config` option if they would like "
        "GenAI-Perf to generate default plots as well as initial YAML config file.",
    )
    compare.set_defaults(func=compare_handler)
    return compare


### Handlers ###


def create_compare_dir() -> None:
    if not os.path.exists(DEFAULT_COMPARE_DIR):
        os.mkdir(DEFAULT_COMPARE_DIR)


def profile_handler(args, extra_args):
    from genai_perf.wrapper import Profiler

    Profiler.run(args=args, extra_args=extra_args)


def compare_handler(args: argparse.Namespace):
    """Handles `compare` subcommand workflow."""
    if args.files:
        create_compare_dir()
        output_dir = Path(f"{DEFAULT_COMPARE_DIR}")
        PlotConfigParser.create_init_yaml_config(args.files, output_dir)
        args.config = output_dir / "config.yaml"

    config_parser = PlotConfigParser(args.config)
    plot_configs = config_parser.generate_configs()
    plot_manager = PlotManager(plot_configs)
    plot_manager.generate_plots()


### Entrypoint ###


def parse_args():
    argv = sys.argv

    parser = argparse.ArgumentParser(
        prog="genai-perf",
        description="CLI to profile LLMs and Generative AI models with Perf Analyzer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.set_defaults(func=profile_handler)

    # Conceptually group args for easier visualization
    _add_endpoint_args(parser)
    _add_input_args(parser)
    _add_profile_args(parser)
    _add_output_args(parser)
    _add_other_args(parser)

    # Add subcommands
    subparsers = parser.add_subparsers(
        help="List of subparser commands.", dest="subcommand"
    )
    compare_parser = _parse_compare_args(subparsers)

    # Check for passthrough args
    if "--" in argv:
        passthrough_index = argv.index("--")
        logger.info(f"Detected passthrough args: {argv[passthrough_index + 1:]}")
    else:
        passthrough_index = len(argv)

    args = parser.parse_args(argv[1:passthrough_index])
    args = _infer_prompt_source(args)
    args = _check_model_args(parser, args)
    args = _check_conditional_args(parser, args)
    args = _check_compare_args(compare_parser, args)
    args = _check_load_manager_args(args)
    args = _set_artifact_paths(args)

    return args, argv[passthrough_index + 1 :]
