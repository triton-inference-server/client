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
from pathlib import Path

import genai_perf.utils as utils
from genai_perf.constants import (
    CNN_DAILY_MAIL,
    DEFAULT_HTTP_URL,
    DEFAULT_INPUT_DATA_JSON,
    LOGGER_NAME,
    OPEN_ORCA,
)
from genai_perf.llm_inputs.llm_inputs import InputType, OutputFormat

logger = logging.getLogger(LOGGER_NAME)


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

    Profiler.run(model=args.model, args=args, extra_args=extra_args)


### Parsers ###


def _add_model_args(parser):
    model_group = parser.add_argument_group("Model")

    model_group.add_argument(
        "--input-type",
        type=str,
        choices=utils.get_enum_names(InputType),
        default="url",
        required=False,
        help=f"The source of the input data.",
    )

    model_group.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help=f"The name of the model to benchmark.",
    )

    model_group.add_argument(
        "--output-format",
        type=str,
        choices=utils.get_enum_names(OutputFormat),
        default="trtllm",
        required=False,
        help=f"The format of the data sent to triton.",
    )


def _add_profile_args(parser):
    profile_group = parser.add_argument_group("Profiling")
    load_management_group = profile_group.add_mutually_exclusive_group(required=True)

    profile_group.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=1,
        required=False,
        help="The batch size to benchmark. The default value is 1.",
    )
    load_management_group.add_argument(
        "--concurrency",
        type=int,
        required=False,
        help="Sets the concurrency value to benchmark.",
    )

    profile_group.add_argument(
        "--input-data",
        type=Path,
        default=DEFAULT_INPUT_DATA_JSON,
        required=False,
        help="Path to the input data json file that contains the list of requests.",
    )

    profile_group.add_argument(
        "-p",
        "--measurement-interval",
        type=int,
        default="10000",
        required=False,
        help="Indicates the time interval used "
        "for each measurement in milliseconds. The perf analyzer will "
        "sample a time interval specified by -p and take measurement over "
        "the requests completed within that time interval. The default "
        "value is 5000 msec.",
    )

    profile_group.add_argument(
        "--profile-export-file",
        type=Path,
        default="profile_export.json",
        help="Specifies the path where the perf_analyzer profile export will be "
        "generated. By default, the profile export will be to profile_export.json. "
        "The genai-perf file will be exported to <profile_export_file>_genai_perf.csv. "
        "For example, if the profile export file is profile_export.json, the genai-perf file will be "
        "exported to profile_export_genai_perf.csv.",
    )

    load_management_group.add_argument(
        "--request-rate",
        type=float,
        required=False,
        help="Sets the request rate for the load generated by PA. ",
    )

    profile_group.add_argument(
        "--service-kind",
        type=str,
        choices=["triton", "openai"],
        default="triton",
        required=False,
        help="Describes the kind of service perf_analyzer will "
        'generate load for. The options are "triton" and '
        '"openai". The default value is "triton".',
    )

    profile_group.add_argument(
        "-s",
        "--stability-percentage",
        type=int,
        default=999,
        required=False,
        help="Indicates the allowed variation in "
        "latency measurements when determining if a result is stable. The "
        "measurement is considered as stable if the ratio of max / min "
        "from the recent 3 measurements is within (stability percentage) "
        "in terms of both infer per second and latency.",
    )

    profile_group.add_argument(
        "--streaming",
        action="store_true",
        required=False,
        help=f"Enables the use of the streaming API.",
    )

    profile_group.add_argument(
        "-v", action="store_true", required=False, help="Enables verbose mode."
    )

    profile_group.add_argument(
        "--version",
        action="store_true",
        required=False,
        help=f"Prints the version and exits.",
    )


def _add_endpoint_args(parser):
    endpoint_group = parser.add_argument_group("Endpoint")

    endpoint_group.add_argument(
        "--endpoint", type=str, required=False, help="Specify an endpoint."
    )

    endpoint_group.add_argument(
        "-u",
        "--url",
        type=str,
        default=DEFAULT_HTTP_URL,
        required=False,
        dest="u",
        metavar="URL",
        help="URL of the endpoint to target for benchmarking.",
    )


def _add_dataset_args(parser):
    dataset_group = parser.add_argument_group("Dataset")

    dataset_group.add_argument(
        "--dataset",
        type=str.lower,
        default=OPEN_ORCA,
        choices=[OPEN_ORCA, CNN_DAILY_MAIL],
        required=False,
        help="HuggingFace dataset to use for benchmarking.",
    )

    # dataset_group.add_argument(
    #     "--tokenizer",
    #     type=str,
    #     default="auto",
    #     choices=["auto"],
    #     required=False,
    #     help="The HuggingFace tokenizer to use to interpret token metrics from final text results",
    # )


### Entrypoint ###


# Optional argv used for testing - will default to sys.argv if None.
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="genai-perf",
        description="CLI to profile LLMs and Generative AI models with Perf Analyzer",
    )
    parser.set_defaults(func=handler)

    # Conceptually group args for easier visualization
    _add_model_args(parser)
    _add_profile_args(parser)
    _add_endpoint_args(parser)
    _add_dataset_args(parser)

    args, extra_args = parser.parse_known_args(argv)
    if extra_args:
        # strip off the "--" demarking the pass through arguments
        extra_args = extra_args[1:]
        logger.info(f"Additional pass through args: {extra_args}")

    args = _update_load_manager_args(args)
    args = _convert_str_to_enum_entry(args, "input_type", InputType)
    args = _convert_str_to_enum_entry(args, "output_format", OutputFormat)
    args = _prune_args(args)

    return args, extra_args
