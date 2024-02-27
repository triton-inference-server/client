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

import argparse
import logging

from genai_pa.constants import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)

### Handlers ###


# NOTE: Placeholder
def handler(args):
    from genai_pa.wrapper import Profiler

    Profiler.run(model=args.model, args=args)


### Parsers ###


def add_model_args(parser):
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help=f"The name of the model to benchmark.",
    )


def add_profile_args(parser):
    parser.add_argument(
        "--async",
        action="store_true",
        required=False,
        help=f"Enables asynchronous mode in perf_analyzer. "
        "By default, perf_analyzer will use synchronous API to "
        "request inference. However, if the model is sequential, "
        "then default mode is asynchronous. Specify --sync to "
        "operate sequential models in synchronous mode. In synchronous "
        "mode, perf_analyzer will start threads equal to the concurrency "
        "level. Use asynchronous mode to limit the number of threads, yet "
        "maintain the concurrency.",
    )
    parser.add_argument(
        "-b",
        type=int,
        default=1,
        required=False,
        help="The batch size to benchmark. The default value is 1.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        required=False,
        help="Sets the concurrency value to benchmark.",
    )
    # TODO: Do we need input length
    # parser.add_argument(
    #     "--input-length",
    #     type=int,
    #     default=128,
    #     required=False,
    #     help="The input length (tokens) to use for benchmarking LLMs. (Default: 128)",
    # )
    parser.add_argument(
        "--max-threads",
        type=int,
        default=16,
        required=False,
        help="Sets the maximum number of threads that will be "
        "created for providing desired concurrency or request rate. "
        "However, when running in synchronous mode,this value will be ignored. "
        "The default value is 16.",
    )
    parser.add_argument(
        "--output-length",
        type=int,
        default=128,
        required=False,
        help="The output length (tokens) to use for benchmarking LLMs. (Default: 128)",
    )
    parser.add_argument(
        "--profile-export-file",
        type=str,
        required=False,
        help="Specifies the path that the profile export will be "
        "generated at. By default, the profile export will not be "
        "generated.",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        required=False,
        help="Sets the request rates for load generated by analyzer. ",
    )
    parser.add_argument(
        "--service-kind",
        type=str,
        choices=["triton", "openai"],
        default="triton",
        required=False,
        help="Sets the request rates for load generated by analyzer. "
        "Describes the kind of service perf_analyzer to "
        'generate load for. The options are "triton" and '
        '"openai". The default value is "triton".',
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        required=False,
        help=f"Enables the use of streaming API. This flag is "
        "only valid with gRPC protocol. By default, it is set false.",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        required=False,
        help=f"Enables asynchronous mode in perf_analyzer. "
        "By default, perf_analyzer will use synchronous API to "
        "request inference. However, if the model is sequential, "
        "then default mode is asynchronous. Specify --sync to "
        "operate sequential models in synchronous mode. In synchronous "
        "mode, perf_analyzer will start threads equal to the concurrency "
        "level. Use asynchronous mode to limit the number of threads, yet "
        "maintain the concurrency.",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        required=False,
        help=f"Enables the printing of the current version of perf_analyzer. "
        "By default, it is set false.",
    )


def add_endpoint_args(parser):
    parser.add_argument(
        "--u",
        type=str,
        default="localhost:8001",
        required=False,
        help="URL of the endpoint to target for benchmarking.",
    )


def add_dataset_args(parser):
    pass
    # TODO: Do we want to remove dataset and tokenizer?
    # parser.add_argument(
    #     "--dataset",
    #     type=str,
    #     default="OpenOrca",
    #     choices=["OpenOrca", "cnn_dailymail"],
    #     required=False,
    #     help="HuggingFace dataset to use for the benchmark.",
    # )
    # parser.add_argument(
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
        prog="genai-pa",
        description="CLI to profile LLMs and Generative AI models with Perf Analyzer",
    )
    parser.set_defaults(func=handler)

    # Conceptually group args for easier visualization
    model_group = parser.add_argument_group("Model")
    add_model_args(model_group)

    profile_group = parser.add_argument_group("Profiling")
    add_profile_args(profile_group)

    endpoint_group = parser.add_argument_group("Endpoint")
    add_endpoint_args(endpoint_group)

    dataset_group = parser.add_argument_group("Dataset")
    add_dataset_args(dataset_group)

    args = parser.parse_args(argv)

    # Concurrency and request rate are mutually exclusive
    # TODO: Review if there is a cleaner way to do this with argparse
    if args.concurrency is not None and args.request_rate is not None:
        parser.error(
            "Arguments --concurrency and --request_rate are mutually exclusive."
        )

    if args.concurrency is None and args.request_rate is None:
        args.concurrency = 1
        print(
            "Neither --concurrency nor --request_rate provided. Setting concurrency to 1."
        )

    # Update GenAI-PA non-range attributes to range format for PA
    for attr_key in ["concurrency", "request_rate"]:
        attr_val = getattr(args, attr_key)
        if attr_val is not None:
            setattr(args, f"{attr_key}_range", f"{attr_val}:{attr_val}:{attr_val}")
        delattr(args, attr_key)

    args = argparse.Namespace(**{k: v for k, v in vars(args).items() if v is not None})
    return args
