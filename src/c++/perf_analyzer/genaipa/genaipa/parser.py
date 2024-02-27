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

from genaipa.constants import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)

### Handlers ###


def handle_profile(args):
    from genaipa.profiler import Profiler

    # TODO: "backend" arg may not translate well to final product
    Profiler.profile(
        model=args.model,
        backend="vllm",
        batch_size=args.batch_size,
        url=args.url,
        input_length=args.input_length,
        output_length=args.output_length,
        offline=False,
        verbose=False,
    )


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
        "-b",
        "--batch-size",
        type=int,
        default=1,
        required=False,
        help="The batch size / concurrency to benchmark. (Default: 1)",
    )
    parser.add_argument(
        "--input-length",
        type=int,
        default=128,
        required=False,
        help="The input length (tokens) to use for benchmarking LLMs. (Default: 128)",
    )
    parser.add_argument(
        "--output-length",
        type=int,
        default=128,
        required=False,
        help="The output length (tokens) to use for benchmarking LLMs. (Default: 128)",
    )


def add_endpoint_args(parser):
    parser.add_argument(
        "--url",
        type=str,
        default="localhost:8001",
        required=False,
        help="URL of the endpoint to target for benchmarking.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["triton", "openai"],
        required=False,
        help="Provider format/schema to use for benchmarking.",
    )


def add_dataset_args(parser):
    parser.add_argument(
        "--dataset",
        type=str,
        default="OpenOrca",
        choices=["OpenOrca", "cnn_dailymail"],
        required=False,
        help="HuggingFace dataset to use for the benchmark.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="auto",
        choices=["auto"],
        required=False,
        help="The HuggingFace tokenizer to use to interpret token metrics from final text results",
    )


### Entrypoint ###


# Optional argv used for testing - will default to sys.argv if None.
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="genaipa",
        description="CLI to profile LLMs and Generative AI models with PA",
    )
    # TODO: Restructure as needed based on desired user interface
    parser.set_defaults(func=handle_profile)

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
    return args
