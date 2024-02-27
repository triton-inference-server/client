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


def add_tokenization_args(parser):
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="auto",
        choices=["auto"],
        required=False,
        help="The huggingface tokenizer to use to interpret token metrics from text results",
    )


# Optional argv used for testing - will default to sys.argv if None.
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="genaipa",
        description="CLI to profile LLMs and Generative AI models with PA",
    )
    # Conceptually group args for easier visualization
    model_group = bench_run.add_argument_group("Model")
    add_model_args(model_group)

    profile_group = parser.add_argument_group("Profiling")
    add_profile_args(profile_group)

    token_group = bench_run.add_argument_group("Tokenization")
    add_tokenization_args(token_group)

    args = parser.parse_args(argv)
    return args
