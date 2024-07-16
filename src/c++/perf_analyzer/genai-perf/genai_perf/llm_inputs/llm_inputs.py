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

import random
from pathlib import Path
from typing import Dict, Optional, cast

from genai_perf.constants import CNN_DAILY_MAIL, DEFAULT_INPUT_DATA_JSON, OPEN_ORCA
from genai_perf.exceptions import GenAIPerfException
from genai_perf.llm_inputs.dataset_retriever import DatasetRetriever
from genai_perf.llm_inputs.inputs_utils import (
    DEFAULT_LENGTH,
    DEFAULT_NUM_PROMPTS,
    DEFAULT_OUTPUT_TOKENS_MEAN,
    DEFAULT_OUTPUT_TOKENS_STDDEV,
    DEFAULT_PROMPT_TOKENS_MEAN,
    DEFAULT_PROMPT_TOKENS_STDDEV,
    DEFAULT_RANDOM_SEED,
    DEFAULT_STARTING_INDEX,
    ModelSelectionStrategy,
    OutputFormat,
    PromptSource,
)
from genai_perf.llm_inputs.json_converter import JSONConverter
from genai_perf.llm_inputs.output_format_converter import OutputFormatConverterFactory
from genai_perf.tokenizer import DEFAULT_TOKENIZER, Tokenizer, get_tokenizer
from genai_perf.utils import write_to_json_file


class LlmInputs:
    """
    This class is the responsible for generating inputs.
    """

    OPEN_ORCA_URL = "https://datasets-server.huggingface.co/rows?dataset=Open-Orca%2FOpenOrca&config=default&split=train"
    CNN_DAILYMAIL_URL = "https://datasets-server.huggingface.co/rows?dataset=cnn_dailymail&config=1.0.0&split=train"

    dataset_url_map = {OPEN_ORCA: OPEN_ORCA_URL, CNN_DAILY_MAIL: CNN_DAILYMAIL_URL}

    @classmethod
    def create_llm_inputs(
        cls,
        input_type: PromptSource,
        output_format: OutputFormat,
        dataset_name: str = "",
        model_name: list = [],
        model_selection_strategy: ModelSelectionStrategy = ModelSelectionStrategy.ROUND_ROBIN,
        input_filename: Optional[Path] = Path(""),
        starting_index: int = DEFAULT_STARTING_INDEX,
        length: int = DEFAULT_LENGTH,
        output_tokens_mean: int = DEFAULT_OUTPUT_TOKENS_MEAN,
        output_tokens_stddev: int = DEFAULT_OUTPUT_TOKENS_STDDEV,
        output_tokens_deterministic: bool = False,
        prompt_tokens_mean: int = DEFAULT_PROMPT_TOKENS_MEAN,
        prompt_tokens_stddev: int = DEFAULT_PROMPT_TOKENS_STDDEV,
        random_seed: int = DEFAULT_RANDOM_SEED,
        num_of_output_prompts: int = DEFAULT_NUM_PROMPTS,
        add_model_name: bool = False,
        add_stream: bool = False,
        tokenizer: Tokenizer = get_tokenizer(DEFAULT_TOKENIZER),
        extra_inputs: Optional[Dict] = None,
        batch_size: int = 1,
        output_dir: Path = Path(""),
    ) -> Dict:
        cls.validate_args(
            input_type, output_format, dataset_name, starting_index, length, tokenizer
        )

        random.seed(random_seed)

        if input_type == PromptSource.DATASET:
            dataset = DatasetRetriever.from_url(
                cls.dataset_url_map[dataset_name], starting_index, length
            )
        elif input_type == PromptSource.SYNTHETIC:
            dataset = DatasetRetriever.from_synthetic(
                tokenizer,
                prompt_tokens_mean,
                prompt_tokens_stddev,
                num_of_output_prompts,
            )
        elif input_type == PromptSource.FILE:
            input_filename = cast(Path, input_filename)
            # TODO: Follow-up ticket to add support for rankings
            # if output_format == OutputFormat.RANKINGS:
            #     dataset = DatasetRetriever.from_directory(input_filename)
            # else:
            dataset = DatasetRetriever.from_file(input_filename)
        else:
            raise GenAIPerfException("Input source is not recognized.")

        generic_dataset_json = JSONConverter.to_generic(dataset)

        if extra_inputs is None:
            extra_inputs = {}

        json_in_pa_format = cls.convert_to_output_format(
            output_format,
            generic_dataset_json,
            add_model_name,
            add_stream,
            extra_inputs,
            output_tokens_mean,
            output_tokens_stddev,
            output_tokens_deterministic,
            model_name,
            model_selection_strategy,
        )

        write_to_json_file(json_in_pa_format, (output_dir / DEFAULT_INPUT_DATA_JSON))
        return json_in_pa_format

    @staticmethod
    def validate_args(
        input_type: PromptSource,
        output_format: OutputFormat,
        dataset_name: str,
        starting_index: int,
        length: int,
        tokenizer: Tokenizer,
    ) -> None:
        unsupported_combinations = {
            OutputFormat.OPENAI_EMBEDDINGS: [
                PromptSource.DATASET,
            ],
            OutputFormat.RANKINGS: [PromptSource.DATASET, PromptSource.SYNTHETIC],
        }

        if input_type in unsupported_combinations.get(output_format, []):
            raise GenAIPerfException(
                f"{output_format.to_lowercase()} does not support input type `{input_type.to_lowercase()}`"
            )

        if input_type == PromptSource.DATASET and not dataset_name:
            raise GenAIPerfException(
                "Input type is dataset, but dataset_name is not specified."
            )
        if input_type == PromptSource.SYNTHETIC and not tokenizer:
            raise GenAIPerfException(
                "Input type is SYNTHETIC, but a tokenizer was not specified."
            )
        if starting_index < 0:
            raise GenAIPerfException(
                f"starting_index: {starting_index} must be non-negative."
            )
        if length < 1:
            raise GenAIPerfException(f"length: {length} must be positive.")

    @classmethod
    def convert_to_output_format(
        cls,
        output_format: OutputFormat,
        generic_dataset: Dict,
        add_model_name: bool,
        add_stream: bool,
        extra_inputs: Dict,
        output_tokens_mean: int,
        output_tokens_stddev: int,
        output_tokens_deterministic: bool,
        model_name: list,
        model_selection_strategy: ModelSelectionStrategy,
    ) -> Dict:
        converter = OutputFormatConverterFactory.create(output_format)
        return converter.convert(
            generic_dataset,
            add_model_name,
            add_stream,
            extra_inputs,
            output_tokens_mean,
            output_tokens_stddev,
            output_tokens_deterministic,
            model_name,
            model_selection_strategy,
        )
