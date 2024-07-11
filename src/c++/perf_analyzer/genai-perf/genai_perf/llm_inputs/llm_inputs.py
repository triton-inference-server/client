# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
from pathlib import Path
from typing import Dict, Optional, cast

from genai_perf.constants import CNN_DAILY_MAIL, OPEN_ORCA
from genai_perf.exceptions import GenAIPerfException
from genai_perf.llm_inputs.dataset_retriever import DatasetRetriever
from genai_perf.llm_inputs.json_converter import JSONConverter
from genai_perf.llm_inputs.json_writer import JSONWriter
from genai_perf.llm_inputs.output_format_converter import OutputFormatConverterFactory
from genai_perf.llm_inputs.shared import (
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
from genai_perf.tokenizer import DEFAULT_TOKENIZER, Tokenizer, get_tokenizer


class LlmInputs:
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
            input_filename = cast(Path, input_filename)
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

        JSONWriter.write_to_file(json_in_pa_format, output_dir)
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
