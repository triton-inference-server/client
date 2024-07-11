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
from typing import Dict, List

from genai_perf.exceptions import GenAIPerfException
from genai_perf.llm_inputs.shared import (
    DEFAULT_TENSORRTLLM_MAX_TOKENS,
    ModelSelectionStrategy,
    OutputFormat,
)


class OutputFormatConverterFactory:
    @staticmethod
    def create(output_format: OutputFormat):
        converters = {
            OutputFormat.OPENAI_CHAT_COMPLETIONS: OpenAIChatCompletionsConverter,
            OutputFormat.OPENAI_COMPLETIONS: OpenAICompletionsConverter,
            OutputFormat.OPENAI_EMBEDDINGS: OpenAIEmbeddingsConverter,
            OutputFormat.RANKINGS: RankingsConverter,
            OutputFormat.VLLM: VLLMConverter,
            OutputFormat.TENSORRTLLM: TensorRTLLMConverter,
        }
        if output_format not in converters:
            raise GenAIPerfException(f"Output format {output_format} is not supported")
        return converters[output_format]()


class BaseConverter:
    def convert(
        self,
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
        raise NotImplementedError

    @staticmethod
    def _select_model_name(
        model_name: List[str], index: int, strategy: ModelSelectionStrategy
    ) -> str:
        if not model_name:
            raise GenAIPerfException("Model name list cannot be empty.")

        if strategy == ModelSelectionStrategy.ROUND_ROBIN:
            return model_name[index % len(model_name)]
        elif strategy == ModelSelectionStrategy.RANDOM:
            return random.choice(model_name)
        else:
            raise GenAIPerfException(
                f"Model selection strategy '{strategy}' is unsupported"
            )


class OpenAIChatCompletionsConverter(BaseConverter):
    def convert(
        self,
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
        pa_json: Dict = {"data": []}

        for index, row in enumerate(generic_dataset["rows"]):
            model = self._select_model_name(model_name, index, model_selection_strategy)
            text_content = row["row"]["text_input"]
            messages = [{"role": "user", "content": text_content}]
            payload: Dict = {"messages": messages}

            if add_model_name:
                payload["model"] = model
            if add_stream:
                payload["stream"] = True
            if output_tokens_mean != -1:
                payload["max_tokens"] = int(
                    random.gauss(output_tokens_mean, output_tokens_stddev)
                )
            payload.update(extra_inputs)

            pa_json["data"].append({"payload": [payload]})

        return pa_json


class OpenAICompletionsConverter(BaseConverter):
    def convert(
        self,
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
        pa_json: Dict = {"data": []}

        for index, row in enumerate(generic_dataset["rows"]):
            text_content = row["row"]["text_input"]
            model = self._select_model_name(model_name, index, model_selection_strategy)
            payload = {"prompt": text_content}

            if add_model_name:
                payload["model"] = model
            if add_stream:
                payload["stream"] = True
            if output_tokens_mean != -1:
                payload["max_tokens"] = int(
                    random.gauss(output_tokens_mean, output_tokens_stddev)
                )
            payload.update(extra_inputs)

            pa_json["data"].append({"payload": [payload]})

        return pa_json


class OpenAIEmbeddingsConverter(BaseConverter):
    def convert(
        self,
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
        pa_json: Dict = {"data": []}

        for index, row in enumerate(generic_dataset["rows"]):
            text_content = row["row"]["text_input"]
            model = self._select_model_name(model_name, index, model_selection_strategy)
            payload: Dict = {"input": [text_content], "model": model}

            if add_stream:
                payload["stream"] = True
            payload.update(extra_inputs)

            pa_json["data"].append({"payload": [payload]})

        return pa_json


class RankingsConverter(BaseConverter):
    def convert(
        self,
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
        pa_json: Dict = {"data": []}

        for index, row in enumerate(generic_dataset["rows"]):
            if "query" not in row or "passages" not in row:
                raise GenAIPerfException(
                    "Expected keys 'query' and 'passages' not found in dataset row."
                )

            model = self._select_model_name(model_name, index, model_selection_strategy)
            payload = {
                "query": row["query"]["text_input"],
                "passages": [p["text_input"] for p in row["passages"]],
                "model": model,
            }

            if add_stream:
                payload["stream"] = True
            payload.update(extra_inputs)

            pa_json["data"].append({"payload": [payload]})

        return pa_json


class VLLMConverter(BaseConverter):
    def convert(
        self,
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
        data = []
        for index, row in enumerate(generic_dataset["rows"]):
            model = self._select_model_name(model_name, index, model_selection_strategy)
            text_input = row["row"]["text_input"]

            data.append(
                {
                    "text_input": [text_input],
                    "model": model,
                    "exclude_input_in_output": [True],
                    **extra_inputs,
                }
            )
        return {"data": data}


class TensorRTLLMConverter(BaseConverter):
    def convert(
        self,
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
        pa_json: Dict = {"data": []}

        for index, row in enumerate(generic_dataset["rows"]):
            text_content = row["row"]["text_input"]
            model = self._select_model_name(model_name, index, model_selection_strategy)
            payload: Dict = {
                "text_input": [text_content],
                "model": model,
                "max_tokens": [DEFAULT_TENSORRTLLM_MAX_TOKENS],
            }

            if add_stream:
                payload["stream"] = True
            if output_tokens_mean != -1:
                max_tokens = int(random.gauss(output_tokens_mean, output_tokens_stddev))
                payload["max_tokens"] = [max_tokens]
                if output_tokens_deterministic:
                    payload["min_length"] = [max_tokens]
            payload.update(extra_inputs)

            pa_json["data"].append(payload)

        return pa_json
