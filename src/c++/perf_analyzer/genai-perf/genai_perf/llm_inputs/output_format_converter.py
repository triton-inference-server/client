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
from typing import Dict, List

from genai_perf.exceptions import GenAIPerfException
from genai_perf.llm_inputs.inputs_utils import (
    DEFAULT_TENSORRTLLM_MAX_TOKENS,
    ModelSelectionStrategy,
    OutputFormat,
)


class OutputFormatConverterFactory:
    """
    This class converts the generic JSON to the specific format
    used by a given endpoint.
    """

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
