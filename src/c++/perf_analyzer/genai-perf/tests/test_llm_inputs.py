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
from collections import namedtuple
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest
import responses
from genai_perf import tokenizer
from genai_perf.constants import OPEN_ORCA
from genai_perf.exceptions import GenAIPerfException
from genai_perf.llm_inputs.inputs_utils import (
    DEFAULT_LENGTH,
    DEFAULT_STARTING_INDEX,
    ImageFormat,
    ModelSelectionStrategy,
    OutputFormat,
    PromptSource,
)
from genai_perf.llm_inputs.llm_inputs import LlmInputs
from genai_perf.llm_inputs.output_format_converter import BaseConverter

mocked_openorca_data = {
    "features": [
        {"feature_idx": 0, "name": "id", "type": {"dtype": "string", "_type": "Value"}},
        {
            "feature_idx": 1,
            "name": "system_prompt",
            "type": {"dtype": "string", "_type": "Value"},
        },
        {
            "feature_idx": 2,
            "name": "question",
            "type": {"dtype": "string", "_type": "Value"},
        },
        {
            "feature_idx": 3,
            "name": "response",
            "type": {"dtype": "string", "_type": "Value"},
        },
    ],
    "rows": [
        {
            "row_idx": 0,
            "row": {
                "id": "niv.242684",
                "system_prompt": "",
                "question": "You will be given a definition of a task first, then some input of the task.\nThis task is about using the specified sentence and converting the sentence to Resource Description Framework (RDF) triplets of the form (subject, predicate object). The RDF triplets generated must be such that the triplets accurately capture the structure and semantics of the input sentence. The input is a sentence and the output is a list of triplets of the form [subject, predicate, object] that capture the relationships present in the sentence. When a sentence has more than 1 RDF triplet possible, the output must contain all of them.\n\nAFC Ajax (amateurs)'s ground is Sportpark De Toekomst where Ajax Youth Academy also play.\nOutput:",
                "response": '[\n  ["AFC Ajax (amateurs)", "has ground", "Sportpark De Toekomst"],\n  ["Ajax Youth Academy", "plays at", "Sportpark De Toekomst"]\n]',
            },
            "truncated_cells": [],
        }
    ],
    "num_rows_total": 2914896,
    "num_rows_per_page": 100,
    "partial": True,
}

TEST_LENGTH = 1


class TestLlmInputs:
    # Define service kind, backend or api, and output format combinations
    SERVICE_KIND_BACKEND_ENDPOINT_TYPE_FORMATS = [
        ("triton", "vllm", OutputFormat.VLLM),
        ("triton", "tensorrtllm", OutputFormat.TENSORRTLLM),
        ("openai", "v1/completions", OutputFormat.OPENAI_COMPLETIONS),
        ("openai", "v1/chat/completions", OutputFormat.OPENAI_CHAT_COMPLETIONS),
        ("openai", "v1/chat/completions", OutputFormat.OPENAI_VISION),
    ]

    @pytest.fixture
    def default_configured_url(self):
        default_configured_url = (
            LlmInputs.OPEN_ORCA_URL
            + f"&offset={DEFAULT_STARTING_INDEX}&length={DEFAULT_LENGTH}"
        )
        yield default_configured_url

    @pytest.fixture(scope="class")
    def default_tokenizer(self):
        yield tokenizer.get_tokenizer(tokenizer.DEFAULT_TOKENIZER)

    def test_input_type_url_no_dataset_name(self):
        """
        Test for exception when input type is URL and no dataset name
        """
        with pytest.raises(GenAIPerfException):
            _ = LlmInputs.create_llm_inputs(
                input_type=PromptSource.DATASET,
                dataset_name="",
                output_format=OutputFormat.OPENAI_CHAT_COMPLETIONS,
            )

    def test_input_type_synthetic_no_tokenizer(self):
        """
        Test for exception when input type is SYNTHETIC and no tokenizer
        """
        with pytest.raises(GenAIPerfException):
            _ = LlmInputs.create_llm_inputs(
                input_type=PromptSource.SYNTHETIC,
                output_format=OutputFormat.OPENAI_CHAT_COMPLETIONS,
                tokenizer=None,  # type: ignore
            )

    def test_illegal_starting_index(self):
        """
        Test for exceptions when illegal values are given for starting index
        """
        with pytest.raises(GenAIPerfException):
            _ = LlmInputs.create_llm_inputs(
                input_type=PromptSource.DATASET,
                dataset_name=OPEN_ORCA,
                output_format=OutputFormat.OPENAI_CHAT_COMPLETIONS,
                starting_index=-1,
            )

    def test_illegal_length(self):
        """
        Test for exceptions when illegal values are given for length
        """
        with pytest.raises(GenAIPerfException):
            _ = LlmInputs.create_llm_inputs(
                input_type=PromptSource.DATASET,
                dataset_name=OPEN_ORCA,
                output_format=OutputFormat.OPENAI_CHAT_COMPLETIONS,
                length=0,
            )

    @responses.activate
    def test_llm_inputs_with_defaults(self, default_configured_url):
        """
        Test that default options work
        """
        responses.add(
            responses.GET,
            f"{default_configured_url}",
            json=mocked_openorca_data,
            status=200,
        )

        pa_json = LlmInputs.create_llm_inputs(
            input_type=PromptSource.DATASET,
            dataset_name=OPEN_ORCA,
            output_format=OutputFormat.OPENAI_CHAT_COMPLETIONS,
            model_name=["test_model_A"],
        )

        assert pa_json is not None
        assert len(pa_json["data"]) == TEST_LENGTH

    def test_get_input_file_without_file_existing(self):
        with pytest.raises(FileNotFoundError):
            LlmInputs.create_llm_inputs(
                input_type=PromptSource.FILE,
                input_filename=Path("prompt.txt"),
                output_format=OutputFormat.OPENAI_CHAT_COMPLETIONS,
            )

    @patch("pathlib.Path.exists", return_value=True)
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{"text_input": "single prompt"}\n',
    )
    def test_get_input_file_with_single_prompt(self, mock_file, mock_exists):
        expected_prompts = ["single prompt"]
        pa_json = LlmInputs.create_llm_inputs(
            input_type=PromptSource.FILE,
            input_filename=Path("prompt.txt"),
            output_format=OutputFormat.OPENAI_CHAT_COMPLETIONS,
            model_name=["test_model"],
        )

        assert pa_json is not None
        assert len(pa_json["data"]) == len(expected_prompts)
        for i, prompt in enumerate(expected_prompts):
            assert pa_json["data"][i]["payload"][0]["messages"][0]["content"] == prompt

    @patch("pathlib.Path.exists", return_value=True)
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{"text_input": "prompt1"}\n{"text_input": "prompt2"}\n{"text_input": "prompt3"}\n',
    )
    def test_get_input_file_with_multiple_prompts(self, mock_file, mock_exists):
        expected_prompts = ["prompt1", "prompt2", "prompt3"]
        pa_json = LlmInputs.create_llm_inputs(
            input_type=PromptSource.FILE,
            input_filename=Path("prompt.txt"),
            output_format=OutputFormat.OPENAI_CHAT_COMPLETIONS,
            model_name=["test_model"],
        )

        assert pa_json is not None
        assert len(pa_json["data"]) == len(expected_prompts)
        for i, prompt in enumerate(expected_prompts):
            assert pa_json["data"][i]["payload"][0]["messages"][0]["content"] == prompt

    @patch("pathlib.Path.exists", return_value=True)
    @patch(
        "genai_perf.llm_inputs.dataset_retriever.DatasetRetriever._encode_image_to_base64",
        return_value="data:image/png;base64,...",
    )
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data=(
            '{"text_input": "prompt1", "image": "image1.png"}\n'
            '{"text_input": "prompt2", "image": "image2.png"}\n'
            '{"text_input": "prompt3", "image": "image3.png"}\n'
        ),
    )
    def test_get_input_file_with_multi_modal_data(
        self, mock_exists, mock_image, mock_file
    ):
        Data = namedtuple("Data", ["text_input", "image"])
        expected_data = [
            Data(text_input="prompt1", image="data:image/png;base64,..."),
            Data(text_input="prompt2", image="data:image/png;base64,..."),
            Data(text_input="prompt3", image="data:image/png;base64,..."),
        ]

        pa_json = LlmInputs.create_llm_inputs(
            input_type=PromptSource.FILE,
            input_filename=Path("somefile.txt"),
            output_format=OutputFormat.OPENAI_VISION,
            model_name=["test_model"],
            image_format=ImageFormat.PNG,
        )

        assert pa_json is not None
        assert len(pa_json["data"]) == len(expected_data)
        for i, data in enumerate(expected_data):
            content = pa_json["data"][i]["payload"][0]["messages"][0]["content"]
            assert content[0]["type"] == "text"
            assert content[0]["text"] == data.text_input
            assert content[1]["type"] == "image_url"
            assert content[1]["image_url"] == {"url": data.image}

    @pytest.mark.parametrize(
        "input_type, output_format",
        [
            (PromptSource.DATASET, OutputFormat.OPENAI_EMBEDDINGS),
            (PromptSource.DATASET, OutputFormat.VLLM),
            (PromptSource.DATASET, OutputFormat.RANKINGS),
            (PromptSource.DATASET, OutputFormat.OPENAI_VISION),
            (PromptSource.SYNTHETIC, OutputFormat.OPENAI_EMBEDDINGS),
            (PromptSource.SYNTHETIC, OutputFormat.VLLM),
            (PromptSource.SYNTHETIC, OutputFormat.RANKINGS),
        ],
    )
    def test_unsupported_combinations(self, input_type, output_format):
        """
        Test that unsupported combinations of input types and output formats raise exceptions
        """
        with pytest.raises(GenAIPerfException):
            _ = LlmInputs.create_llm_inputs(
                input_type=input_type,
                output_format=output_format,
                dataset_name=OPEN_ORCA,
            )

    @pytest.mark.parametrize(
        "seed, model_name_list, index,model_selection_strategy,expected_model",
        [
            (
                1,
                ["test_model_A", "test_model_B", "test_model_C"],
                0,
                ModelSelectionStrategy.ROUND_ROBIN,
                "test_model_A",
            ),
            (
                1,
                ["test_model_A", "test_model_B", "test_model_C"],
                1,
                ModelSelectionStrategy.ROUND_ROBIN,
                "test_model_B",
            ),
            (
                1,
                ["test_model_A", "test_model_B", "test_model_C"],
                2,
                ModelSelectionStrategy.ROUND_ROBIN,
                "test_model_C",
            ),
            (
                1,
                ["test_model_A", "test_model_B", "test_model_C"],
                3,
                ModelSelectionStrategy.ROUND_ROBIN,
                "test_model_A",
            ),
            (
                100,
                ["test_model_A", "test_model_B", "test_model_C"],
                0,
                ModelSelectionStrategy.RANDOM,
                "test_model_A",
            ),
            (
                100,
                ["test_model_A", "test_model_B", "test_model_C"],
                1,
                ModelSelectionStrategy.RANDOM,
                "test_model_A",
            ),
            (
                1652,
                ["test_model_A", "test_model_B", "test_model_C"],
                0,
                ModelSelectionStrategy.RANDOM,
                "test_model_B",
            ),
            (
                95,
                ["test_model_A", "test_model_B", "test_model_C"],
                0,
                ModelSelectionStrategy.RANDOM,
                "test_model_C",
            ),
        ],
    )
    def test_select_model_name(
        self, seed, model_name_list, index, model_selection_strategy, expected_model
    ):
        """
        Test that model selection strategy controls the model selected
        """
        random.seed(seed)

        actual_model = BaseConverter._select_model_name(
            model_name_list, index, model_selection_strategy
        )
        assert actual_model == expected_model


# TODO (TMA-1754): Add tests that verify json schemas
# TODO (TPA-114) Refactor LLM inputs and testing to include dataset path testing
