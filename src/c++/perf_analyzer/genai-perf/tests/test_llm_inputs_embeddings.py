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

import json
import random
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest
from genai_perf import tokenizer
from genai_perf.llm_inputs.llm_inputs import LlmInputs, OutputFormat, PromptSource


class TestLlmInputsEmbeddings:
    @pytest.fixture(scope="class")
    def default_tokenizer(self):
        yield tokenizer.get_tokenizer(tokenizer.DEFAULT_TOKENIZER)

    @patch("pathlib.Path.exists", return_value=True)
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data=json.dumps(
            [
                {
                    "question": "example question",
                    "pos_doc": ["example of a relevant passage"],
                    "neg_doc": [
                        "example of an irrelevant passage",
                        "another example of an irrelevant passage",
                    ],
                },
                {
                    "question": "example question_2",
                    "pos_doc": ["example of a relevant passage_2"],
                    "neg_doc": [
                        "example of an irrelevant passage_2",
                        "another example of an irrelevant passage_2",
                    ],
                },
            ]
        ),
    )
    def test_get_input_embeddings_file_with_multiple_entries_query(
        self, mock_file, mock_exists
    ):
        dataset = LlmInputs._get_input_dataset_from_embeddings_file(
            Path("embeddings.json"), "query"
        )

        expected_rows = [
            {"input": "example question", "input_type": "query"},
            {"input": "example question_2", "input_type": "query"},
        ]

        assert dataset is not None
        assert len(dataset["rows"]) == len(expected_rows)
        for i, row in enumerate(expected_rows):
            assert dataset["rows"][i]["row"] == row

    @patch("pathlib.Path.exists", return_value=True)
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data=json.dumps(
            [
                {
                    "question": "example question",
                    "pos_doc": ["example of a relevant passage"],
                    "neg_doc": [
                        "example of an irrelevant passage",
                        "another example of an irrelevant passage",
                    ],
                },
                {
                    "question": "example question_2",
                    "pos_doc": ["example of a relevant passage_2"],
                    "neg_doc": [
                        "example of an irrelevant passage_2",
                        "another example of an irrelevant passage_2",
                    ],
                },
            ]
        ),
    )
    def test_get_input_embeddings_file_with_multiple_entries_passage(
        self, mock_file, mock_exists
    ):
        dataset = LlmInputs._get_input_dataset_from_embeddings_file(
            Path("embeddings.json"), "passage"
        )

        expected_rows = [
            {"input": "example of a relevant passage", "input_type": "passage"},
            {"input": "example of an irrelevant passage", "input_type": "passage"},
            {
                "input": "another example of an irrelevant passage",
                "input_type": "passage",
            },
            {"input": "example of a relevant passage_2", "input_type": "passage"},
            {"input": "example of an irrelevant passage_2", "input_type": "passage"},
            {
                "input": "another example of an irrelevant passage_2",
                "input_type": "passage",
            },
        ]

        assert dataset is not None
        assert len(dataset["rows"]) == len(expected_rows)
        for i, row in enumerate(expected_rows):
            assert dataset["rows"][i]["row"] == row

    def test_generate_synthetic_prompts_for_openai_embeddings_query(
        self, default_tokenizer
    ):
        dataset = LlmInputs._generate_synthetic_prompts_for_openai_embeddings(
            default_tokenizer,
            prompts_mean=10,
            prompts_stddev=2,
            num_of_output_prompts=5,
            embeddings_input_type="query",
            embeddings_prompts_mean=3,
            embeddings_prompts_stddev=1,
        )

        assert dataset is not None
        assert len(dataset["rows"]) == 5
        for row in dataset["rows"]:
            assert "input" in row["row"]
            assert "input_type" in row["row"]
            assert row["row"]["input_type"] == "query"

    def test_generate_synthetic_prompts_for_openai_embeddings_passage(
        self, default_tokenizer
    ):
        dataset = LlmInputs._generate_synthetic_prompts_for_openai_embeddings(
            default_tokenizer,
            prompts_mean=10,
            prompts_stddev=2,
            num_of_output_prompts=5,
            embeddings_input_type="passage",
            embeddings_prompts_mean=3,
            embeddings_prompts_stddev=1,
        )

        assert dataset is not None
        assert len(dataset["rows"]) == 5
        for row in dataset["rows"]:
            assert "input" in row["row"]
            assert "input_type" in row["row"]
            assert row["row"]["input_type"] == "passage"
