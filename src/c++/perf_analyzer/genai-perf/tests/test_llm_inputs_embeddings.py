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

import json
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest
from genai_perf import tokenizer
from genai_perf.llm_inputs.llm_inputs import LlmInputs, ModelSelectionStrategy


class TestLlmInputsEmbeddings:
    # TODO: Add extra inputs tests for other inputs
    # Check number of prompts
    # Check correct number of prompts generated
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
            {"row": {"payload": {"input": "example question", "input_type": "query"}}},
            {
                "row": {
                    "payload": {"input": "example question_2", "input_type": "query"}
                }
            },
        ]

        assert dataset is not None
        assert len(dataset["rows"]) == len(expected_rows)

        for i, row in enumerate(expected_rows):
            assert "row" in dataset["rows"][i]
            assert "payload" in dataset["rows"][i]["row"]
            assert dataset["rows"][i]["row"]["payload"] == row["row"]["payload"]

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
            {
                "row": {
                    "payload": {
                        "input": "example of a relevant passage",
                        "input_type": "passage",
                    }
                }
            },
            {
                "row": {
                    "payload": {
                        "input": "example of an irrelevant passage",
                        "input_type": "passage",
                    }
                }
            },
            {
                "row": {
                    "payload": {
                        "input": "another example of an irrelevant passage",
                        "input_type": "passage",
                    }
                }
            },
            {
                "row": {
                    "payload": {
                        "input": "example of a relevant passage_2",
                        "input_type": "passage",
                    }
                }
            },
            {
                "row": {
                    "payload": {
                        "input": "example of an irrelevant passage_2",
                        "input_type": "passage",
                    }
                }
            },
            {
                "row": {
                    "payload": {
                        "input": "another example of an irrelevant passage_2",
                        "input_type": "passage",
                    }
                }
            },
        ]

        assert dataset is not None
        assert len(dataset["rows"]) == len(expected_rows)
        for i, row in enumerate(expected_rows):
            assert "row" in dataset["rows"][i]
            assert "payload" in dataset["rows"][i]["row"]
            assert dataset["rows"][i]["row"]["payload"] == row["row"]["payload"]

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
            embeddings_prompts_stddev=0,
        )

        assert dataset is not None
        assert len(dataset["rows"]) == 5
        for row in dataset["rows"]:
            assert "row" in row
            assert "payload" in row["row"]
            payload = row["row"]["payload"]
            assert isinstance(payload["input"], list)
            assert len(payload["input"]) == 3
            assert "input_type" in payload
            assert payload["input_type"] == "query"

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
            embeddings_prompts_stddev=0,
        )

        assert dataset is not None
        assert len(dataset["rows"]) == 5
        for row in dataset["rows"]:
            assert "row" in row
            assert "payload" in row["row"]
            payload = row["row"]["payload"]
            assert isinstance(payload["input"], list)
            assert len(payload["input"]) == 3
            assert "input_type" in payload
            assert payload["input_type"] == "passage"

    def test_convert_generic_json_to_openai_embeddings_format(self):
        generic_dataset = {
            "rows": [
                {"payload": {"input": "text 1", "input_type": "passage"}},
                {"payload": {"input": "text 2", "input_type": "passage"}},
                {"payload": {"input": None, "input_type": "passage"}},
                {"payload": {"input": "text 3", "input_type": None}},
            ]
        }

        expected_result = {
            "data": [
                {
                    "payload": [
                        {
                            "input": "text 1",
                            "model": "test_model",
                            "input_type": "passage",
                        }
                    ]
                },
                {
                    "payload": [
                        {
                            "input": "text 2",
                            "model": "test_model",
                            "input_type": "passage",
                        }
                    ]
                },
            ]
        }

        # Check that the valid case works
        result = LlmInputs._convert_generic_json_to_openai_embeddings_format(
            {
                "rows": generic_dataset["rows"][:2]
            },  # Only take the first two valid entries
            extra_inputs={},
            model_name=["test_model"],
            model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
        )

        for i, item in enumerate(expected_result["data"]):
            assert "payload" in result["data"][i]
            payload_list = result["data"][i]["payload"]
            expected_payload_list = item["payload"]
            assert len(payload_list) == 1
            assert len(expected_payload_list) == 1
            assert payload_list[0]["input"] == expected_payload_list[0]["input"]
            assert payload_list[0]["model"] == expected_payload_list[0]["model"]
            assert (
                payload_list[0]["input_type"] == expected_payload_list[0]["input_type"]
            )

        # Now, check that the invalid cases raise a ValueError
        with pytest.raises(
            ValueError, match="Missing required fields 'input' in dataset entry"
        ):
            LlmInputs._convert_generic_json_to_openai_embeddings_format(
                {"rows": generic_dataset["rows"][2:3]},
                extra_inputs={},
                model_name=["test_model"],
                model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
            )

        with pytest.raises(
            ValueError, match="Missing required fields 'input_type' in dataset entry"
        ):
            LlmInputs._convert_generic_json_to_openai_embeddings_format(
                {"rows": generic_dataset["rows"][3:]},
                extra_inputs={},
                model_name=["test_model"],
                model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
            )

    def test_convert_generic_json_to_openai_embeddings_format_with_extra_inputs(self):
        generic_dataset = {
            "rows": [
                {"payload": {"input": "text 1", "input_type": "passage"}},
                {"payload": {"input": "text 2", "input_type": "passage"}},
            ]
        }

        extra_inputs = {
            "encoding_format": "base64",
            "truncate": "END",
            "additional_key": "additional_value",
        }

        expected_result = {
            "data": [
                {
                    "payload": [
                        {
                            "input": "text 1",
                            "model": "test_model",
                            "input_type": "passage",
                            "encoding_format": "base64",
                            "truncate": "END",
                            "additional_key": "additional_value",
                        }
                    ]
                },
                {
                    "payload": [
                        {
                            "input": "text 2",
                            "model": "test_model",
                            "input_type": "passage",
                            "encoding_format": "base64",
                            "truncate": "END",
                            "additional_key": "additional_value",
                        }
                    ]
                },
            ]
        }

        result = LlmInputs._convert_generic_json_to_openai_embeddings_format(
            generic_dataset,
            extra_inputs=extra_inputs,
            model_name=["test_model"],
            model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
        )

        for i, item in enumerate(expected_result["data"]):
            assert "payload" in result["data"][i]
            payload_list = result["data"][i]["payload"]
            expected_payload_list = item["payload"]
            assert len(payload_list) == 1
            assert len(expected_payload_list) == 1
            assert payload_list[0]["input"] == expected_payload_list[0]["input"]
            assert payload_list[0]["model"] == expected_payload_list[0]["model"]
            assert (
                payload_list[0]["input_type"] == expected_payload_list[0]["input_type"]
            )
            assert (
                payload_list[0]["encoding_format"]
                == expected_payload_list[0]["encoding_format"]
            )
            assert payload_list[0]["truncate"] == expected_payload_list[0]["truncate"]
            assert (
                payload_list[0]["additional_key"]
                == expected_payload_list[0]["additional_key"]
            )
