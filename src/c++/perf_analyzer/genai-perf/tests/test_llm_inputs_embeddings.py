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
from genai_perf.llm_inputs.llm_inputs import LlmInputs, ModelSelectionStrategy


class TestLlmInputsEmbeddings:
    @patch("pathlib.Path.exists", return_value=True)
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data="\n".join(
            [
                '{"text": "What production company co-owned by Kevin Loader and Rodger Michell produced My Cousin Rachel?"}',
                '{"text": "Who served as the 1st Vice President of Colombia under El Libertador?"}',
                '{"text": "Are the Barton Mine and Hermiston-McCauley Mine located in The United States of America?"}',
                '{"text": "what state did they film daddy\'s home 2"}',
            ]
        ),
    )
    def test_get_input_dataset_from_embeddings_file(self, mock_file, mock_exists):
        input_filename = Path("embeddings.jsonl")
        batch_size = 3
        dataset = LlmInputs._get_input_dataset_from_embeddings_file(
            input_filename, batch_size, num_prompts=100
        )

        assert dataset is not None
        assert len(dataset["rows"]) == 100
        for row in dataset["rows"]:
            assert "row" in row
            assert "payload" in row["row"]
            payload = row["row"]["payload"]
            assert "input" in payload
            assert isinstance(payload["input"], list)
            assert len(payload["input"]) == batch_size

        # Try error case where batch size is larger than the number of available texts
        with pytest.raises(
            ValueError,
            match="Batch size cannot be larger than the number of available texts",
        ):
            LlmInputs._get_input_dataset_from_embeddings_file(
                input_filename, 5, num_prompts=10
            )

    def test_convert_generic_json_to_openai_embeddings_format(self):
        generic_dataset = {
            "rows": [
                {"payload": {"input": ["text 1", "text 2"]}},
                {"payload": {"input": ["text 3", "text 4"]}},
            ]
        }

        expected_result = {
            "data": [
                {
                    "payload": [
                        {
                            "input": ["text 1", "text 2"],
                            "model": "test_model",
                            "input_type": "query",
                            "encoding_format": "float",
                            "truncate": "NONE",
                        }
                    ]
                },
                {
                    "payload": [
                        {
                            "input": ["text 3", "text 4"],
                            "input_type": "query",
                            "model": "test_model",
                            "encoding_format": "float",
                            "truncate": "NONE",
                        }
                    ]
                },
            ]
        }

        result = LlmInputs._convert_generic_json_to_openai_embeddings_format(
            generic_dataset,
            extra_inputs={},
            model_name=["test_model"],
            model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
        )

        assert result is not None
        assert "data" in result
        assert len(result["data"]) == len(expected_result["data"])

        for i, item in enumerate(expected_result["data"]):
            assert "payload" in result["data"][i]
            assert result["data"][i]["payload"] == item["payload"]

    def test_convert_generic_json_to_openai_embeddings_format_with_extra_inputs(self):
        generic_dataset = {
            "rows": [
                {"payload": {"input": ["text 1", "text 2"]}},
                {"payload": {"input": ["text 3", "text 4"]}},
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
                            "input": ["text 1", "text 2"],
                            "input_type": "query",
                            "model": "test_model",
                            "encoding_format": "base64",
                            "truncate": "END",
                            "additional_key": "additional_value",
                        }
                    ]
                },
                {
                    "payload": [
                        {
                            "input": ["text 3", "text 4"],
                            "input_type": "query",
                            "model": "test_model",
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

        assert result is not None
        assert "data" in result
        assert len(result["data"]) == len(expected_result["data"])

        for i, item in enumerate(expected_result["data"]):
            assert "payload" in result["data"][i]
            assert result["data"][i]["payload"] == item["payload"]
