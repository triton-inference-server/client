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

from pathlib import Path
from unittest.mock import mock_open, patch

import pytest
from genai_perf.llm_inputs.llm_inputs import LlmInputs, ModelSelectionStrategy


class TestLlmInputsRankings:

    def open_side_effects(filepath, *args, **kwargs):
        queries_content = "\n".join(
            [
                '{"text": "What production company co-owned by Kevin Loader and Rodger Michell produced My Cousin Rachel?"}',
                '{"text": "Who served as the 1st Vice President of Colombia under El Libertador?"}',
                '{"text": "Are the Barton Mine and Hermiston-McCauley Mine located in The United States of America?"}',
            ]
        )
        passages_content = "\n".join(
            [
                '{"text": "Eric Anderson (sociologist) Eric Anderson (born January 18, 1968) is an American sociologist"}',
                '{"text": "Kevin Loader is a British film and television producer. "}',
                '{"text": "Barton Mine, also known as Net Lake Mine, is an abandoned surface and underground mine in Northeastern Ontario"}',
            ]
        )

        file_contents = {
            "queries.jsonl": queries_content,
            "passages.jsonl": passages_content,
        }
        return mock_open(
            read_data=file_contents.get(filepath, file_contents["queries.jsonl"])
        )()

    mock_open_obj = mock_open()
    mock_open_obj.side_effect = open_side_effects

    @patch("pathlib.Path.exists", return_value=True)
    @patch("builtins.open", mock_open_obj)
    def test_get_input_dataset_from_rankings_file(self, mock_file):
        queries_filename = Path("queries.jsonl")
        passages_filename = Path("passages.jsonl")
        batch_size = 2
        dataset = LlmInputs._get_input_dataset_from_rankings_files(
            queries_filename, passages_filename, batch_size, num_prompts=100
        )

        assert dataset is not None
        assert len(dataset["rows"]) == 100
        for row in dataset["rows"]:
            assert "row" in row
            assert "payload" in row["row"]
            payload = row["row"]["payload"]
            assert "query" in payload
            assert "passages" in payload
            assert isinstance(payload["passages"], list)
            assert len(payload["passages"]) == batch_size

        # Try error case where batch size is larger than the number of available texts
        with pytest.raises(
            ValueError,
            match="Batch size cannot be larger than the number of available passages",
        ):
            LlmInputs._get_input_dataset_from_rankings_files(
                queries_filename, passages_filename, 5, num_prompts=10
            )

    def test_convert_generic_json_to_openai_rankings_format(self):
        generic_dataset = {
            "rows": [
                {
                    "payload": {
                        "query": {"text": "1"},
                        "passages": [{"text": "2"}, {"text": "3"}, {"text": "4"}],
                    }
                }
            ]
        }

        expected_result = {
            "data": [
                {
                    "payload": [
                        {
                            "query": {"text": "1"},
                            "passages": [{"text": "2"}, {"text": "3"}, {"text": "4"}],
                            "model": "test_model",
                        }
                    ]
                }
            ]
        }

        result = LlmInputs._convert_generic_json_to_rankings_format(
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

    def test_convert_generic_json_to_openai_rankings_format_with_extra_inputs(self):
        generic_dataset = {
            "rows": [
                {
                    "payload": {
                        "query": {"text": "1"},
                        "passages": [{"text": "2"}, {"text": "3"}, {"text": "4"}],
                    }
                }
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
                            "query": {"text": "1"},
                            "passages": [{"text": "2"}, {"text": "3"}, {"text": "4"}],
                            "model": "test_model",
                            "encoding_format": "base64",
                            "truncate": "END",
                            "additional_key": "additional_value",
                        }
                    ]
                }
            ]
        }

        result = LlmInputs._convert_generic_json_to_rankings_format(
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
