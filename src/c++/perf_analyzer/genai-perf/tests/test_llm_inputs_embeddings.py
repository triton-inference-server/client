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

# from pathlib import Path
# from unittest.mock import mock_open, patch

from genai_perf.llm_inputs.llm_inputs import LlmInputs, ModelSelectionStrategy
from genai_perf.llm_inputs.shared import OutputFormat


class TestLlmInputsEmbeddings:
    # TODO: 100 inputs should be generated in this test
    # @patch("pathlib.Path.exists", return_value=True)
    # @patch(
    #     "builtins.open",
    #     new_callable=mock_open,
    #     read_data="\n".join(
    #         [
    #             '{"text_input": "What production company co-owned by Kevin Loader and Rodger Michell produced My Cousin Rachel?"}',
    #             '{"text_input": "Who served as the 1st Vice President of Colombia under El Libertador?"}',
    #             '{"text_input": "Are the Barton Mine and Hermiston-McCauley Mine located in The United States of America?"}',
    #             '{"text_input": "what state did they film daddy\'s home 2"}',
    #         ]
    #     ),
    # )
    # def test_get_input_dataset_from_embeddings_file(self, mock_file, mock_exists):
    # input_filename = Path("embeddings.jsonl")
    # batch_size = 1
    # pa_json = LlmInputs.create_llm_inputs(
    #     model_name=["test_model"],
    #     input_type=PromptSource.FILE,
    #     input_filename=input_filename,
    #     output_format=OutputFormat.OPENAI_EMBEDDINGS,
    #     batch_size=batch_size,
    #     num_of_output_prompts=100,
    # )

    # assert pa_json is not None
    # assert len(pa_json["data"]) == 100
    # for row in pa_json["data"]:
    #     assert "payload" in row
    #     payload = row["payload"][0]
    #     assert "input" in payload
    #     assert isinstance(payload["input"], list)
    #     assert len(payload["input"]) == batch_size

    # TODO: Add and test batching support
    # Try error case where batch size is larger than the number of available texts
    # with pytest.raises(
    #     ValueError,
    #     match="Batch size cannot be larger than the number of available texts",
    # ):
    #     LlmInputs.create_llm_inputs(
    #         input_type=PromptSource.FILE,
    #         input_filename=input_filename,
    #         output_format=OutputFormat.OPENAI_EMBEDDINGS,
    #         batch_size=5,
    #         num_of_output_prompts=10,
    #     )

    def test_convert_generic_json_to_openai_embeddings_format(self):
        generic_dataset = {
            "rows": [
                {"row": {"text_input": "text 1"}},
                {"row": {"text_input": "text 2"}},
            ]
        }

        expected_result = {
            "data": [
                {
                    "payload": [
                        {
                            "input": ["text 1"],
                            "model": "test_model",
                        }
                    ]
                },
                {
                    "payload": [
                        {
                            "input": ["text 2"],
                            "model": "test_model",
                        }
                    ]
                },
            ]
        }

        result = LlmInputs.convert_to_output_format(
            output_format=OutputFormat.OPENAI_EMBEDDINGS,
            generic_dataset=generic_dataset,
            add_model_name=True,
            add_stream=False,
            extra_inputs={},
            output_tokens_mean=0,
            output_tokens_stddev=0,
            output_tokens_deterministic=False,
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
                {"row": {"text_input": "text 1"}},
                {"row": {"text_input": "text 2"}},
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
                            "input": ["text 1"],
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
                            "input": ["text 2"],
                            "model": "test_model",
                            "encoding_format": "base64",
                            "truncate": "END",
                            "additional_key": "additional_value",
                        }
                    ]
                },
            ]
        }

        result = LlmInputs.convert_to_output_format(
            output_format=OutputFormat.OPENAI_EMBEDDINGS,
            generic_dataset=generic_dataset,
            add_model_name=True,
            add_stream=False,
            extra_inputs=extra_inputs,
            output_tokens_mean=0,
            output_tokens_stddev=0,
            output_tokens_deterministic=False,
            model_name=["test_model"],
            model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
        )

        assert result is not None
        assert "data" in result
        assert len(result["data"]) == len(expected_result["data"])

        for i, item in enumerate(expected_result["data"]):
            assert "payload" in result["data"][i]
            assert result["data"][i]["payload"] == item["payload"]
