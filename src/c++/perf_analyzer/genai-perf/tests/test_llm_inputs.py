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
import os

import pytest
from genai_perf import tokenizer
from genai_perf.constants import CNN_DAILY_MAIL, DEFAULT_INPUT_DATA_JSON, OPEN_ORCA
from genai_perf.exceptions import GenAIPerfException
from genai_perf.llm_inputs.llm_inputs import LlmInputs, OutputFormat, PromptSource


class TestLlmInputs:
    @pytest.fixture
    def default_configured_url(self):
        default_configured_url = LlmInputs._create_configured_url(
            LlmInputs.OPEN_ORCA_URL,
            LlmInputs.DEFAULT_STARTING_INDEX,
            LlmInputs.DEFAULT_LENGTH,
        )

        yield default_configured_url

    # TODO (TMA-1754): Add tests that verify json schemas
    @pytest.fixture
    def default_tokenizer(self):
        yield tokenizer.get_tokenizer(tokenizer.DEFAULT_TOKENIZER)

    def test_input_type_url_no_dataset_name(self):
        """
        Test for exception when input type is URL and no dataset name
        """
        with pytest.raises(GenAIPerfException):
            _ = LlmInputs._check_for_dataset_name_if_input_type_is_url(
                input_type=PromptSource.DATASET, dataset_name=""
            )

    def test_input_type_synthetic_no_tokenizer(self):
        """
        Test for exception when input type is SYNTHETIC and no tokenizer
        """
        with pytest.raises(GenAIPerfException):
            _ = LlmInputs._check_for_tokenzier_if_input_type_is_synthetic(
                input_type=PromptSource.SYNTHETIC, tokenizer=None
            )

    def test_illegal_starting_index(self):
        """
        Test for exceptions when illegal values are given for starting index
        """
        with pytest.raises(GenAIPerfException):
            _ = LlmInputs._check_for_valid_starting_index(starting_index="foo")

        with pytest.raises(GenAIPerfException):
            _ = LlmInputs._check_for_valid_starting_index(starting_index=-1)

    def test_illegal_length(self):
        """
        Test for exceptions when illegal values are given for length
        """
        with pytest.raises(GenAIPerfException):
            _ = LlmInputs._check_for_valid_length(length="foo")

        with pytest.raises(GenAIPerfException):
            _ = LlmInputs._check_for_valid_length(length=0)

    def test_create_configured_url(self):
        """
        Test that we are appending and configuring the URL correctly
        """
        expected_configured_url = (
            "http://test-url.com"
            + f"&offset={LlmInputs.DEFAULT_STARTING_INDEX}"
            + f"&length={LlmInputs.DEFAULT_LENGTH}"
        )
        configured_url = LlmInputs._create_configured_url(
            "http://test-url.com",
            LlmInputs.DEFAULT_STARTING_INDEX,
            LlmInputs.DEFAULT_LENGTH,
        )

        assert configured_url == expected_configured_url

    def test_download_dataset_illegal_url(self):
        """
        Test for exception when URL is bad
        """
        with pytest.raises(GenAIPerfException):
            _ = LlmInputs._download_dataset(
                "https://bad-url.zzz",
                LlmInputs.DEFAULT_STARTING_INDEX,
                LlmInputs.DEFAULT_LENGTH,
            )

    def test_llm_inputs_error_in_server_response(self):
        """
        Test for exception when length is out of range
        """
        with pytest.raises(GenAIPerfException):
            _ = LlmInputs.create_llm_inputs(
                input_type=PromptSource.DATASET,
                dataset_name=OPEN_ORCA,
                output_format=OutputFormat.OPENAI_CHAT_COMPLETIONS,
                starting_index=LlmInputs.DEFAULT_STARTING_INDEX,
                length=int(LlmInputs.DEFAULT_LENGTH * 100),
            )

    def test_llm_inputs_with_defaults(self, default_configured_url):
        """
        Test that default options work
        """
        dataset = LlmInputs._download_dataset(
            default_configured_url,
            LlmInputs.DEFAULT_STARTING_INDEX,
            LlmInputs.DEFAULT_LENGTH,
        )
        dataset_json = LlmInputs._convert_input_url_dataset_to_generic_json(
            dataset=dataset
        )

        assert dataset_json is not None
        assert len(dataset_json["rows"]) == LlmInputs.DEFAULT_LENGTH

    def test_llm_inputs_with_non_default_length(self):
        """
        Test that non-default length works
        """
        configured_url = LlmInputs._create_configured_url(
            LlmInputs.OPEN_ORCA_URL,
            LlmInputs.DEFAULT_STARTING_INDEX,
            (int(LlmInputs.DEFAULT_LENGTH / 2)),
        )
        dataset = LlmInputs._download_dataset(
            configured_url,
            LlmInputs.DEFAULT_STARTING_INDEX,
            length=(int(LlmInputs.DEFAULT_LENGTH / 2)),
        )
        dataset_json = LlmInputs._convert_input_url_dataset_to_generic_json(
            dataset=dataset
        )

        assert dataset_json is not None
        assert len(dataset_json["rows"]) == LlmInputs.DEFAULT_LENGTH / 2

    def test_convert_default_json_to_pa_format(self, default_configured_url):
        """
        Test that conversion to PA JSON format is correct
        """
        dataset = LlmInputs._download_dataset(
            default_configured_url,
            LlmInputs.DEFAULT_STARTING_INDEX,
            LlmInputs.DEFAULT_LENGTH,
        )
        dataset_json = LlmInputs._convert_input_url_dataset_to_generic_json(
            dataset=dataset
        )
        pa_json = LlmInputs._convert_generic_json_to_output_format(
            output_format=OutputFormat.OPENAI_CHAT_COMPLETIONS,
            generic_dataset=dataset_json,
            add_model_name=False,
            add_stream=False,
        )

        assert pa_json is not None
        assert len(pa_json["data"]) == LlmInputs.DEFAULT_LENGTH

    def test_create_openai_llm_inputs_cnn_dailymail(self):
        """
        Test CNN_DAILYMAIL can be accessed
        """
        pa_json = LlmInputs.create_llm_inputs(
            input_type=PromptSource.DATASET,
            dataset_name=CNN_DAILY_MAIL,
            output_format=OutputFormat.OPENAI_CHAT_COMPLETIONS,
        )

        os.remove(DEFAULT_INPUT_DATA_JSON)

        assert pa_json is not None
        assert len(pa_json["data"]) == LlmInputs.DEFAULT_LENGTH

    def test_write_to_file(self):
        """
        Test that write to file is working correctly
        """
        pa_json = LlmInputs.create_llm_inputs(
            input_type=PromptSource.DATASET,
            dataset_name=OPEN_ORCA,
            output_format=OutputFormat.OPENAI_CHAT_COMPLETIONS,
            model_name="open_orca",
            add_model_name=True,
            add_stream=True,
        )
        try:
            f = open(DEFAULT_INPUT_DATA_JSON, "r")
            json_str = f.read()
        finally:
            f.close()
            os.remove(DEFAULT_INPUT_DATA_JSON)

        assert pa_json == json.loads(json_str)

    def test_create_openai_to_vllm(self):
        """
        Test conversion of openai to vllm
        """
        pa_json = LlmInputs.create_llm_inputs(
            input_type=PromptSource.DATASET,
            output_format=OutputFormat.VLLM,
            dataset_name=OPEN_ORCA,
            add_model_name=False,
            add_stream=True,
        )

        os.remove(DEFAULT_INPUT_DATA_JSON)

        assert pa_json is not None
        assert len(pa_json["data"]) == LlmInputs.DEFAULT_LENGTH

    def test_create_openai_to_completions(self):
        """
        Test conversion of openai to completions
        """
        pa_json = LlmInputs.create_llm_inputs(
            input_type=PromptSource.DATASET,
            output_format=OutputFormat.OPENAI_COMPLETIONS,
            dataset_name=OPEN_ORCA,
            add_model_name=False,
            add_stream=True,
        )

        os.remove(DEFAULT_INPUT_DATA_JSON)

        assert pa_json is not None
        assert len(pa_json["data"]) == LlmInputs.DEFAULT_LENGTH

    def test_create_openai_to_trtllm(self):
        """
        Test conversion of openai to trtllm
        """
        pa_json = LlmInputs.create_llm_inputs(
            input_type=PromptSource.DATASET,
            output_format=OutputFormat.TRTLLM,
            dataset_name=OPEN_ORCA,
            add_model_name=False,
            add_stream=True,
        )

        os.remove(DEFAULT_INPUT_DATA_JSON)

        assert pa_json is not None
        assert len(pa_json["data"]) == LlmInputs.DEFAULT_LENGTH

    def test_random_synthetic(self, default_tokenizer):
        """
        Test that we can produce deterministic random synthetic prompts
        """
        synthetic_prompt, synthetic_prompt_tokens = LlmInputs._create_synthetic_prompt(
            default_tokenizer,
            LlmInputs.DEFAULT_PROMPT_TOKENS_MEAN,
            LlmInputs.DEFAULT_PROMPT_TOKENS_STDDEV,
            LlmInputs.DEFAULT_REQUESTED_OUTPUT_TOKENS,
            LlmInputs.DEFAULT_RANDOM_SEED,
        )

        # 550 is the num of tokens returned for the default seed
        assert synthetic_prompt_tokens == 550

        synthetic_prompt, synthetic_prompt_tokens = LlmInputs._create_synthetic_prompt(
            default_tokenizer,
            LlmInputs.DEFAULT_PROMPT_TOKENS_MEAN,
            LlmInputs.DEFAULT_PROMPT_TOKENS_STDDEV + 250,
            LlmInputs.DEFAULT_REQUESTED_OUTPUT_TOKENS,
            LlmInputs.DEFAULT_RANDOM_SEED + 1,
        )
        assert synthetic_prompt_tokens != 785

    def test_synthetic_to_vllm(self, default_tokenizer):
        """
        Test generating synthetic prompts and converting to vllm
        """
        pa_json = LlmInputs.create_llm_inputs(
            input_type=PromptSource.SYNTHETIC,
            output_format=OutputFormat.VLLM,
            num_of_output_prompts=5,
            add_model_name=False,
            add_stream=True,
            tokenizer=default_tokenizer,
        )

        os.remove(DEFAULT_INPUT_DATA_JSON)

        assert pa_json is not None
        assert len(pa_json["data"]) == 5

    def test_synthetic_to_trtllm(self, default_tokenizer):
        """
        Test generating synthetic prompts and converting to trtllm
        """
        pa_json = LlmInputs.create_llm_inputs(
            input_type=PromptSource.SYNTHETIC,
            output_format=OutputFormat.TRTLLM,
            num_of_output_prompts=5,
            add_model_name=False,
            add_stream=True,
            tokenizer=default_tokenizer,
        )

        os.remove(DEFAULT_INPUT_DATA_JSON)

        assert pa_json is not None
        assert len(pa_json["data"]) == 5

    def test_synthetic_to_openai_chat_completions(self, default_tokenizer):
        """
        Test generating synthetic prompts and converting to OpenAI chat completions
        """
        pa_json = LlmInputs.create_llm_inputs(
            input_type=PromptSource.SYNTHETIC,
            output_format=OutputFormat.OPENAI_CHAT_COMPLETIONS,
            num_of_output_prompts=5,
            add_model_name=False,
            add_stream=True,
            tokenizer=default_tokenizer,
        )

        os.remove(DEFAULT_INPUT_DATA_JSON)

        assert pa_json is not None
        assert len(pa_json["data"]) == 5

    def test_synthetic_to_openai_completions(self, default_tokenizer):
        """
        Test generating synthetic prompts and converting to OpenAI completions
        """
        pa_json = LlmInputs.create_llm_inputs(
            input_type=PromptSource.SYNTHETIC,
            output_format=OutputFormat.OPENAI_COMPLETIONS,
            num_of_output_prompts=5,
            add_model_name=False,
            add_stream=True,
            tokenizer=default_tokenizer,
        )

        os.remove(DEFAULT_INPUT_DATA_JSON)

        assert pa_json is not None
        assert len(pa_json["data"]) == 5

    @pytest.mark.parametrize(
        "output_format",
        [
            (OutputFormat.OPENAI_CHAT_COMPLETIONS),
            (OutputFormat.OPENAI_COMPLETIONS),
            (OutputFormat.TRTLLM),
            (OutputFormat.VLLM),
        ],
    )
    def test_llm_inputs_extra_inputs(self, default_tokenizer, output_format) -> None:
        # Simulate --request-input arguments
        request_inputs = {"additional_key": "additional_value"}

        # Generate input data with the additional request inputs
        pa_json = LlmInputs.create_llm_inputs(
            input_type=PromptSource.SYNTHETIC,
            output_format=output_format,
            num_of_output_prompts=5,  # Generate a small number of prompts for the test
            add_model_name=False,
            add_stream=True,
            tokenizer=default_tokenizer,
            extra_inputs=request_inputs,  # Pass the simulated --request-input arguments here
        )

        assert len(pa_json["data"]) == 5

        # Verify that each entry in the generated JSON includes the additional key-value pairs
        if (
            output_format == OutputFormat.OPENAI_CHAT_COMPLETIONS
            or output_format == OutputFormat.OPENAI_COMPLETIONS
        ):
            for entry in pa_json.get("data", []):
                assert "payload" in entry, "Payload is missing in the request"
                payload = entry["payload"]
                for item in payload:
                    assert (
                        "additional_key" in item
                    ), "The additional_key is not present in the request"
                    assert (
                        item["additional_key"] == "additional_value"
                    ), "The value of additional_key is incorrect"
        elif output_format == OutputFormat.TRTLLM or output_format == OutputFormat.VLLM:
            for entry in pa_json.get("data", []):
                assert (
                    "additional_key" in entry
                ), "The additional_key is not present in the request"
                assert entry["additional_key"] == [
                    "additional_value"
                ], "The value of additional_key is incorrect"
        else:
            assert False, f"Unsupported output format: {output_format}"
