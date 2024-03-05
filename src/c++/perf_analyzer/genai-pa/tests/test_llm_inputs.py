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
from genai_pa.constants import CNN_DAILY_MAIL, OPEN_ORCA
from genai_pa.exceptions import GenAiPAException
from genai_pa.llm_inputs.llm_inputs import (
    InputFormat,
    InputType,
    LlmInputs,
    OutputFormat,
)


class TestLlmInputs:
    @pytest.fixture
    def default_configured_url(self):
        default_configured_url = LlmInputs._create_configured_url(
            LlmInputs.OPEN_ORCA_URL,
            LlmInputs.DEFAULT_STARTING_INDEX,
            LlmInputs.DEFAULT_LENGTH,
        )

        yield default_configured_url

    # TODO: Add tests that verify json schemas

    def test_input_type_url_no_model_name(self):
        """
        Test for exception when input type is URL and no model name
        """
        with pytest.raises(GenAiPAException):
            _ = LlmInputs._check_for_model_name_if_input_type_is_url(
                input_type=InputType.URL, model_name=""
            )

    def test_illegal_starting_index(self):
        """
        Test for exceptions when illegal values are given for starting index
        """
        with pytest.raises(GenAiPAException):
            _ = LlmInputs._check_for_valid_starting_index(starting_index="foo")

        with pytest.raises(GenAiPAException):
            _ = LlmInputs._check_for_valid_starting_index(starting_index=-1)

    def test_illegal_length(self):
        """
        Test for exceptions when illegal values are given for length
        """
        with pytest.raises(GenAiPAException):
            _ = LlmInputs._check_for_valid_length(length="foo")

        with pytest.raises(GenAiPAException):
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
        with pytest.raises(GenAiPAException):
            _ = LlmInputs._download_dataset(
                "https://bad-url.zzz",
                LlmInputs.DEFAULT_STARTING_INDEX,
                LlmInputs.DEFAULT_LENGTH,
            )

    def test_llm_inputs_error_in_server_response(self):
        """
        Test for exception when length is out of range
        """
        with pytest.raises(GenAiPAException):
            _ = LlmInputs.create_llm_inputs(
                input_type=InputType.URL,
                input_format=InputFormat.OPENAI,
                output_format=OutputFormat.OPENAI,
                model_name=OPEN_ORCA,
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
        dataset_json = LlmInputs._convert_input_dataset_to_generic_json(
            input_format=InputFormat.OPENAI, dataset=dataset
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
        dataset_json = LlmInputs._convert_input_dataset_to_generic_json(
            input_format=InputFormat.OPENAI, dataset=dataset
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
        dataset_json = LlmInputs._convert_input_dataset_to_generic_json(
            input_format=InputFormat.OPENAI, dataset=dataset
        )
        pa_json = LlmInputs._convert_generic_json_to_output_format(
            output_format=OutputFormat.OPENAI,
            generic_dataset=dataset_json,
            add_model_name=False,
            add_stream=False,
        )

        assert pa_json is not None
        assert len(pa_json["data"][0]["payload"]) == LlmInputs.DEFAULT_LENGTH

    def test_create_openai_llm_inputs_cnn_dailymail(self):
        """
        Test CNN_DAILYMAIL can be accessed
        """
        pa_json = LlmInputs.create_llm_inputs(
            input_type=InputType.URL,
            input_format=InputFormat.OPENAI,
            output_format=OutputFormat.OPENAI,
            model_name=CNN_DAILY_MAIL,
        )

        os.remove(LlmInputs.OUTPUT_FILENAME)

        assert pa_json is not None
        assert len(pa_json["data"][0]["payload"]) == LlmInputs.DEFAULT_LENGTH

    def test_write_to_file(self):
        """
        Test that write to file is working correctly
        """
        pa_json = LlmInputs.create_llm_inputs(
            input_type=InputType.URL,
            input_format=InputFormat.OPENAI,
            output_format=OutputFormat.OPENAI,
            model_name=OPEN_ORCA,
            add_model_name=True,
            add_stream=True,
        )
        try:
            f = open(LlmInputs.OUTPUT_FILENAME, "r")
            json_str = f.read()
        finally:
            f.close()
            os.remove(LlmInputs.OUTPUT_FILENAME)

        assert pa_json == json.loads(json_str)

    def test_create_openai_to_vllm(self):
        """
        Test conversion of openai to vllm
        """
        pa_json = LlmInputs.create_llm_inputs(
            input_type=InputType.URL,
            input_format=InputFormat.OPENAI,
            output_format=OutputFormat.VLLM,
            model_name=OPEN_ORCA,
            add_model_name=False,
            add_stream=True,
        )

        os.remove(LlmInputs.OUTPUT_FILENAME)

        assert pa_json is not None
        assert len(pa_json["data"]) == LlmInputs.DEFAULT_LENGTH
