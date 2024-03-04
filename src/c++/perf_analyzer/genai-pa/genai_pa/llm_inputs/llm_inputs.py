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
from copy import deepcopy
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import requests
from genai_pa.constants import CNN_DAILY_MAIL, OPEN_ORCA
from genai_pa.exceptions import GenAiPAException
from requests import Response


class InputType(Enum):
    URL = auto()
    FILE = auto()


class InputFormat(Enum):
    OPENAI = auto()
    TRTLLM = auto()
    VLLM = auto()


class OutputFormat(Enum):
    OPENAI = auto()
    TRTLLM = auto()
    VLLM = auto()


class LlmInputs:
    """
    A library of methods that control the generation of LLM Inputs
    """

    OUTPUT_FILENAME = "./llm_inputs.json"

    OPEN_ORCA_URL = "https://datasets-server.huggingface.co/rows?dataset=Open-Orca%2FOpenOrca&config=default&split=train"
    CNN_DAILYMAIL_URL = "https://datasets-server.huggingface.co/rows?dataset=cnn_dailymail&config=1.0.0&split=train"

    DEFAULT_STARTING_INDEX = 0
    MINIMUM_STARTING_INDEX = 0

    DEFAULT_LENGTH = 100
    MINIMUM_LENGTH = 1

    EMPTY_JSON_IN_VLLM_PA_FORMAT = {"data": []}
    EMPTY_JSON_IN_TRTLLM_PA_FORMAT = {"data": []}
    EMPTY_JSON_IN_OPENAI_PA_FORMAT = {"data": [{"payload": []}]}

    dataset_url_map = {OPEN_ORCA: OPEN_ORCA_URL, CNN_DAILY_MAIL: CNN_DAILYMAIL_URL}

    @classmethod
    def create_llm_inputs(
        cls,
        input_type: InputType,
        input_format: InputFormat,
        output_format: OutputFormat,
        model_name: str = "",
        input_filename: str = "",
        starting_index: int = DEFAULT_STARTING_INDEX,
        length: int = DEFAULT_LENGTH,
        add_model_name: bool = False,
        add_stream: bool = False,
    ) -> Dict:
        """
        Given an input type, input format, and output type. Output a string of LLM Inputs
        (in a JSON dictionary) to a file

        Required Parameters
        -------------------
        input_type:
            Specify how the input is received (file or URL)
        input_format:
            Specify the input format
        output_format:
            Specify the output format

        Optional Parameters
        -------------------
        model_name:
            The model name
        starting_index:
            Offset from within the list to start gathering inputs
        length:
            Number of entries to gather
        add_model_name:
            If true adds a model name field to each payload
        add_stream:
            If true adds a steam field to each payload
        """

        LlmInputs._check_for_valid_args(input_type, model_name, starting_index, length)

        if input_type == InputType.URL:
            dataset = LlmInputs._get_input_dataset_from_url(
                model_name, starting_index, length
            )
        elif input_type == InputType.FILE:
            raise GenAiPAException(
                "Using a file to supply LLM Input is not supported at this time"
            )

        generic_dataset_json = LlmInputs._convert_input_dataset_to_generic_json(
            input_format, dataset
        )

        json_in_pa_format = LlmInputs._convert_generic_json_to_output_format(
            output_format, generic_dataset_json, add_model_name, add_stream, model_name
        )
        LlmInputs._write_json_to_file(json_in_pa_format)

        return json_in_pa_format

    @classmethod
    def _check_for_valid_args(
        cls, input_type: InputType, model_name: str, starting_index: int, length: int
    ) -> None:
        try:
            LlmInputs._check_for_model_name_if_input_type_is_url(input_type, model_name)
            LlmInputs._check_for_valid_starting_index(starting_index)
            LlmInputs._check_for_valid_length(length)
        except Exception as e:
            raise GenAiPAException(e)

    @classmethod
    def _get_input_dataset_from_url(
        cls, model_name: str, starting_index: int, length: int
    ) -> Response:
        url = LlmInputs._resolve_url(model_name)
        configured_url = LlmInputs._create_configured_url(url, starting_index, length)
        dataset = LlmInputs._download_dataset(configured_url, starting_index, length)

        return dataset

    @classmethod
    def _resolve_url(cls, model_name: str) -> str:
        if model_name in LlmInputs.dataset_url_map:
            return LlmInputs.dataset_url_map[model_name]
        else:
            raise GenAiPAException(
                f"{model_name} does not have a corresponding URL in the dataset_url_map."
            )

    @classmethod
    def _create_configured_url(cls, url: str, starting_index: int, length: int) -> str:
        starting_index_str = str(starting_index)
        length_str = str(length)
        configured_url = url + f"&offset={starting_index_str}&length={length_str}"

        return configured_url

    @classmethod
    def _download_dataset(cls, configured_url, starting_index, length) -> Response:
        dataset = LlmInputs._query_server(configured_url)

        return dataset

    @classmethod
    def _convert_input_dataset_to_generic_json(
        cls, input_format: InputFormat, dataset: Response
    ) -> Dict:
        dataset_json = dataset.json()
        try:
            LlmInputs._check_for_error_in_json_of_dataset(dataset_json)
        except Exception as e:
            raise GenAiPAException(e)

        if input_format == InputFormat.OPENAI:
            generic_dataset_json = LlmInputs._convert_openai_to_generic_input_json(
                dataset_json
            )
        else:
            raise GenAiPAException(
                f"An input format of {input_format} is not supported at this time"
            )

        return generic_dataset_json

    @classmethod
    def _convert_openai_to_generic_input_json(cls, dataset_json: Dict) -> Dict:
        generic_input_json = {}
        if "features" in dataset_json.keys():
            generic_input_json["features"] = []
            for feature in dataset_json["features"]:
                generic_input_json["features"].append(feature["name"])

        generic_input_json["rows"] = []
        for row in dataset_json["rows"]:
            generic_input_json["rows"].append(row["row"])

        return generic_input_json

    @classmethod
    def _convert_generic_json_to_output_format(
        cls,
        output_format: OutputFormat,
        generic_dataset: Dict,
        add_model_name: bool,
        add_stream: bool,
        model_name: str = "",
    ) -> Dict:
        if output_format == OutputFormat.OPENAI:
            output_json = LlmInputs._convert_generic_json_to_openai_format(
                generic_dataset, add_model_name, add_stream, model_name
            )
        elif output_format == OutputFormat.VLLM:
            output_json = LlmInputs._convert_generic_json_to_vllm_format(
                generic_dataset, add_model_name, add_stream, model_name
            )
        else:
            raise GenAiPAException(
                f"Output format {output_format} is not currently supported"
            )

        return output_json

    @classmethod
    def _convert_generic_json_to_openai_format(
        cls,
        dataset_json: Dict,
        add_model_name: bool,
        add_stream: bool,
        model_name: str = "",
    ) -> Dict:
        # OPEN: Don't know how to select a role for `text_input`
        (
            system_role_headers,
            user_role_headers,
            _,
        ) = LlmInputs._determine_json_feature_roles(dataset_json)
        pa_json = LlmInputs._populate_openai_output_json(
            dataset_json,
            system_role_headers,
            user_role_headers,
            add_model_name,
            add_stream,
            model_name,
        )

        return pa_json

    @classmethod
    def _convert_generic_json_to_vllm_format(
        cls,
        dataset_json: Dict,
        add_model_name: bool,
        add_stream: bool,
        model_name: str = "",
    ) -> Dict:
        (
            system_role_headers,
            user_role_headers,
            text_input_headers,
        ) = LlmInputs._determine_json_feature_roles(dataset_json)
        pa_json = LlmInputs._populate_vllm_output_json(
            dataset_json,
            system_role_headers,
            user_role_headers,
            text_input_headers,
            add_model_name,
            add_stream,
            model_name,
        )

        return pa_json

    @classmethod
    def _write_json_to_file(cls, json_in_pa_format: Dict):
        try:
            f = open(LlmInputs.OUTPUT_FILENAME, "w")
            f.write(json.dumps(json_in_pa_format, indent=2))
        finally:
            f.close()

    @classmethod
    def _determine_json_feature_roles(
        cls, dataset_json: Dict
    ) -> Tuple[List[str], List[str]]:
        SYSTEM_ROLE_LIST = ["system_prompt"]
        USER_ROLE_LIST = ["question", "article"]
        TEXT_INPUT_LIST = ["text_input"]

        system_role_headers, user_role_headers, text_input_headers = [], [], []
        if "features" in dataset_json.keys():
            for index, feature in enumerate(dataset_json["features"]):
                if feature in SYSTEM_ROLE_LIST:
                    system_role_headers.append(feature)
                if feature in USER_ROLE_LIST:
                    user_role_headers.append(feature)
                if feature in TEXT_INPUT_LIST:
                    user_role_headers.append(feature)

        assert (
            system_role_headers is not None
            or user_role_headers is not None
            or text_input_headers is not None
        )

        return system_role_headers, user_role_headers, text_input_headers

    @classmethod
    def _populate_openai_output_json(
        cls,
        dataset_json: Dict,
        system_role_headers: List[str],
        user_role_headers: List[str],
        add_model_name: bool,
        add_stream: bool,
        model_name: str = "",
    ) -> Dict:
        pa_json = LlmInputs._create_empty_openai_pa_json()

        for index, entry in enumerate(dataset_json["rows"]):
            pa_json["data"][0]["payload"].append({"messages": []})

            for header, content in entry.items():
                new_message = LlmInputs._create_new_message(
                    header, system_role_headers, user_role_headers, content
                )

                pa_json = LlmInputs._add_new_message_to_json(
                    pa_json, index, new_message
                )

            pa_json = LlmInputs._add_optional_tags_to_openai_json(
                pa_json, index, add_model_name, add_stream, model_name
            )

        return pa_json

    @classmethod
    def _populate_vllm_output_json(
        cls,
        dataset_json: Dict,
        system_role_headers: List[str],
        user_role_headers: List[str],
        text_input_headers: List[str],
        add_model_name: bool,
        add_stream: bool,
        model_name: str = "",
    ) -> Dict:
        pa_json = LlmInputs._create_empty_vllm_pa_json()

        for index, entry in enumerate(dataset_json["rows"]):
            pa_json["data"].append({"text_input": []})

            for header, content in entry.items():
                new_text_input = LlmInputs._create_new_text_input(
                    header,
                    system_role_headers,
                    user_role_headers,
                    text_input_headers,
                    content,
                )

                pa_json = LlmInputs._add_new_text_input_to_json(
                    pa_json, index, new_text_input
                )

            pa_json = LlmInputs._add_optional_tags_to_vllm_json(
                pa_json, index, add_model_name, add_stream, model_name
            )

        return pa_json

    @classmethod
    def _create_empty_openai_pa_json(cls) -> Dict:
        empty_pa_json = deepcopy(LlmInputs.EMPTY_JSON_IN_OPENAI_PA_FORMAT)

        return empty_pa_json

    @classmethod
    def _create_empty_vllm_pa_json(cls) -> Dict:
        empty_pa_json = deepcopy(LlmInputs.EMPTY_JSON_IN_VLLM_PA_FORMAT)

        return empty_pa_json

    @classmethod
    def _create_new_message(
        cls,
        header: str,
        system_role_headers: List[str],
        user_role_headers: List[str],
        content: str,
    ) -> Optional[Dict]:
        if header in system_role_headers:
            new_message = {
                "role": "system",
                "content": content,
            }
        elif header in user_role_headers:
            new_message = {
                "role": "user",
                "content": content,
            }
        else:
            new_message = {}

        return new_message

    @classmethod
    def _create_new_text_input(
        cls,
        header: str,
        system_role_headers: List[str],
        user_role_headers: List[str],
        text_input_headers: List[str],
        content: str,
    ) -> Optional[str]:
        new_text_input = ""

        if (
            header in system_role_headers
            or header in user_role_headers
            or header in text_input_headers
        ):
            new_text_input = content

        return new_text_input

    @classmethod
    def _add_new_message_to_json(
        cls, pa_json: Dict, index: int, new_message: Optional[Dict]
    ) -> Dict:
        if new_message:
            pa_json["data"][0]["payload"][index]["messages"].append(new_message)

        return pa_json

    @classmethod
    def _add_new_text_input_to_json(
        cls, pa_json: Dict, index: int, new_text_input: str
    ) -> Dict:
        if new_text_input:
            pa_json["data"][index]["text_input"].append(new_text_input)

        return pa_json

    @classmethod
    def _add_optional_tags_to_openai_json(
        cls,
        pa_json: Dict,
        index: int,
        add_model_name: bool,
        add_stream: bool,
        model_name: str = "",
    ) -> Dict:
        if add_model_name:
            pa_json["data"][0]["payload"][index]["model"] = model_name
        if add_stream:
            pa_json["data"][0]["payload"][index]["stream"] = [True]

        return pa_json

    @classmethod
    def _add_optional_tags_to_vllm_json(
        cls,
        pa_json: Dict,
        index: int,
        add_model_name: bool,
        add_stream: bool,
        model_name: str = "",
    ) -> Dict:
        if add_model_name:
            pa_json["data"][index]["model"] = model_name
        if add_stream:
            pa_json["data"][index]["stream"] = [True]

        return pa_json

    @classmethod
    def _check_for_model_name_if_input_type_is_url(
        cls, input_type: InputType, model_name: str
    ) -> None:
        if input_type == InputType.URL and not model_name:
            raise GenAiPAException(
                "Input type is URL, but model_name is not specified."
            )

    @classmethod
    def _check_for_valid_starting_index(cls, starting_index: int) -> None:
        if not isinstance(starting_index, int):
            raise GenAiPAException(
                f"starting_index: {starting_index} must be an integer."
            )

        if starting_index < LlmInputs.MINIMUM_STARTING_INDEX:
            raise GenAiPAException(
                f"starting_index: {starting_index} must be larger than {LlmInputs.MINIMUM_STARTING_INDEX}."
            )

    @classmethod
    def _check_for_valid_length(cls, length: int) -> None:
        if not isinstance(length, int):
            raise GenAiPAException(f"length: {length} must be an integer.")

        if length < LlmInputs.MINIMUM_LENGTH:
            raise GenAiPAException(
                f"starting_index: {length} must be larger than {LlmInputs.MINIMUM_LENGTH}."
            )

    @classmethod
    def _query_server(cls, configured_url: str) -> Response:
        try:
            response = requests.get(configured_url)
        except Exception as e:
            error_message = LlmInputs._create_error_message(e)
            raise GenAiPAException(error_message)

        return response

    @classmethod
    def _create_error_message(cls, exception: Exception) -> str:
        url_str = exception.args[0].args[0]
        url_start = url_str.find("'")
        url_end = url_str.find("'", url_start + 1) + 1
        error_message = f"Invalid URL: {url_str[url_start:url_end]}"

        return error_message

    @classmethod
    def _check_for_error_in_json_of_dataset(cls, json_of_dataset: str) -> None:
        if "error" in json_of_dataset.keys():
            raise GenAiPAException(json_of_dataset["error"])
