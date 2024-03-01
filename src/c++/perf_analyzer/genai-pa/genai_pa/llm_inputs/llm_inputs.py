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
from typing import Dict, List, Optional, Tuple

import requests
from genai_pa.exceptions import GenAiPAException
from requests import Response


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

    EMPTY_JSON_IN_OPENAI_PA_FORMAT = {"data": [{"payload": []}]}

    @classmethod
    def create_openai_llm_inputs(
        cls,
        url: str = OPEN_ORCA_URL,
        starting_index: int = DEFAULT_STARTING_INDEX,
        length: int = DEFAULT_LENGTH,
        model_name: str = None,
        add_stream: bool = False,
    ) -> Dict:
        """
        Given a URL and indexing parameters, it will write a string of LLM Inputs
        (in a JSON dictionary) to a file

        Parameters
        ----------
        url:
            URL to gather LLM Inputs from
        starting_index:
            Offset from within the list to start gathering inputs
        length:
            Number of entries to gather
        model_name:
            If included adds this model name field to each payload
        add_stream:
            If true adds a steam field to each payload
        """

        LlmInputs._check_for_valid_args(starting_index, length)
        configured_url = LlmInputs._create_configured_url(url, starting_index, length)
        dataset = LlmInputs._download_dataset(configured_url, starting_index, length)
        dataset_json = LlmInputs._convert_dataset_to_json(dataset)
        json_in_pa_format = LlmInputs._convert_json_to_pa_format(
            dataset_json, model_name, add_stream
        )
        LlmInputs._write_json_to_file(json_in_pa_format)

        return json_in_pa_format

    @classmethod
    def _check_for_valid_args(cls, starting_index: int, length: int) -> None:
        try:
            LlmInputs._check_for_valid_starting_index(starting_index)
            LlmInputs._check_for_valid_length(length)
        except Exception as e:
            raise GenAiPAException(e)

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
    def _convert_dataset_to_json(cls, dataset: Response) -> Dict:
        dataset_json = dataset.json()
        try:
            LlmInputs._check_for_error_in_json_of_dataset(dataset_json)
        except Exception as e:
            raise GenAiPAException(e)

        return dataset_json

    @classmethod
    def _convert_json_to_pa_format(
        cls, dataset_json: Dict, model_name: str, add_stream: bool
    ) -> Dict:
        system_role_headers, user_role_headers = LlmInputs._determine_json_pa_roles(
            dataset_json
        )
        pa_json = LlmInputs._populate_openai_pa_json(
            dataset_json,
            system_role_headers,
            user_role_headers,
            model_name,
            add_stream,
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
    def _determine_json_pa_roles(
        cls, dataset_json: Dict
    ) -> Tuple[List[str], List[str]]:
        SYSTEM_ROLE_LIST = ["system_prompt"]
        USER_ROLE_LIST = ["question", "article"]

        system_role_headers, user_role_headers = [], []
        if "features" in dataset_json.keys():
            for index, feature in enumerate(dataset_json["features"]):
                if feature["name"] in SYSTEM_ROLE_LIST:
                    system_role_headers.append(feature["name"])
                if feature["name"] in USER_ROLE_LIST:
                    user_role_headers.append(feature["name"])

        assert system_role_headers is not None or user_role_headers is not None

        return system_role_headers, user_role_headers

    @classmethod
    def _populate_openai_pa_json(
        cls,
        dataset_json: Dict,
        system_role_headers: List[str],
        user_role_headers: List[str],
        model_name: str,
        add_stream: bool,
    ) -> Dict:
        pa_json = LlmInputs._create_empty_openai_pa_json()

        for entry in dataset_json["rows"]:
            pa_json["data"][0]["payload"].append({"messages": []})

            for header in entry["row"]:
                new_message = LlmInputs._create_new_message(
                    header, system_role_headers, user_role_headers, entry["row"][header]
                )

                pa_json = LlmInputs._add_new_message_to_json(
                    pa_json, entry["row_idx"], new_message
                )

            pa_json = LlmInputs._add_optional_tags_to_json(
                pa_json, entry["row_idx"], model_name, add_stream
            )

        return pa_json

    @classmethod
    def _create_empty_openai_pa_json(cls) -> Dict:
        empty_pa_json = deepcopy(LlmInputs.EMPTY_JSON_IN_OPENAI_PA_FORMAT)

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
    def _add_new_message_to_json(
        cls, pa_json: Dict, index: int, new_message: Optional[Dict]
    ) -> Dict:
        if new_message:
            pa_json["data"][0]["payload"][index]["messages"].append(new_message)

        return pa_json

    @classmethod
    def _add_optional_tags_to_json(
        cls, pa_json: Dict, index: int, model_name: str, add_stream: bool
    ) -> Dict:
        if model_name:
            pa_json["data"][0]["payload"][index]["model"] = model_name
        if add_stream:
            pa_json["data"][0]["payload"][index]["steam"] = "true"

        return pa_json

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
