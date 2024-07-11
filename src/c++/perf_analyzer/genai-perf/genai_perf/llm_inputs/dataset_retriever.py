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

from pathlib import Path
from typing import Any, Dict, List

import requests
from genai_perf.exceptions import GenAIPerfException
from genai_perf.llm_inputs.synthetic_prompt_generator import SyntheticPromptGenerator
from genai_perf.tokenizer import Tokenizer
from genai_perf.utils import load_json_str


class DatasetRetriever:
    @staticmethod
    def from_url(url: str, starting_index: int, length: int) -> List[Dict[str, Any]]:
        url += f"&offset={starting_index}&length={length}"
        response = requests.get(url)
        response.raise_for_status()
        dataset = response.json()
        rows = dataset.get("rows", [])[starting_index : starting_index + length]
        formatted_rows = [
            {
                "text_input": row["row"].get("question", ""),
                "system_prompt": row["row"].get("system_prompt", ""),
                "response": row["row"].get("response", ""),
            }
            for row in rows
        ]
        return formatted_rows

    @staticmethod
    def from_file(file_path: Path) -> List[Dict[str, str]]:
        with open(file_path, "r") as file:
            data = [load_json_str(line) for line in file]

            for item in data:
                if not isinstance(item, dict):
                    raise GenAIPerfException(
                        "File content is not in the expected format."
                    )
                if "text_input" not in item:
                    raise GenAIPerfException(
                        "Missing 'text_input' field in one or more items."
                    )
                if len(item) != 1 or "text_input" not in item:
                    raise GenAIPerfException(
                        "Each item must only contain the 'text_input' field."
                    )

            return [{"text_input": item["text_input"]} for item in data]

    @staticmethod
    def from_directory(directory_path: Path) -> Dict:
        # TODO: Add support for an extra preprocessing step after loading the files to optionally create/modify the dataset.
        # For files calling this method (e.g. rankings), it is a must to create the dataset before converting to the generic format.
        dataset: Dict = {"rows": []}
        data = {}

        # Check all JSONL files in the directory
        for file_path in directory_path.glob("*.jsonl"):
            # Get the file name without suffix
            key = file_path.stem
            with open(file_path, "r") as file:
                data[key] = [load_json_str(line) for line in file]

        # Create rows with keys based on file names without suffix
        num_entries = len(next(iter(data.values())))
        for i in range(num_entries):
            row = {key: data[key][i] for key in data}
            dataset["rows"].append({"row": row})

        return dataset

    @staticmethod
    def from_synthetic(
        tokenizer: Tokenizer,
        prompt_tokens_mean: int,
        prompt_tokens_stddev: int,
        num_of_output_prompts: int,
    ) -> List[Dict[str, str]]:
        synthetic_prompts = []
        for _ in range(num_of_output_prompts):
            synthetic_prompt = SyntheticPromptGenerator.create_synthetic_prompt(
                tokenizer, prompt_tokens_mean, prompt_tokens_stddev
            )
            synthetic_prompts.append({"text_input": synthetic_prompt})
        return synthetic_prompts
