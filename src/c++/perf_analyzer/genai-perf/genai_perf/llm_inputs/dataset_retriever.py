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
from typing import Any, Dict, List

import requests
from genai_perf.exceptions import GenAIPerfException
from genai_perf.llm_inputs.synthetic_prompt_generator import SyntheticPromptGenerator
from genai_perf.tokenizer import Tokenizer
from genai_perf.utils import load_json_str


class DatasetRetriever:
    """
    This class retrieves the dataset from different sources and formats it into a corresponding format.
    """

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
                        f"Missing 'text_input' field in file item: {item}"
                    )
                if len(item) != 1:
                    raise GenAIPerfException(
                        f"Field other than 'text_input' field found in file item: {item}"
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
