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
from genai_perf import utils
from genai_perf.exceptions import GenAIPerfException
from genai_perf.llm_inputs.inputs_utils import ImageFormat, OutputFormat
from genai_perf.llm_inputs.synthetic_image_generator import SyntheticImageGenerator
from genai_perf.llm_inputs.synthetic_prompt_generator import SyntheticPromptGenerator
from genai_perf.tokenizer import Tokenizer
from PIL import Image


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
    def from_file(file_path: Path, output_format: OutputFormat) -> List[Dict[str, str]]:
        contents = DatasetRetriever._load_file_content(file_path)

        dataset = []
        for content in contents:
            data = {"text_input": content.get("text_input", "")}

            if output_format == OutputFormat.OPENAI_VISION:
                img_filename = content.get("image", "")
                encoded_img = DatasetRetriever._encode_image_to_base64(img_filename)
                data["image"] = encoded_img

            dataset.append(data)
        return dataset

    @staticmethod
    def _load_file_content(file_path: Path) -> List[Dict[str, str]]:
        contents = []
        with open(file_path, "r") as file:
            for line in file:
                content = utils.load_json_str(line)
                if not isinstance(content, dict):
                    raise GenAIPerfException(
                        "File content is not in the expected format."
                    )
                if "text_input" not in content:
                    raise GenAIPerfException(
                        f"Missing 'text_input' field in file content: {content}"
                    )
                contents.append(content)

        return contents

    @staticmethod
    def _encode_image_to_base64(filename: str) -> str:
        try:
            img = Image.open(filename)
        except:
            raise GenAIPerfException(
                f"Error occurred while opening an image file: {filename}"
            )

        if img.format.lower() not in utils.get_enum_names(ImageFormat):
            raise GenAIPerfException(
                f"Unsupported image format '{img.format}' of "
                f"the image '{filename}'."
            )

        img_base64 = utils.encode_image(img, img.format)
        return f"data:image/{img.format.lower()};base64,{img_base64}"

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
                data[key] = [utils.load_json_str(line) for line in file]

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
        image_width_mean: int,
        image_width_stddev: int,
        image_height_mean: int,
        image_height_stddev: int,
        image_format: ImageFormat,
        output_format: OutputFormat,
    ) -> List[Dict[str, str]]:
        synthetic_dataset = []
        for _ in range(num_of_output_prompts):
            prompt = SyntheticPromptGenerator.create_synthetic_prompt(
                tokenizer, prompt_tokens_mean, prompt_tokens_stddev
            )
            data = {"text_input": prompt}

            if output_format == OutputFormat.OPENAI_VISION:
                image = SyntheticImageGenerator.create_synthetic_image(
                    image_width_mean=image_width_mean,
                    image_width_stddev=image_width_stddev,
                    image_height_mean=image_height_mean,
                    image_height_stddev=image_height_stddev,
                    image_format=image_format,
                )
                data["image"] = image

            synthetic_dataset.append(data)
        return synthetic_dataset
