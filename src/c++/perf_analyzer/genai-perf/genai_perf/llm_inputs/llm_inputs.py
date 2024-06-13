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
import random
from copy import deepcopy
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import requests
from genai_perf import utils
from genai_perf.constants import CNN_DAILY_MAIL, DEFAULT_INPUT_DATA_JSON, OPEN_ORCA
from genai_perf.exceptions import GenAIPerfException
from genai_perf.llm_inputs.synthetic_image_generator import (
    ImageFormat,
    SyntheticImageGenerator,
)
from genai_perf.llm_inputs.synthetic_prompt_generator import SyntheticPromptGenerator
from genai_perf.tokenizer import DEFAULT_TOKENIZER, Tokenizer, get_tokenizer
from genai_perf.utils import load_json_str
from PIL import Image
from requests import Response


class ModelSelectionStrategy(Enum):
    ROUND_ROBIN = auto()
    RANDOM = auto()


class PromptSource(Enum):
    SYNTHETIC = auto()
    DATASET = auto()
    FILE = auto()


class OutputFormat(Enum):
    OPENAI_CHAT_COMPLETIONS = auto()
    OPENAI_COMPLETIONS = auto()
    OPENAI_EMBEDDINGS = auto()
    OPENAI_VISION = auto()
    RANKINGS = auto()
    TENSORRTLLM = auto()
    VLLM = auto()
    TENSORRTLLM_ENGINE = auto()

    def to_lowercase(self):
        return self.name.lower()


class LlmInputs:
    """
    A library of methods that control the generation of LLM Inputs
    """

    OPEN_ORCA_URL = "https://datasets-server.huggingface.co/rows?dataset=Open-Orca%2FOpenOrca&config=default&split=train"
    CNN_DAILYMAIL_URL = "https://datasets-server.huggingface.co/rows?dataset=cnn_dailymail&config=1.0.0&split=train"

    DEFAULT_STARTING_INDEX = 0
    MINIMUM_STARTING_INDEX = 0

    DEFAULT_LENGTH = 100
    MINIMUM_LENGTH = 1

    DEFAULT_TENSORRTLLM_MAX_TOKENS = 256

    DEFAULT_BATCH_SIZE = 1
    DEFAULT_RANDOM_SEED = 0
    DEFAULT_PROMPT_TOKENS_MEAN = 550
    DEFAULT_PROMPT_TOKENS_STDDEV = 0
    DEFAULT_OUTPUT_TOKENS_MEAN = -1
    DEFAULT_OUTPUT_TOKENS_STDDEV = 0
    DEFAULT_NUM_PROMPTS = 100

    DEFAULT_IMAGE_WIDTH_MEAN = 100
    DEFAULT_IMAGE_WIDTH_STDDEV = 0
    DEFAULT_IMAGE_HEIGHT_MEAN = 100
    DEFAULT_IMAGE_HEIGHT_STDDEV = 0

    EMPTY_JSON_IN_VLLM_PA_FORMAT: Dict = {"data": []}
    EMPTY_JSON_IN_TENSORRTLLM_PA_FORMAT: Dict = {"data": []}
    EMPTY_JSON_IN_OPENAI_PA_FORMAT: Dict = {"data": []}

    dataset_url_map = {OPEN_ORCA: OPEN_ORCA_URL, CNN_DAILY_MAIL: CNN_DAILYMAIL_URL}

    @classmethod
    def create_llm_inputs(
        cls,
        input_type: PromptSource,
        output_format: OutputFormat,
        dataset_name: str = "",
        model_name: list = [],
        model_selection_strategy: ModelSelectionStrategy = ModelSelectionStrategy.ROUND_ROBIN,
        input_filename: Optional[Path] = Path(""),
        starting_index: int = DEFAULT_STARTING_INDEX,
        length: int = DEFAULT_LENGTH,
        output_tokens_mean: int = DEFAULT_OUTPUT_TOKENS_MEAN,
        output_tokens_stddev: int = DEFAULT_OUTPUT_TOKENS_STDDEV,
        output_tokens_deterministic: bool = False,
        prompt_tokens_mean: int = DEFAULT_PROMPT_TOKENS_MEAN,
        prompt_tokens_stddev: int = DEFAULT_PROMPT_TOKENS_STDDEV,
        image_width_mean: int = DEFAULT_IMAGE_WIDTH_MEAN,
        image_width_stddev: int = DEFAULT_IMAGE_WIDTH_STDDEV,
        image_height_mean: int = DEFAULT_IMAGE_HEIGHT_MEAN,
        image_height_stddev: int = DEFAULT_IMAGE_HEIGHT_STDDEV,
        image_format: ImageFormat = ImageFormat.PNG,
        random_seed: int = DEFAULT_RANDOM_SEED,
        num_of_output_prompts: int = DEFAULT_NUM_PROMPTS,
        add_model_name: bool = False,
        add_stream: bool = False,
        tokenizer: Tokenizer = get_tokenizer(DEFAULT_TOKENIZER),
        extra_inputs: Optional[Dict] = None,
        batch_size: int = 1,
        output_dir: Path = Path(""),
    ) -> Dict:
        """
        Given an input type, input format, and output type. Output a string of LLM Inputs
        (in a JSON dictionary) to a file

        Required Parameters
        -------------------
        input_type:
            Specify how the input is received
        output_format:
            Specify the output format

        Optional Parameters
        -------------------
        dataset_name:
            The name of the dataset
        model_name:
            The model name
        starting_index:
            Offset from within the list to start gathering inputs
        length:
            Number of entries to gather
        add_model_name:
            If true, adds a model name field to each payload
        add_stream:
            If true, adds a steam field to each payload
        extra_inputs:
            If provided, append these inputs to every request
        output_tokens_mean:
            The mean length of the output to generate. If not using fixed output lengths, this should be set to -1.
        output_tokens_stddev:
            The standard deviation of the length of the output to generate. This is only used if output_tokens_mean is provided.
        output_tokens_deterministic:
            If true, the output tokens will set the minimum and maximum tokens to be equivalent.
        image_width_mean:
            The mean width of images when generating synthetic image data.
        image_width_stddev:
            The standard deviation of width of images when generating synthetic image data.
        image_height_mean:
            The mean height of images when generating synthetic image data.
        image_height_stddev:
            The standard deviation of height of images when generating synthetic image data.
        image_format:
            The compression format of the images.
        batch_size:
            The number of inputs per request (currently only used for the embeddings and rankings endpoints)

        Required Synthetic Prompt Generation Parameters
        -----------------------------------------------
        tokenizer:
           The tokenizer to use when generating synthetic prompts

        Optional Synthetic Prompt Generation Parameters
        -----------------------------------------------
        prompt_tokens_mean:
            The mean length of the prompt to generate
        prompt_tokens_stddev:
            The standard deviation of the length of the prompt to generate
        num_of_output_prompts:
            The number of synthetic output prompts to generate
        random_seed:
            Seed used to generate random values
        """

        cls._check_for_valid_args(
            input_type, dataset_name, starting_index, length, tokenizer
        )

        random.seed(random_seed)

        generic_dataset_json = cls.get_generic_dataset_json(
            input_type,
            output_format,
            dataset_name,
            starting_index,
            length,
            tokenizer,
            prompt_tokens_mean,
            prompt_tokens_stddev,
            num_of_output_prompts,
            image_width_mean,
            image_width_stddev,
            image_height_mean,
            image_height_stddev,
            image_format,
            batch_size,
            input_filename,
        )

        if extra_inputs is None:
            extra_inputs = {}

        json_in_pa_format = cls._convert_generic_json_to_output_format(
            output_format,
            generic_dataset_json,
            add_model_name,
            add_stream,
            extra_inputs,
            output_tokens_mean,
            output_tokens_stddev,
            output_tokens_deterministic,
            model_name,
            model_selection_strategy,
        )
        cls._write_json_to_file(json_in_pa_format, output_dir)

        return json_in_pa_format

    @classmethod
    def get_generic_dataset_json(
        cls,
        input_type: PromptSource,
        output_format: OutputFormat,
        dataset_name: str,
        starting_index: int,
        length: int,
        tokenizer: Tokenizer,
        prompt_tokens_mean: int,
        prompt_tokens_stddev: int,
        num_of_output_prompts: int,
        image_width_mean: int,
        image_width_stddev: int,
        image_height_mean: int,
        image_height_stddev: int,
        image_format: ImageFormat,
        batch_size: int,
        input_filename: Optional[Path],
    ) -> Dict:
        """
        Retrieve and convert the dataset based on the input type.

        Parameters
        ----------
        input_type:
            Specify how the input is received
        output_format:
            Specify the output format
        dataset_name:
            The name of the dataset
        starting_index:
            Offset from within the list to start gathering inputs
        length:
            Number of entries to gather
        tokenizer:
            The tokenizer to use when generating synthetic prompts
        prompt_tokens_mean:
            The mean length of the prompt to generate
        prompt_tokens_stddev:
            The standard deviation of the length of the prompt to generate
        num_of_output_prompts:
            The number of synthetic output prompts to generate
        image_width_mean:
            The mean width of images when generating synthetic image data.
        image_width_stddev:
            The standard deviation of width of images when generating synthetic image data.
        image_height_mean:
            The mean height of images when generating synthetic image data.
        image_height_stddev:
            The standard deviation of height of images when generating synthetic image data.
        image_format:
            The compression format of the images.
        batch_size:
            The number of inputs per request (currently only used for the embeddings and rankings endpoints)
        input_filename:
            The path to the input file containing the prompts in JSONL format.
        Returns
        -------
        Dict:
            The generic dataset JSON
        """

        if output_format == OutputFormat.OPENAI_EMBEDDINGS:
            if input_type != PromptSource.FILE:
                raise GenAIPerfException(
                    f"{OutputFormat.OPENAI_EMBEDDINGS.to_lowercase()} only supports a file as input."
                )
            input_filename = cast(Path, input_filename)
            input_file_dataset = cls._get_input_dataset_from_embeddings_file(
                input_filename,
                batch_size,
                num_of_output_prompts,
            )
            generic_dataset_json = (
                cls._convert_input_synthetic_or_file_dataset_to_generic_json(
                    input_file_dataset
                )
            )
        elif output_format == OutputFormat.RANKINGS:
            if input_type != PromptSource.FILE:
                raise GenAIPerfException(
                    f"{OutputFormat.RANKINGS.to_lowercase()} only supports a directory as input."
                )
            queries_filename = cast(Path, input_filename) / "queries.jsonl"
            passages_filename = cast(Path, input_filename) / "passages.jsonl"
            input_file_dataset = cls._get_input_dataset_from_rankings_files(
                queries_filename, passages_filename, batch_size, num_of_output_prompts
            )

            generic_dataset_json = (
                cls._convert_input_synthetic_or_file_dataset_to_generic_json(
                    input_file_dataset
                )
            )
        else:
            if input_type == PromptSource.DATASET:
                # (TMA-1990) support VLM input from public dataset
                if output_format == OutputFormat.OPENAI_VISION:
                    raise GenAIPerfException(
                        f"{OutputFormat.OPENAI_VISION.to_lowercase()} currently "
                        "does not support dataset as input."
                    )
                dataset = cls._get_input_dataset_from_url(
                    dataset_name, starting_index, length
                )
                generic_dataset_json = cls._convert_input_url_dataset_to_generic_json(
                    dataset
                )
            elif input_type == PromptSource.SYNTHETIC:
                synthetic_dataset = cls._get_input_dataset_from_synthetic(
                    tokenizer,
                    prompt_tokens_mean,
                    prompt_tokens_stddev,
                    num_of_output_prompts,
                    image_width_mean,
                    image_width_stddev,
                    image_height_mean,
                    image_height_stddev,
                    image_format,
                    output_format,
                )
                generic_dataset_json = (
                    cls._convert_input_synthetic_or_file_dataset_to_generic_json(
                        synthetic_dataset
                    )
                )
            elif input_type == PromptSource.FILE:
                input_filename = cast(Path, input_filename)
                input_file_dataset = cls._get_input_dataset_from_file(input_filename)
                input_file_dataset = cls._encode_images_in_input_dataset(
                    input_file_dataset
                )
                generic_dataset_json = (
                    cls._convert_input_synthetic_or_file_dataset_to_generic_json(
                        input_file_dataset
                    )
                )
            else:
                raise GenAIPerfException("Input source is not recognized.")

            # When the generic_dataset_json contains multi-modal data (e.g. images),
            # convert the format of the content to OpenAI multi-modal format:
            # see https://platform.openai.com/docs/guides/vision
            if output_format == OutputFormat.OPENAI_VISION:
                generic_dataset_json = cls._convert_to_openai_multi_modal_content(
                    generic_dataset_json
                )

        return generic_dataset_json

    @classmethod
    def _get_input_dataset_from_embeddings_file(
        cls, input_filename: Path, batch_size: int, num_prompts: int
    ) -> Dict[str, Any]:
        with open(input_filename, "r") as file:
            file_content = [load_json_str(line) for line in file]

        texts = [item["text"] for item in file_content]

        if batch_size > len(texts):
            raise ValueError(
                "Batch size cannot be larger than the number of available texts"
            )

        dataset_json: Dict[str, Any] = {}
        dataset_json["features"] = [{"name": "input"}]
        dataset_json["rows"] = []

        for _ in range(num_prompts):
            sampled_texts = random.sample(texts, batch_size)
            dataset_json["rows"].append({"row": {"payload": {"input": sampled_texts}}})

        return dataset_json

    @classmethod
    def _get_input_dataset_from_rankings_files(
        cls,
        queries_filename: Path,
        passages_filename: Path,
        batch_size: int,
        num_prompts: int,
    ) -> Dict[str, Any]:

        with open(queries_filename, "r") as file:
            queries_content = [load_json_str(line) for line in file]
        queries_texts = [item for item in queries_content]

        with open(passages_filename, "r") as file:
            passages_content = [load_json_str(line) for line in file]
        passages_texts = [item for item in passages_content]

        if batch_size > len(passages_texts):
            raise ValueError(
                "Batch size cannot be larger than the number of available passages"
            )

        dataset_json: Dict[str, Any] = {}
        dataset_json["features"] = [{"name": "input"}]
        dataset_json["rows"] = []

        for _ in range(num_prompts):
            sampled_texts = random.sample(passages_texts, batch_size)
            query_sample = random.choice(queries_texts)
            entry_dict: Dict = {}
            entry_dict["query"] = query_sample
            entry_dict["passages"] = sampled_texts
            dataset_json["rows"].append({"row": {"payload": entry_dict}})
        return dataset_json

    @classmethod
    def _check_for_valid_args(
        cls,
        input_type: PromptSource,
        dataset_name: str,
        starting_index: int,
        length: int,
        tokenizer: Tokenizer,
    ) -> None:
        try:
            cls._check_for_dataset_name_if_input_type_is_url(input_type, dataset_name)
            cls._check_for_tokenzier_if_input_type_is_synthetic(input_type, tokenizer)
            cls._check_for_valid_starting_index(starting_index)
            cls._check_for_valid_length(length)

        except Exception as e:
            raise GenAIPerfException(e)

    @classmethod
    def _get_input_dataset_from_url(
        cls, dataset_name: str, starting_index: int, length: int
    ) -> Response:
        url = cls._resolve_url(dataset_name)
        configured_url = cls._create_configured_url(url, starting_index, length)
        dataset = cls._download_dataset(configured_url)

        return dataset

    @classmethod
    def _get_input_dataset_from_synthetic(
        cls,
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
    ) -> Dict[str, Any]:
        dataset_json: Dict[str, Any] = {}
        dataset_json["features"] = [{"name": "text_input"}]
        dataset_json["rows"] = []
        for _ in range(num_of_output_prompts):
            row: Dict["str", Any] = {"row": {}}
            synthetic_prompt = cls._create_synthetic_prompt(
                tokenizer,
                prompt_tokens_mean,
                prompt_tokens_stddev,
            )
            row["row"]["text_input"] = synthetic_prompt

            if output_format == OutputFormat.OPENAI_VISION:
                synthetic_image = cls._create_synthetic_image(
                    image_width_mean=image_width_mean,
                    image_width_stddev=image_width_stddev,
                    image_height_mean=image_height_mean,
                    image_height_stddev=image_height_stddev,
                    image_format=image_format,
                )
                row["row"]["image"] = synthetic_image

            dataset_json["rows"].append(row)

        return dataset_json

    @classmethod
    def _resolve_url(cls, dataset_name: str) -> str:
        if dataset_name in cls.dataset_url_map:
            return cls.dataset_url_map[dataset_name]
        else:
            raise GenAIPerfException(
                f"{dataset_name} does not have a corresponding URL in the dataset_url_map."
            )

    @classmethod
    def _create_configured_url(cls, url: str, starting_index: int, length: int) -> str:
        starting_index_str = str(starting_index)
        length_str = str(length)
        configured_url = url + f"&offset={starting_index_str}&length={length_str}"

        return configured_url

    @classmethod
    def _download_dataset(cls, configured_url: str) -> Response:
        dataset = cls._query_server(configured_url)

        return dataset

    @classmethod
    def _convert_input_url_dataset_to_generic_json(cls, dataset: Response) -> Dict:
        dataset_json = dataset.json()
        try:
            cls._check_for_error_in_json_of_dataset(dataset_json)
        except Exception as e:
            raise GenAIPerfException(e)

        generic_dataset_json = cls._convert_dataset_to_generic_input_json(dataset_json)

        return generic_dataset_json

    @classmethod
    def _convert_input_synthetic_or_file_dataset_to_generic_json(
        cls, dataset: Dict
    ) -> Dict[str, List[Dict]]:
        generic_dataset_json = cls._convert_dataset_to_generic_input_json(dataset)

        return generic_dataset_json

    @classmethod
    def _convert_dataset_to_generic_input_json(
        cls, dataset_json: Dict
    ) -> Dict[str, List[Dict]]:
        generic_input_json = cls._add_features_to_generic_json({}, dataset_json)
        generic_input_json = cls._add_rows_to_generic_json(
            generic_input_json, dataset_json
        )

        return generic_input_json

    @classmethod
    def _add_features_to_generic_json(
        cls, generic_input_json: Dict, dataset_json: Dict
    ) -> Dict:
        if "features" in dataset_json.keys():
            generic_input_json["features"] = []
            for feature in dataset_json["features"]:
                generic_input_json["features"].append(feature["name"])

        return generic_input_json

    @classmethod
    def _add_rows_to_generic_json(
        cls, generic_input_json: Dict, dataset_json: Dict
    ) -> Dict[str, List[Dict]]:
        generic_input_json["rows"] = []
        for row in dataset_json["rows"]:
            generic_input_json["rows"].append(row["row"])

        return generic_input_json

    @classmethod
    def _get_input_dataset_from_file(cls, input_filename: Path) -> Dict:
        """
        Reads the input prompts and images from a JSONL file and converts them
        into the required dataset format.

        Parameters
        ----------
        input_filename : Path
            The path to the input file containing the prompts and/or images in
            JSONL format.

        Returns
        -------
        Dict
            The dataset in the required format with the prompts and/or images
            read from the file.
        """
        cls.verify_file(input_filename)
        prompts, images = cls._get_prompts_from_input_file(input_filename)
        dataset_json: Dict[str, Any] = {}
        dataset_json["features"] = [{"name": "text_input"}]
        dataset_json["rows"] = []
        for prompt, image in zip(prompts, images):
            content = {"text_input": prompt}
            content.update({"image": image} if image else {})
            dataset_json["rows"].append({"row": content})

        return dataset_json

    @classmethod
    def _get_prompts_from_input_file(
        cls, input_filename: Path
    ) -> Tuple[List[str], List[str]]:
        """
        Reads the input prompts from a JSONL file and returns a list of prompts.

        Parameters
        ----------
        input_filename : Path
            The path to the input file containing the prompts in JSONL format.

        Returns
        -------
        Tuple[List[str], List[str]]
            A list of prompts and images read from the file.
        """
        prompts = []
        images = []
        with open(input_filename, mode="r", newline=None) as file:
            for line in file:
                if line.strip():
                    prompts.append(load_json_str(line).get("text_input", "").strip())
                    images.append(load_json_str(line).get("image", "").strip())
        return prompts, images

    @classmethod
    def verify_file(cls, input_filename: Path) -> None:
        if not input_filename.exists():
            raise FileNotFoundError(f"The file '{input_filename}' does not exist.")

    @classmethod
    def _convert_to_openai_multi_modal_content(
        cls, generic_dataset_json: Dict[str, List[Dict]]
    ) -> Dict[str, List[Dict]]:
        """
        Converts to multi-modal content format of OpenAI Chat Completions API.
        """
        for row in generic_dataset_json["rows"]:
            if row["image"]:
                row["text_input"] = [
                    {
                        "type": "text",
                        "text": row["text_input"],
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": row["image"]},
                    },
                ]

        return generic_dataset_json

    @classmethod
    def _encode_images_in_input_dataset(cls, input_file_dataset: Dict) -> Dict:
        for row in input_file_dataset["rows"]:
            filename = row["row"].get("image")
            if filename:
                img = Image.open(filename)
                if img.format.lower() not in utils.get_enum_names(ImageFormat):
                    raise GenAIPerfException(
                        f"Unsupported image format '{img.format}' of "
                        f"the image '{filename}'."
                    )

                img_base64 = utils.encode_image(img, img.format)
                payload = f"data:image/{img.format.lower()};base64,{img_base64}"
                row["row"]["image"] = payload

        return input_file_dataset

    @classmethod
    def _convert_generic_json_to_output_format(
        cls,
        output_format: OutputFormat,
        generic_dataset: Dict,
        add_model_name: bool,
        add_stream: bool,
        extra_inputs: Dict,
        output_tokens_mean: int,
        output_tokens_stddev: int,
        output_tokens_deterministic: bool,
        model_name: list = [],
        model_selection_strategy: ModelSelectionStrategy = ModelSelectionStrategy.ROUND_ROBIN,
    ) -> Dict:
        if (
            output_format == OutputFormat.OPENAI_CHAT_COMPLETIONS
            or output_format == OutputFormat.OPENAI_VISION
        ):
            output_json = cls._convert_generic_json_to_openai_chat_completions_format(
                generic_dataset,
                add_model_name,
                add_stream,
                extra_inputs,
                output_tokens_mean,
                output_tokens_stddev,
                output_tokens_deterministic,
                model_name,
                model_selection_strategy,
            )
        elif output_format == OutputFormat.OPENAI_COMPLETIONS:
            output_json = cls._convert_generic_json_to_openai_completions_format(
                generic_dataset,
                add_model_name,
                add_stream,
                extra_inputs,
                output_tokens_mean,
                output_tokens_stddev,
                output_tokens_deterministic,
                model_name,
                model_selection_strategy,
            )
        elif output_format == OutputFormat.OPENAI_EMBEDDINGS:
            output_json = cls._convert_generic_json_to_openai_embeddings_format(
                generic_dataset,
                extra_inputs,
                model_name,
                model_selection_strategy,
            )
        elif output_format == OutputFormat.RANKINGS:
            output_json = cls._convert_generic_json_to_rankings_format(
                generic_dataset,
                extra_inputs,
                model_name,
                model_selection_strategy,
            )
        elif output_format == OutputFormat.VLLM:
            output_json = cls._convert_generic_json_to_vllm_format(
                generic_dataset,
                add_model_name,
                add_stream,
                extra_inputs,
                output_tokens_mean,
                output_tokens_stddev,
                output_tokens_deterministic,
                model_name,
                model_selection_strategy,
            )
        elif output_format == OutputFormat.TENSORRTLLM:
            output_json = cls._convert_generic_json_to_trtllm_format(
                generic_dataset,
                add_model_name,
                add_stream,
                extra_inputs,
                output_tokens_mean,
                output_tokens_stddev,
                output_tokens_deterministic,
                model_name,
                model_selection_strategy,
            )
        else:
            raise GenAIPerfException(
                f"Output format {output_format} is not currently supported"
            )

        return output_json

    @classmethod
    def _convert_generic_json_to_openai_chat_completions_format(
        cls,
        dataset_json: Dict,
        add_model_name: bool,
        add_stream: bool,
        extra_inputs: Dict,
        output_tokens_mean: int,
        output_tokens_stddev: int,
        output_tokens_deterministic: bool,
        model_name: list = [],
        model_selection_strategy: ModelSelectionStrategy = ModelSelectionStrategy.ROUND_ROBIN,
    ) -> Dict:
        # TODO (TMA-1757): Implement a way to select a role for `text_input`
        (
            system_role_headers,
            user_role_headers,
            _,
        ) = cls._determine_json_feature_roles(dataset_json)
        pa_json = cls._populate_openai_chat_completions_output_json(
            dataset_json,
            system_role_headers,
            user_role_headers,
            add_model_name,
            add_stream,
            extra_inputs,
            output_tokens_mean,
            output_tokens_stddev,
            output_tokens_deterministic,
            model_name,
            model_selection_strategy,
        )

        return pa_json

    @classmethod
    def _convert_generic_json_to_openai_completions_format(
        cls,
        dataset_json: Dict,
        add_model_name: bool,
        add_stream: bool,
        extra_inputs: Dict,
        output_tokens_mean: int,
        output_tokens_stddev: int,
        output_tokens_deterministic: bool,
        model_name: list = [],
        model_selection_strategy: ModelSelectionStrategy = ModelSelectionStrategy.ROUND_ROBIN,
    ) -> Dict:
        (
            system_role_headers,
            user_role_headers,
            text_input_headers,
        ) = cls._determine_json_feature_roles(dataset_json)
        pa_json = cls._populate_openai_completions_output_json(
            dataset_json,
            system_role_headers,
            user_role_headers,
            text_input_headers,
            add_model_name,
            add_stream,
            extra_inputs,
            output_tokens_mean,
            output_tokens_stddev,
            output_tokens_deterministic,
            model_name,
            model_selection_strategy,
        )

        return pa_json

    @classmethod
    def _convert_generic_json_to_openai_embeddings_format(
        cls,
        generic_dataset: Dict,
        extra_inputs: Dict,
        model_name: list = [],
        model_selection_strategy: ModelSelectionStrategy = ModelSelectionStrategy.ROUND_ROBIN,
    ) -> Dict[str, Any]:
        pa_json: Dict[str, Any] = {"data": []}

        for index, entry in enumerate(generic_dataset["rows"]):
            iter_model_name = cls._select_model_name(
                model_name, index, model_selection_strategy
            )
            payload = entry.get("payload", {})
            input_values = payload.get("input")

            if input_values is None:
                raise ValueError("Missing required fields 'input' in dataset entry")
            if not isinstance(input_values, list):
                raise ValueError(
                    f"Required field 'input' must be a list (actual: {type(input_values)})"
                )

            payload = {
                "input": input_values,
                "model": iter_model_name,
            }

            for key, value in extra_inputs.items():
                payload[key] = value

            pa_json["data"].append({"payload": [payload]})

        return pa_json

    @classmethod
    def contains_rankings_tei(cls, extra_inputs: Optional[Dict]) -> bool:
        """
        Check if user specified that they are using the Hugging Face
        Text Embeddings Interface for ranking models
        """
        if extra_inputs and extra_inputs.get("rankings") == "tei":
            return True
        return False

    @classmethod
    def _convert_generic_json_to_rankings_format(
        cls,
        generic_dataset: Dict,
        extra_inputs: Dict,
        model_name: list = [],
        model_selection_strategy: ModelSelectionStrategy = ModelSelectionStrategy.ROUND_ROBIN,
    ) -> Dict[str, Any]:
        pa_json: Dict[str, Any] = {"data": []}
        use_tei_format = cls.contains_rankings_tei(extra_inputs)

        for index, entry in enumerate(generic_dataset["rows"]):
            iter_model_name = cls._select_model_name(
                model_name, index, model_selection_strategy
            )
            payload = entry.get("payload", {})
            query_values = payload.get("query")

            if use_tei_format:
                passage_values = payload.get("passages", [])
                passage_values = [item.get("text", "") for item in passage_values]
            else:
                passage_values = payload.get("passages")

            if query_values is None:
                raise ValueError("Missing required fields 'query' in dataset entry")
            if passage_values is None:
                raise ValueError(
                    f"Missing required fields '{'texts' if use_tei_format else 'passages'}' in dataset entry"
                )
            if not isinstance(passage_values, list):
                raise ValueError(
                    f"Required field '{'texts' if use_tei_format else 'passages'}' must be a list (actual: {type(passage_values)})"
                )

            if use_tei_format:
                payload = {"query": query_values["text"], "texts": passage_values}
            else:
                payload = {
                    "query": query_values,
                    "passages": passage_values,
                    "model": iter_model_name,
                }

            for key, value in extra_inputs.items():
                if not (key == "rankings" and value == "tei"):
                    payload[key] = value

            pa_json["data"].append({"payload": [payload]})

        return pa_json

    @classmethod
    def _convert_generic_json_to_vllm_format(
        cls,
        dataset_json: Dict,
        add_model_name: bool,
        add_stream: bool,
        extra_inputs: Dict,
        output_tokens_mean: int,
        output_tokens_stddev: int,
        output_tokens_deterministic: bool,
        model_name: list = [],
        model_selection_strategy: ModelSelectionStrategy = ModelSelectionStrategy.ROUND_ROBIN,
    ) -> Dict:
        (
            system_role_headers,
            user_role_headers,
            text_input_headers,
        ) = cls._determine_json_feature_roles(dataset_json)

        pa_json = cls._populate_vllm_output_json(
            dataset_json,
            system_role_headers,
            user_role_headers,
            text_input_headers,
            add_model_name,
            add_stream,
            extra_inputs,
            output_tokens_mean,
            output_tokens_stddev,
            output_tokens_deterministic,
            model_name,
            model_selection_strategy,
        )

        return pa_json

    @classmethod
    def _convert_generic_json_to_trtllm_format(
        cls,
        dataset_json: Dict,
        add_model_name: bool,
        add_stream: bool,
        extra_inputs: Dict,
        output_tokens_mean: int,
        output_tokens_stddev: int,
        output_tokens_deterministic: bool,
        model_name: list = [],
        model_selection_strategy: ModelSelectionStrategy = ModelSelectionStrategy.ROUND_ROBIN,
    ) -> Dict:
        (
            system_role_headers,
            user_role_headers,
            text_input_headers,
        ) = cls._determine_json_feature_roles(dataset_json)

        pa_json = cls._populate_trtllm_output_json(
            dataset_json,
            system_role_headers,
            user_role_headers,
            text_input_headers,
            add_model_name,
            add_stream,
            extra_inputs,
            output_tokens_mean,
            output_tokens_stddev,
            output_tokens_deterministic,
            model_name,
            model_selection_strategy,
        )

        return pa_json

    @classmethod
    def _write_json_to_file(cls, json_in_pa_format: Dict, output_dir: Path) -> None:
        filename = output_dir / DEFAULT_INPUT_DATA_JSON
        with open(str(filename), "w") as f:
            f.write(json.dumps(json_in_pa_format, indent=2))

    @classmethod
    def _determine_json_feature_roles(
        cls, dataset_json: Dict
    ) -> Tuple[List[str], List[str], List[str]]:
        SYSTEM_ROLE_LIST = ["system_prompt"]
        USER_ROLE_LIST = ["question", "article"]
        TEXT_INPUT_LIST = ["text_input"]

        system_role_headers: List[str] = []
        user_role_headers: List[str] = []
        text_input_headers: List[str] = []

        if "features" in dataset_json.keys():
            # TODO (TPA-53) remove enumerate if index isnt useful
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
    def _select_model_name(cls, model_name, index, model_selection_strategy):
        if model_selection_strategy == ModelSelectionStrategy.ROUND_ROBIN:
            return model_name[index % len(model_name)]
        elif model_selection_strategy == ModelSelectionStrategy.RANDOM:
            return random.choice(model_name)
        else:
            raise GenAIPerfException(
                f"Model selection strategy '{model_selection_strategy}' is unsupported"
            )

    @classmethod
    def _populate_openai_chat_completions_output_json(
        cls,
        dataset_json: Dict,
        system_role_headers: List[str],
        user_role_headers: List[str],
        add_model_name: bool,
        add_stream: bool,
        extra_inputs: Dict,
        output_tokens_mean: int,
        output_tokens_stddev: int,
        output_tokens_deterministic: bool,
        model_name: list = [],
        model_selection_strategy: ModelSelectionStrategy = ModelSelectionStrategy.ROUND_ROBIN,
    ) -> Dict:
        pa_json = cls._create_empty_openai_pa_json()

        for index, entry in enumerate(dataset_json["rows"]):
            iter_model_name = cls._select_model_name(
                model_name, index, model_selection_strategy
            )
            pa_json["data"].append({"payload": []})
            pa_json["data"][index]["payload"].append({"messages": []})

            for header, content in entry.items():
                new_message = cls._create_new_openai_chat_completions_message(
                    header, system_role_headers, user_role_headers, content
                )

                pa_json = cls._add_new_message_to_json(pa_json, index, new_message)

            pa_json = cls._add_optional_tags_to_openai_json(
                pa_json,
                index,
                add_model_name,
                add_stream,
                extra_inputs,
                output_tokens_mean,
                output_tokens_stddev,
                output_tokens_deterministic,
                iter_model_name,
            )

        return pa_json

    @classmethod
    def _populate_openai_completions_output_json(
        cls,
        dataset_json: Dict,
        system_role_headers: List[str],
        user_role_headers: List[str],
        text_input_headers: List[str],
        add_model_name: bool,
        add_stream: bool,
        extra_inputs: Dict,
        output_tokens_mean: int,
        output_tokens_stddev: int,
        output_tokens_deterministic: bool,
        model_name: list = [],
        model_selection_strategy: ModelSelectionStrategy = ModelSelectionStrategy.ROUND_ROBIN,
    ) -> Dict:
        pa_json = cls._create_empty_openai_pa_json()

        for index, entry in enumerate(dataset_json["rows"]):
            iter_model_name = cls._select_model_name(
                model_name, index, model_selection_strategy
            )
            pa_json["data"].append({"payload": []})
            pa_json["data"][index]["payload"].append({"prompt": ""})

            for header, content in entry.items():
                new_prompt = cls._create_new_prompt(
                    header,
                    system_role_headers,
                    user_role_headers,
                    text_input_headers,
                    content,
                )

                pa_json = cls._add_new_prompt_to_json(pa_json, index, new_prompt)

            pa_json = cls._add_optional_tags_to_openai_json(
                pa_json,
                index,
                add_model_name,
                add_stream,
                extra_inputs,
                output_tokens_mean,
                output_tokens_stddev,
                output_tokens_deterministic,
                iter_model_name,
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
        extra_inputs: Dict,
        output_tokens_mean: int,
        output_tokens_stddev: int,
        output_tokens_deterministic: bool,
        model_name: list = [],
        model_selection_strategy: ModelSelectionStrategy = ModelSelectionStrategy.ROUND_ROBIN,
    ) -> Dict:
        pa_json = cls._create_empty_vllm_pa_json()

        for index, entry in enumerate(dataset_json["rows"]):
            iter_model_name = cls._select_model_name(
                model_name, index, model_selection_strategy
            )
            pa_json["data"].append({"text_input": [""]})

            for header, content in entry.items():
                new_text_input = cls._create_new_text_input(
                    header,
                    system_role_headers,
                    user_role_headers,
                    text_input_headers,
                    content,
                )

                pa_json = cls._add_new_text_input_to_json(
                    pa_json, index, new_text_input
                )

            pa_json = cls._add_optional_tags_to_vllm_json(
                pa_json,
                index,
                add_model_name,
                add_stream,
                extra_inputs,
                output_tokens_mean,
                output_tokens_stddev,
                output_tokens_deterministic,
                iter_model_name,
            )

        return pa_json

    @classmethod
    def _populate_trtllm_output_json(
        cls,
        dataset_json: Dict,
        system_role_headers: List[str],
        user_role_headers: List[str],
        text_input_headers: List[str],
        add_model_name: bool,
        add_stream: bool,
        extra_inputs: Dict,
        output_tokens_mean: int,
        output_tokens_stddev: int,
        output_tokens_deterministic: bool,
        model_name: list = [],
        model_selection_strategy: ModelSelectionStrategy = ModelSelectionStrategy.ROUND_ROBIN,
    ) -> Dict:
        pa_json = cls._create_empty_trtllm_pa_json()
        default_max_tokens = (
            "max_tokens" not in extra_inputs
            or output_tokens_mean != cls.DEFAULT_OUTPUT_TOKENS_MEAN
        )

        for index, entry in enumerate(dataset_json["rows"]):
            iter_model_name = cls._select_model_name(
                model_name, index, model_selection_strategy
            )
            pa_json["data"].append({"text_input": [""]})

            for header, content in entry.items():
                new_text_input = cls._create_new_text_input(
                    header,
                    system_role_headers,
                    user_role_headers,
                    text_input_headers,
                    content,
                )

                pa_json = cls._add_new_text_input_to_json(
                    pa_json, index, new_text_input
                )

            pa_json = cls._add_required_tags_to_trtllm_json(
                pa_json, index, default_max_tokens
            )
            pa_json = cls._add_optional_tags_to_trtllm_json(
                pa_json,
                index,
                add_model_name,
                add_stream,
                extra_inputs,
                output_tokens_mean,
                output_tokens_stddev,
                output_tokens_deterministic,
                iter_model_name,
            )

        return pa_json

    @classmethod
    def _create_empty_openai_pa_json(cls) -> Dict:
        empty_pa_json = deepcopy(cls.EMPTY_JSON_IN_OPENAI_PA_FORMAT)

        return empty_pa_json

    @classmethod
    def _create_empty_vllm_pa_json(cls) -> Dict:
        empty_pa_json = deepcopy(cls.EMPTY_JSON_IN_VLLM_PA_FORMAT)

        return empty_pa_json

    @classmethod
    def _create_empty_trtllm_pa_json(cls) -> Dict:
        empty_pa_json = deepcopy(cls.EMPTY_JSON_IN_TENSORRTLLM_PA_FORMAT)

        return empty_pa_json

    @classmethod
    def _create_new_openai_chat_completions_message(
        cls,
        header: str,
        system_role_headers: List[str],
        user_role_headers: List[str],
        content: str,
    ) -> Optional[Dict]:
        # Do not add messages with blank content
        if not content:
            return {}

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
    def _create_new_prompt(
        cls,
        header: str,
        system_role_headers: List[str],
        user_role_headers: List[str],
        text_input_headers: List[str],
        content: str,
    ) -> str:
        new_prompt = ""

        if (
            header in system_role_headers
            or header in user_role_headers
            or header in text_input_headers
        ):
            new_prompt = content

        return new_prompt

    @classmethod
    def _create_new_text_input(
        cls,
        header: str,
        system_role_headers: List[str],
        user_role_headers: List[str],
        text_input_headers: List[str],
        content: str,
    ) -> str:
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
            pa_json["data"][index]["payload"][0]["messages"].append(new_message)

        return pa_json

    @classmethod
    def _add_new_text_input_to_json(
        cls, pa_json: Dict, index: int, new_text_input: str
    ) -> Dict:
        if new_text_input:
            if pa_json["data"][index]["text_input"][0]:
                pa_json["data"][index]["text_input"][0] = (
                    pa_json["data"][index]["text_input"][0] + f" {new_text_input}"
                )
            else:
                pa_json["data"][index]["text_input"][0] = new_text_input

        return pa_json

    @classmethod
    def _add_new_prompt_to_json(
        cls,
        pa_json: Dict,
        index: int,
        new_prompt: str,
    ) -> Dict:
        if new_prompt:
            if pa_json["data"][index]["payload"][0]["prompt"]:
                pa_json["data"][index]["payload"][0]["prompt"] += f" {new_prompt}"
            else:
                pa_json["data"][index]["payload"][0]["prompt"] = new_prompt

        return pa_json

    @classmethod
    def _add_optional_tags_to_openai_json(
        cls,
        pa_json: Dict,
        index: int,
        add_model_name: bool,
        add_stream: bool,
        extra_inputs: Dict,
        output_tokens_mean: int,
        output_tokens_stddev: int,
        output_tokens_deterministic: bool,
        model_name: str = "",
    ) -> Dict:
        row = pa_json["data"][index]["payload"][0]
        if add_model_name:
            row["model"] = model_name
        if add_stream:
            row["stream"] = True
        if output_tokens_mean != cls.DEFAULT_OUTPUT_TOKENS_MEAN:
            row["max_tokens"] = int(
                random.gauss(output_tokens_mean, output_tokens_stddev)
            )
        for key, value in extra_inputs.items():
            row[key] = value

        return pa_json

    @classmethod
    def _add_optional_tags_to_vllm_json(
        cls,
        pa_json: Dict,
        index: int,
        add_model_name: bool,
        add_stream: bool,
        extra_inputs: Dict,
        output_tokens_mean: int,
        output_tokens_stddev: int,
        output_tokens_deterministic: bool,
        model_name: str = "",
    ) -> Dict:
        row = pa_json["data"][index]
        if add_model_name:
            row["model"] = model_name
        if add_stream:
            row["stream"] = [True]
        if output_tokens_mean != cls.DEFAULT_OUTPUT_TOKENS_MEAN:
            number_of_tokens = str(
                int(max(0, random.gauss(output_tokens_mean, output_tokens_stddev)))
            )
            sampling_parameters = {
                "max_tokens": number_of_tokens,
            }
            if output_tokens_deterministic:
                sampling_parameters["min_tokens"] = number_of_tokens
            sampling_parameters_str = json.dumps(sampling_parameters)
            row["sampling_parameters"] = [sampling_parameters_str]
        for key, value in extra_inputs.items():
            row[key] = [value]
        if "exclude_input_in_output" not in row:
            row["exclude_input_in_output"] = [True]

        return pa_json

    @classmethod
    def _add_optional_tags_to_trtllm_json(
        cls,
        pa_json: Dict,
        index: int,
        add_model_name: bool,
        add_stream: bool,
        extra_inputs: Dict,
        output_tokens_mean: int,
        output_tokens_stddev: int,
        output_tokens_deterministic: bool,
        model_name: str = "",
    ) -> Dict:
        row = pa_json["data"][index]
        if add_model_name:
            row["model"] = model_name
        if add_stream:
            row["stream"] = [True]
        if output_tokens_mean != cls.DEFAULT_OUTPUT_TOKENS_MEAN:
            number_of_tokens = int(
                random.gauss(output_tokens_mean, output_tokens_stddev)
            )
            if output_tokens_deterministic:
                row["min_length"] = [number_of_tokens]
            row["max_tokens"] = [number_of_tokens]
        for key, value in extra_inputs.items():
            row[key] = [value]

        return pa_json

    @classmethod
    def _add_required_tags_to_trtllm_json(
        cls,
        pa_json: Dict,
        index: int,
        default_max_tokens: bool,
    ) -> Dict:
        row = pa_json["data"][index]
        if default_max_tokens:
            row["max_tokens"] = [cls.DEFAULT_TENSORRTLLM_MAX_TOKENS]

        return pa_json

    @classmethod
    def _check_for_dataset_name_if_input_type_is_url(
        cls, input_type: PromptSource, dataset_name: str
    ) -> None:
        if input_type == PromptSource.DATASET and not dataset_name:
            raise GenAIPerfException(
                "Input type is dataset, but dataset_name is not specified."
            )

    @classmethod
    def _check_for_tokenzier_if_input_type_is_synthetic(
        cls,
        input_type: PromptSource,
        tokenizer: Tokenizer,
    ) -> None:
        if input_type == PromptSource.SYNTHETIC and not tokenizer:
            raise GenAIPerfException(
                "Input type is SYNTHETIC, but a tokenizer was not specified."
            )

    @classmethod
    def _check_for_valid_starting_index(cls, starting_index: int) -> None:
        if not isinstance(starting_index, int):
            raise GenAIPerfException(
                f"starting_index: {starting_index} must be an integer."
            )

        if starting_index < cls.MINIMUM_STARTING_INDEX:
            raise GenAIPerfException(
                f"starting_index: {starting_index} must be larger than {cls.MINIMUM_STARTING_INDEX}."
            )

    @classmethod
    def _check_for_valid_length(cls, length: int) -> None:
        if not isinstance(length, int):
            raise GenAIPerfException(f"length: {length} must be an integer.")

        if length < cls.MINIMUM_LENGTH:
            raise GenAIPerfException(
                f"starting_index: {length} must be larger than {cls.MINIMUM_LENGTH}."
            )

    @classmethod
    def _query_server(cls, configured_url: str) -> Response:
        try:
            response = requests.get(configured_url)
        except Exception as e:
            error_message = cls._create_error_message(e)
            raise GenAIPerfException(error_message)

        return response

    @classmethod
    def _create_error_message(cls, exception: Exception) -> str:
        url_str = exception.args[0].args[0]
        url_start = url_str.find("'")
        url_end = url_str.find("'", url_start + 1) + 1
        error_message = f"Invalid URL: {url_str[url_start:url_end]}"

        return error_message

    @classmethod
    def _check_for_error_in_json_of_dataset(cls, dataset_json: Dict) -> None:
        if "error" in dataset_json:
            raise GenAIPerfException(dataset_json["error"])

    @classmethod
    def _create_synthetic_prompt(
        cls,
        tokenizer: Tokenizer,
        prompt_tokens_mean: int,
        prompt_tokens_stddev: int,
    ) -> str:
        return SyntheticPromptGenerator.create_synthetic_prompt(
            tokenizer, prompt_tokens_mean, prompt_tokens_stddev
        )

    @classmethod
    def _create_synthetic_image(
        cls,
        image_width_mean: int,
        image_width_stddev: int,
        image_height_mean: int,
        image_height_stddev: int,
        image_format: ImageFormat,
    ) -> str:
        return SyntheticImageGenerator.create_synthetic_image(
            image_width_mean=image_width_mean,
            image_width_stddev=image_width_stddev,
            image_height_mean=image_height_mean,
            image_height_stddev=image_height_stddev,
            image_format=image_format,
        )
