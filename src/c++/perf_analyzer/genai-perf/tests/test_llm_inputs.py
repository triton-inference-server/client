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
import random
import statistics
from pathlib import Path

import pytest
from genai_perf import tokenizer
from genai_perf.constants import CNN_DAILY_MAIL, DEFAULT_INPUT_DATA_JSON, OPEN_ORCA
from genai_perf.exceptions import GenAIPerfException
from genai_perf.llm_inputs.llm_inputs import (
    LlmInputs,
    ModelSelectionStrategy,
    OutputFormat,
    PromptSource,
)
from genai_perf.tokenizer import Tokenizer


class TestLlmInputs:
    # Define service kind, backend or api, and output format combinations
    SERVICE_KIND_BACKEND_ENDPOINT_TYPE_FORMATS = [
        ("triton", "vllm", OutputFormat.VLLM),
        ("triton", "tensorrtllm", OutputFormat.TENSORRTLLM),
        ("openai", "v1/completions", OutputFormat.OPENAI_COMPLETIONS),
        ("openai", "v1/chat/completions", OutputFormat.OPENAI_CHAT_COMPLETIONS),
    ]

    @pytest.fixture
    def default_configured_url(self):
        default_configured_url = LlmInputs._create_configured_url(
            LlmInputs.OPEN_ORCA_URL,
            LlmInputs.DEFAULT_STARTING_INDEX,
            LlmInputs.DEFAULT_LENGTH,
        )

        yield default_configured_url

    # TODO (TMA-1754): Add tests that verify json schemas
    @pytest.fixture(scope="class")
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
                input_type=PromptSource.SYNTHETIC, tokenizer=None  # type: ignore
            )

    def test_illegal_starting_index(self):
        """
        Test for exceptions when illegal values are given for starting index
        """
        with pytest.raises(GenAIPerfException):
            _ = LlmInputs._check_for_valid_starting_index(starting_index="foo")  # type: ignore

        with pytest.raises(GenAIPerfException):
            _ = LlmInputs._check_for_valid_starting_index(starting_index=-1)

    def test_illegal_length(self):
        """
        Test for exceptions when illegal values are given for length
        """
        with pytest.raises(GenAIPerfException):
            _ = LlmInputs._check_for_valid_length(length="foo")  # type: ignore

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
        )
        dataset_json = LlmInputs._convert_input_url_dataset_to_generic_json(
            dataset=dataset
        )
        pa_json = LlmInputs._convert_generic_json_to_output_format(
            output_format=OutputFormat.OPENAI_CHAT_COMPLETIONS,
            generic_dataset=dataset_json,
            add_model_name=False,
            add_stream=False,
            extra_inputs={},
            output_tokens_mean=LlmInputs.DEFAULT_OUTPUT_TOKENS_MEAN,
            output_tokens_stddev=LlmInputs.DEFAULT_OUTPUT_TOKENS_STDDEV,
            output_tokens_deterministic=False,
            model_name=["test_model_A"],
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
            model_name=["test_model_A"],
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
            with open(DEFAULT_INPUT_DATA_JSON, "r") as f:
                json_str = f.read()
        finally:
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
            model_name=["test_model_A"],
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
            model_name=["test_model_A"],
        )

        os.remove(DEFAULT_INPUT_DATA_JSON)

        assert pa_json is not None
        assert len(pa_json["data"]) == LlmInputs.DEFAULT_LENGTH
        # NIM legacy completion endpoint only supports string and not
        # array of strings. Verify that the prompt is of type string
        # not list
        assert isinstance(pa_json["data"][0]["payload"][0]["prompt"], str)

    def test_create_openai_to_trtllm(self):
        """
        Test conversion of openai to trtllm
        """
        pa_json = LlmInputs.create_llm_inputs(
            input_type=PromptSource.DATASET,
            output_format=OutputFormat.TENSORRTLLM,
            dataset_name=OPEN_ORCA,
            add_model_name=False,
            add_stream=True,
            model_name=["test_model_A"],
        )

        os.remove(DEFAULT_INPUT_DATA_JSON)

        assert pa_json is not None
        assert len(pa_json["data"]) == LlmInputs.DEFAULT_LENGTH

    def test_random_synthetic_no_stddev(self, default_tokenizer):
        """
        Test that we can produce an exact number of random synthetic tokens
        """
        random.seed(1)

        def _subtest(token_length):
            synthetic_prompt = LlmInputs._create_synthetic_prompt(
                tokenizer=default_tokenizer,
                prompt_tokens_mean=token_length,
                prompt_tokens_stddev=0,
            )

            actual_token_length = len(default_tokenizer.encode(synthetic_prompt))
            assert token_length == actual_token_length

        # Test all of 500-600 to make sure exact
        for i in range(500, 600):
            _subtest(i)

        # Test some larger values
        _subtest(1500)
        _subtest(10000)

    def test_random_synthetic_stddev(self, default_tokenizer):
        """
        Test that we can produce random synthetic tokens within a requested stddev
        """
        random.seed(1)

        def _subtest(num_samples, mean, stddev):
            prompt_tokens = []
            for _ in range(num_samples):
                prompt = LlmInputs._create_synthetic_prompt(
                    tokenizer=default_tokenizer,
                    prompt_tokens_mean=mean,
                    prompt_tokens_stddev=stddev,
                )
                prompt_tokens.append(len(default_tokenizer.encode(prompt)))

            assert statistics.mean(prompt_tokens) == pytest.approx(mean, rel=0.1)
            assert statistics.stdev(prompt_tokens) == pytest.approx(stddev, rel=0.2)

        _subtest(50, 200, 20)
        _subtest(50, 400, 10)
        _subtest(200, 50, 10)

    def test_random_seed(self, default_tokenizer):
        """
        Test that when given the same seed, create_llm_inputs will return the same result,
        and that when given a different seed, it will produce a different result
        """

        inputs_seed5_a = LlmInputs.create_llm_inputs(
            tokenizer=default_tokenizer,
            input_type=PromptSource.SYNTHETIC,
            output_format=OutputFormat.TENSORRTLLM,
            prompt_tokens_mean=300,
            prompt_tokens_stddev=20,
            num_of_output_prompts=5,
            random_seed=5,
            model_name=["test_model_A"],
        )

        inputs_seed5_b = LlmInputs.create_llm_inputs(
            tokenizer=default_tokenizer,
            input_type=PromptSource.SYNTHETIC,
            output_format=OutputFormat.TENSORRTLLM,
            prompt_tokens_mean=300,
            prompt_tokens_stddev=20,
            num_of_output_prompts=5,
            random_seed=5,
            model_name=["test_model_A"],
        )

        inputs_seed10 = LlmInputs.create_llm_inputs(
            tokenizer=default_tokenizer,
            input_type=PromptSource.SYNTHETIC,
            output_format=OutputFormat.TENSORRTLLM,
            prompt_tokens_mean=300,
            prompt_tokens_stddev=20,
            num_of_output_prompts=5,
            random_seed=10,
            model_name=["test_model_A"],
        )

        assert inputs_seed5_a == inputs_seed5_b
        assert inputs_seed5_a != inputs_seed10

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
            model_name=["test_model_A"],
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
            output_format=OutputFormat.TENSORRTLLM,
            num_of_output_prompts=5,
            add_model_name=False,
            add_stream=True,
            tokenizer=default_tokenizer,
            model_name=["test_model_A"],
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
            model_name=["test_model_A"],
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
            model_name=["test_model_A"],
        )

        os.remove(DEFAULT_INPUT_DATA_JSON)

        assert pa_json is not None
        assert len(pa_json["data"]) == 5

    @pytest.mark.parametrize(
        "output_format",
        [format[2] for format in SERVICE_KIND_BACKEND_ENDPOINT_TYPE_FORMATS],
    )
    def test_extra_inputs(
        self, default_tokenizer: Tokenizer, output_format: OutputFormat
    ) -> None:
        input_name = "max_tokens"
        input_value = 5
        request_inputs = {input_name: input_value}

        pa_json = LlmInputs.create_llm_inputs(
            input_type=PromptSource.SYNTHETIC,
            output_format=output_format,
            num_of_output_prompts=5,
            add_model_name=False,
            add_stream=True,
            tokenizer=default_tokenizer,
            extra_inputs=request_inputs,
            model_name=["test_model_A"],
        )

        assert len(pa_json["data"]) == 5

        if (
            output_format == OutputFormat.OPENAI_CHAT_COMPLETIONS
            or output_format == OutputFormat.OPENAI_COMPLETIONS
        ):
            for entry in pa_json["data"]:
                assert "payload" in entry, "Payload is missing in the request"
                payload = entry["payload"]
                for item in payload:
                    assert (
                        input_name in item
                    ), f"The input name {input_name} is not present in the request"
                    assert (
                        item[input_name] == input_value
                    ), f"The value of {input_name} is incorrect"
        elif (
            output_format == OutputFormat.TENSORRTLLM
            or output_format == OutputFormat.VLLM
        ):
            for entry in pa_json["data"]:
                assert (
                    input_name in entry
                ), f"The {input_name} is not present in the request"
                assert entry[input_name] == [
                    input_value
                ], f"The value of {input_name} is incorrect"
        else:
            assert False, f"Unsupported output format: {output_format}"

    def test_trtllm_default_max_tokens(self, default_tokenizer: Tokenizer) -> None:
        input_name = "max_tokens"
        input_value = 256

        pa_json = LlmInputs.create_llm_inputs(
            input_type=PromptSource.SYNTHETIC,
            output_format=OutputFormat.TENSORRTLLM,
            num_of_output_prompts=5,
            add_model_name=False,
            add_stream=True,
            tokenizer=default_tokenizer,
            model_name=["test_model_A"],
        )

        assert len(pa_json["data"]) == 5
        for entry in pa_json["data"]:
            assert (
                input_name in entry
            ), f"The {input_name} is not present in the request"
            assert entry[input_name] == [
                input_value
            ], f"The value of {input_name} is incorrect"

    @pytest.mark.parametrize(
        "output_format",
        [format[2] for format in SERVICE_KIND_BACKEND_ENDPOINT_TYPE_FORMATS],
    )
    def test_output_tokens_mean(self, output_format, default_tokenizer):
        if (
            output_format != OutputFormat.VLLM
            and output_format != OutputFormat.TENSORRTLLM
        ):
            return

        output_tokens_mean = 100
        output_tokens_stddev = 0
        for deterministic in [True, False]:
            _ = LlmInputs.create_llm_inputs(
                input_type=PromptSource.SYNTHETIC,
                output_format=output_format,
                num_of_output_prompts=5,
                add_model_name=False,
                add_stream=True,
                tokenizer=default_tokenizer,
                output_tokens_mean=output_tokens_mean,
                output_tokens_stddev=output_tokens_stddev,
                output_tokens_deterministic=deterministic,
                model_name=["test_model_A"],
            )

            assert os.path.exists(
                DEFAULT_INPUT_DATA_JSON
            ), "llm_inputs.json file is not created"

            with open(DEFAULT_INPUT_DATA_JSON, "r") as f:
                llm_inputs_data = json.load(f)

            for entry in llm_inputs_data["data"]:
                if output_format == OutputFormat.VLLM:
                    assert (
                        "sampling_parameters" in entry
                    ), "sampling_parameters is missing in llm_inputs.json"
                    sampling_parameters = json.loads(entry["sampling_parameters"][0])
                    assert (
                        "max_tokens" in sampling_parameters
                    ), "max_tokens parameter is missing in sampling_parameters"
                    assert sampling_parameters["max_tokens"] == str(
                        output_tokens_mean
                    ), "max_tokens parameter is not properly set"
                    if deterministic:
                        assert (
                            "min_tokens" in sampling_parameters
                        ), "min_tokens parameter is missing in sampling_parameters"
                        assert sampling_parameters["min_tokens"] == str(
                            output_tokens_mean
                        ), "min_tokens parameter is not properly set"
                    else:
                        assert (
                            "min_tokens" not in sampling_parameters
                        ), "min_tokens parameter is present in sampling_parameters"
                elif output_format == OutputFormat.TENSORRTLLM:
                    assert (
                        "max_tokens" in entry
                    ), "max_tokens parameter is missing in llm_inputs.json"
                    assert (
                        entry["max_tokens"][0] == output_tokens_mean
                    ), "max_tokens parameter is not properly set"
                    if deterministic:
                        assert (
                            "min_length" in entry
                        ), "min_length parameter is missing in llm_inputs.json"
                        assert (
                            entry["min_length"][0] == output_tokens_mean
                        ), "min_length parameter is not properly set"
                    else:
                        assert (
                            "min_length" not in entry
                        ), "min_length parameter is present in llm_inputs.json"
                else:
                    assert False, f"Unsupported output format: {output_format}"

            os.remove(DEFAULT_INPUT_DATA_JSON)

    def test_get_input_file_without_file_existing(self):
        with pytest.raises(FileNotFoundError):
            LlmInputs._get_input_dataset_from_file(Path("prompt.txt"))

    @pytest.mark.parametrize(
        "seed, model_name_list, index,model_selection_strategy,expected_model",
        [
            (
                1,
                ["test_model_A", "test_model_B", "test_model_C"],
                0,
                ModelSelectionStrategy.ROUND_ROBIN,
                "test_model_A",
            ),
            (
                1,
                ["test_model_A", "test_model_B", "test_model_C"],
                1,
                ModelSelectionStrategy.ROUND_ROBIN,
                "test_model_B",
            ),
            (
                1,
                ["test_model_A", "test_model_B", "test_model_C"],
                2,
                ModelSelectionStrategy.ROUND_ROBIN,
                "test_model_C",
            ),
            (
                1,
                ["test_model_A", "test_model_B", "test_model_C"],
                3,
                ModelSelectionStrategy.ROUND_ROBIN,
                "test_model_A",
            ),
            (
                100,
                ["test_model_A", "test_model_B", "test_model_C"],
                0,
                ModelSelectionStrategy.RANDOM,
                "test_model_A",
            ),
            (
                100,
                ["test_model_A", "test_model_B", "test_model_C"],
                1,
                ModelSelectionStrategy.RANDOM,
                "test_model_A",
            ),
            (
                1652,
                ["test_model_A", "test_model_B", "test_model_C"],
                0,
                ModelSelectionStrategy.RANDOM,
                "test_model_B",
            ),
            (
                95,
                ["test_model_A", "test_model_B", "test_model_C"],
                0,
                ModelSelectionStrategy.RANDOM,
                "test_model_C",
            ),
        ],
    )
    def test_select_model_name(
        self, seed, model_name_list, index, model_selection_strategy, expected_model
    ):
        """
        Test that model selection strategy controls the model selected
        """
        random.seed(seed)

        actual_model = LlmInputs._select_model_name(
            model_name_list, index, model_selection_strategy
        )
        assert actual_model == expected_model
