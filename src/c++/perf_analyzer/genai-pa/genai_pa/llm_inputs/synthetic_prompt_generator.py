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

import math
import pathlib
import random
from typing import Any, Dict, List, Tuple

from transformers import LlamaTokenizerFast


class SyntheticPromptGenerator:
    @classmethod
    def create_synthetic_prompt(
        cls,
        prompt_tokens_mean: int = 550,
        prompt_tokens_stddev: int = 250,
        expect_output_tokens: int = 150,
    ) -> Tuple[str, int]:
        """
        Generate a prompt that randomly samples lines from a the sonnet at sonnet.txt.

        Args:
            prompt_length_mean:
                The mean length of the prompt to generate
            prompt_len_stddev:
                The standard deviation of the length of the prompt to generate
            expect_output_tokens:
                The number of tokens to expect in the output. This is used to
                determine the length of the prompt. The prompt will be generated such that the output
                will be approximately this many tokens.

        Note:
            Tokens will be counted from the sonnet using the Llama tokenizer. Using one tokenizer
            ensures a fairer comparison across different LLMs. For example, if gpt 3.5 tokenizes
            a prompt in less tokens than Llama2, then this will be reflected in the results since
            they will be fed identical prompts.

        Returns:
            A tuple of the prompt and the length of the prompt.
        """

        prompt = (
            "Randomly stream lines from the following text "
            f"with {expect_output_tokens} output tokens. "
            "Don't generate eos tokens:\n\n"
        )

        prompt_token_length = SyntheticPromptGenerator._get_prompt_token_length(prompt)
        num_prompt_tokens = SyntheticPromptGenerator._get_num_prompt_tokens(
            prompt_tokens_mean, prompt_tokens_stddev, prompt_token_length
        )
        remaining_prompt_tokens = num_prompt_tokens - prompt_token_length

        sonnet_lines = SyntheticPromptGenerator._create_sonnet_lines()
        prompt = SyntheticPromptGenerator._create_prompt_from_sonnet_lines(
            prompt, remaining_prompt_tokens, sonnet_lines
        )

        return (prompt, num_prompt_tokens)

    @classmethod
    def _get_prompt_token_length(cls, prompt: str) -> int:
        tokenizer = SyntheticPromptGenerator._get_tokenizer()
        get_token_length = lambda text: len(tokenizer.encode(text))

        prompt_token_length = get_token_length(prompt)

        return prompt_token_length

    @classmethod
    def _get_num_prompt_tokens(
        cls, mean: int, stddev: int, prompt_token_length: int
    ) -> int:
        num_prompt_tokens = SyntheticPromptGenerator._sample_random_positive_int(
            mean, stddev
        )
        # Ensure prompt length is at least as long as the base
        while num_prompt_tokens < prompt_token_length:
            num_prompt_tokens = SyntheticPromptGenerator._sample_random_positive_int(
                prompt_tokens_mean, prompt_tokens_stddev
            )

        return num_prompt_tokens

    @classmethod
    def _create_sonnet_lines(cls) -> List[str]:
        sonnet_path = pathlib.Path(__file__).parent.resolve() / "sonnet.txt"
        with open(sonnet_path, "r") as f:
            sonnet_lines = f.readlines()
        random.shuffle(sonnet_lines)

        return sonnet_lines

    @classmethod
    def _create_prompt_from_sonnet_lines(
        cls, prompt: str, remaining_prompt_tokens: int, sonnet_lines: List[str]
    ) -> str:
        tokenizer = SyntheticPromptGenerator._get_tokenizer()
        get_token_length = lambda text: len(tokenizer.encode(text))

        sampling_lines = True
        while sampling_lines:
            for line in sonnet_lines:
                line_to_add = line
                if remaining_prompt_tokens - get_token_length(line_to_add) < 0:
                    # This will cut off a line in the middle of a word, but that's ok since an
                    # llm should be able to handle that.
                    line_to_add = line_to_add[: int(math.ceil(remaining_prompt_tokens))]
                    sampling_lines = False
                    prompt += line_to_add
                    break
                prompt += line_to_add
                remaining_prompt_tokens -= get_token_length(line_to_add)

        return prompt

    @classmethod
    def _sample_random_positive_int(cls, mean: int, stddev: int) -> int:
        random_pos_int = -1
        while random_pos_int <= 0:
            random_pos_int = int(random.gauss(mean, stddev))

        return random_pos_int

    @classmethod
    def _get_tokenizer(cls) -> LlamaTokenizerFast:
        tokenizer = LlamaTokenizerFast.from_pretrained(
            "hf-internal-testing/llama-tokenizer"
        )

        return tokenizer
