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

import contextlib
import io
import math
import pathlib
import random
from typing import List, Tuple

# Silence tokenizer warning on import
with contextlib.redirect_stdout(io.StringIO()) as stdout, contextlib.redirect_stderr(
    io.StringIO()
) as stderr:
    from transformers import LlamaTokenizerFast

# TODO (TMA-1718): This should be passed in (and should not be in bare code)
tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")


class SyntheticPromptGenerator:
    @classmethod
    def create_synthetic_prompt(
        cls,
        prompt_tokens_mean: int = 550,
        prompt_tokens_stddev: int = 250,
        expected_output_tokens: int = 150,
    ) -> Tuple[str, int]:
        """
        Generate a prompt that randomly samples lines from
        Washington's farewell address at farewell.txt.

        Args:
            prompt_tokens_mean:
                The mean length of the prompt to generate
            prompt_tokens_stddev:
                The standard deviation of the length of the prompt to generate
            expected_output_tokens:
                The number of tokens to expect in the output. This is used to
                determine the length of the prompt. The prompt will be generated such that the output
                will be approximately this many tokens.

        Returns:
            A tuple of the prompt and the length of the prompt.
        """

        prompt = (
            "Randomly stream lines from the following text "
            f"with {expected_output_tokens} output tokens. "
            "Don't generate eos tokens:\n\n"
        )

        prompt_token_length = SyntheticPromptGenerator._get_prompt_token_length(prompt)
        num_prompt_tokens = SyntheticPromptGenerator._get_num_prompt_tokens(
            prompt_tokens_mean, prompt_tokens_stddev, prompt_token_length
        )
        remaining_prompt_tokens = num_prompt_tokens - prompt_token_length

        farewell_lines = SyntheticPromptGenerator._create_farewell_lines()
        prompt = SyntheticPromptGenerator._create_prompt_from_farewell_lines(
            prompt, remaining_prompt_tokens, farewell_lines
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
    def _create_farewell_lines(cls) -> List[str]:
        farewell_path = pathlib.Path(__file__).parent.resolve() / "farewell.txt"
        with open(farewell_path, "r") as f:
            farewell_lines = f.readlines()
        random.shuffle(farewell_lines)

        return farewell_lines

    @classmethod
    def _create_prompt_from_farewell_lines(
        cls, prompt: str, remaining_prompt_tokens: int, farewell_lines: List[str]
    ) -> str:
        tokenizer = SyntheticPromptGenerator._get_tokenizer()
        get_token_length = lambda text: len(tokenizer.encode(text))

        sampling_lines = True
        while sampling_lines:
            for line in farewell_lines:
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
        return tokenizer
