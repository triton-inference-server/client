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
from typing import List

from genai_perf.tokenizer import Tokenizer


class SyntheticPromptGenerator:
    @classmethod
    def create_synthetic_prompt(
        cls,
        tokenizer: Tokenizer,
        prompt_tokens_mean: int = 550,
        prompt_tokens_stddev: int = 250,
    ) -> str:
        """
        Generate a prompt that randomly samples lines from
        Washington's farewell address at farewell.txt.

        Args:
            prompt_tokens_mean:
                The mean length of the prompt to generate
            prompt_tokens_stddev:
                The standard deviation of the length of the prompt to generate

        Returns:
            The prompt.
        """

        num_prompt_tokens = SyntheticPromptGenerator._sample_random_positive_int(
            prompt_tokens_mean, prompt_tokens_stddev
        )

        farewell_lines = SyntheticPromptGenerator._create_farewell_lines()
        prompt = SyntheticPromptGenerator._create_prompt_from_farewell_lines(
            num_prompt_tokens, farewell_lines, tokenizer
        )

        return prompt

    @classmethod
    def _create_farewell_lines(cls) -> List[str]:
        farewell_path = pathlib.Path(__file__).parent.resolve() / "farewell.txt"
        with open(farewell_path, "r") as f:
            farewell_lines = f.readlines()
        random.shuffle(farewell_lines)

        return farewell_lines

    @classmethod
    def _create_prompt_from_farewell_lines(
        cls,
        remaining_prompt_tokens: int,
        farewell_lines: List[str],
        tokenizer: Tokenizer,
    ) -> str:
        prompt_tokens = remaining_prompt_tokens
        prompt = ""
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
                remaining_prompt_tokens = prompt_tokens - get_token_length(prompt)

        return prompt

    @classmethod
    def _sample_random_positive_int(cls, mean: int, stddev: int) -> int:
        random_pos_int = -1
        while random_pos_int <= 0:
            random_pos_int = int(random.gauss(mean, stddev))

        return random_pos_int
