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

import itertools
import math
import pathlib
import random
import re
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
        prompt = SyntheticPromptGenerator._create_prompt_from_lines(
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
    def _create_prompt_from_lines(
        cls,
        requested_prompt_tokens: int,
        source_lines: List[str],
        tokenizer: Tokenizer,
    ) -> str:
        get_token_length = lambda text: len(tokenizer.encode(text))

        line_iterator = itertools.cycle(source_lines)

        def word_generator():
            while True:
                next_line = next(line_iterator)
                words = re.split("[ \n]+", next_line)
                for word in words:
                    yield word

        word_iterator = word_generator()

        # Fast add lines
        remaining_tokens = requested_prompt_tokens
        prompt = ""
        num_tokens_in_avg_line = get_token_length(source_lines[0] + source_lines[1]) / 2
        num_lines_to_add_fast = math.floor(
            0.5 * requested_prompt_tokens / num_tokens_in_avg_line
        )
        while num_lines_to_add_fast:
            for _ in range(num_lines_to_add_fast):
                next_line = next(line_iterator)
                prompt = prompt + next_line

            curr_tokens = get_token_length(prompt)
            remaining_tokens = requested_prompt_tokens - curr_tokens
            num_lines_to_add_fast = math.floor(
                0.5 * remaining_tokens / num_tokens_in_avg_line
            )

        # Fast add words
        final_line = ""
        while get_token_length(final_line) < remaining_tokens - 3:
            next_word = next(word_iterator)
            final_line += next_word + " "
        prompt += final_line

        # Final tweaks
        diff = requested_prompt_tokens - get_token_length(prompt)
        for _ in range(diff):
            prompt = "hi " + prompt

        return prompt

    @classmethod
    def _sample_random_positive_int(cls, mean: int, stddev: int) -> int:
        random_pos_int = -1
        while random_pos_int <= 0:
            random_pos_int = int(random.gauss(mean, stddev))

        return random_pos_int
