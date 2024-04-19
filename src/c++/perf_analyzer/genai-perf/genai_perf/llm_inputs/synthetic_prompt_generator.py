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
        requested_prompt_tokens = remaining_prompt_tokens
        prompt = ""
        get_token_length = lambda text: len(tokenizer.encode(text))

        sampling_lines = True
        while sampling_lines:
            for line in farewell_lines:
                line_to_add = line

                # If the line is too big, we are near the boundary and need to
                # go slow and potentially add one word at a time
                if remaining_prompt_tokens - get_token_length(line_to_add) < 0:
                    # First check if we are really over the boundary. Adding
                    # the line may actually be less tokens than the sum of the
                    # individual parts
                    proposed_prompt = prompt + line_to_add

                    if get_token_length(proposed_prompt) == requested_prompt_tokens:
                        prompt = proposed_prompt
                        sampling_lines = False
                        break

                    # Now add one word at a time
                    words_to_add = line_to_add.split()
                    for word in words_to_add:
                        proposed_prompt = prompt + word + " "

                        num_tokens = get_token_length(proposed_prompt)
                        if num_tokens > requested_prompt_tokens:
                            while get_token_length(prompt) < requested_prompt_tokens:
                                prompt += "hi"
                        else:
                            prompt = proposed_prompt

                        if get_token_length(prompt) >= requested_prompt_tokens:
                            break

                    sampling_lines = False
                    break
                prompt += line_to_add
                remaining_prompt_tokens = requested_prompt_tokens - get_token_length(
                    prompt
                )
        return prompt

    @classmethod
    def _sample_random_positive_int(cls, mean: int, stddev: int) -> int:
        random_pos_int = -1
        while random_pos_int <= 0:
            random_pos_int = int(random.gauss(mean, stddev))

        return random_pos_int
