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

import itertools
import math
import pathlib
import random
import re
from typing import List

from genai_perf.tokenizer import Tokenizer


class SyntheticPromptGenerator:
    """
    This class generates synthetic prompts for inputs generation.
    """

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
