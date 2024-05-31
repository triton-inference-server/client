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
from typing import List

from genai_perf.exceptions import GenAIPerfException

# Silence tokenizer warning on import
with contextlib.redirect_stdout(io.StringIO()) as stdout, contextlib.redirect_stderr(
    io.StringIO()
) as stderr:
    from transformers import AutoTokenizer, BatchEncoding
    from transformers import logging as token_logger

    token_logger.set_verbosity_error()

DEFAULT_TOKENIZER = "hf-internal-testing/llama-tokenizer"


class Tokenizer:
    """
    A small wrapper class around Huggingface Tokenizer
    """

    def __init__(self, name: str) -> None:
        """
        Initialize by downloading the tokenizer from Huggingface.co
        """
        try:
            # Silence tokenizer warning on first use
            with contextlib.redirect_stdout(
                io.StringIO()
            ) as stdout, contextlib.redirect_stderr(io.StringIO()) as stderr:
                tokenizer = AutoTokenizer.from_pretrained(name)
        except Exception as e:
            raise GenAIPerfException(e)

        self._tokenizer = tokenizer

        # default tokenizer parameters for __call__, encode, decode methods
        self._call_args = {"add_special_tokens": False}
        self._encode_args = {"add_special_tokens": False}
        self._decode_args = {"skip_special_tokens": True}

    def __call__(self, text, **kwargs) -> BatchEncoding:
        self._call_args.update(kwargs)
        return self._tokenizer(text, **self._call_args)

    def encode(self, text, **kwargs) -> List[int]:
        self._encode_args.update(kwargs)
        return self._tokenizer.encode(text, **self._encode_args)

    def decode(self, token_ids, **kwargs) -> str:
        self._decode_args.update(kwargs)
        return self._tokenizer.decode(token_ids, **self._decode_args)

    def __repr__(self) -> str:
        return self._tokenizer.__repr__()


def get_tokenizer(tokenizer_model: str) -> Tokenizer:
    """
    Return tokenizer for the given model name
    """
    return Tokenizer(tokenizer_model)
