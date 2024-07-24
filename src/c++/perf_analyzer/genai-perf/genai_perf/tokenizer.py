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
