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

import pytest
from genai_perf.exceptions import GenAIPerfException
from genai_perf.tokenizer import DEFAULT_TOKENIZER, get_tokenizer


class TestTokenizer:
    def test_default_tokenizer(self):
        tokenizer_model = DEFAULT_TOKENIZER
        get_tokenizer(tokenizer_model)

    def test_non_default_tokenizer(self):
        tokenizer_model = "gpt2"
        get_tokenizer(tokenizer_model)

    def test_bad_tokenizer(self):
        with pytest.raises(GenAIPerfException):
            get_tokenizer("bad_tokenizer")

    def test_default_args(self):
        tokenizer_model = DEFAULT_TOKENIZER
        tokenizer = get_tokenizer(tokenizer_model)

        # There are 3 special tokens in the default tokenizer
        #  - <unk>: 0  (unknown)
        #  - <s>: 1  (beginning of sentence)
        #  - </s>: 2  (end of sentence)
        special_tokens = list(tokenizer._tokenizer.added_tokens_encoder.keys())
        special_token_ids = list(tokenizer._tokenizer.added_tokens_encoder.values())

        # special tokens are disabled by default
        text = "This is test."
        tokens = tokenizer(text)["input_ids"]
        assert all([s not in tokens for s in special_token_ids])

        tokens = tokenizer.encode(text)
        assert all([s not in tokens for s in special_token_ids])

        output = tokenizer.decode(tokens)
        assert all([s not in output for s in special_tokens])

        # check special tokens is enabled
        text = "This is test."
        tokens = tokenizer(text, add_special_tokens=True)["input_ids"]
        assert any([s in tokens for s in special_token_ids])

        tokens = tokenizer.encode(text, add_special_tokens=True)
        assert any([s in tokens for s in special_token_ids])

        output = tokenizer.decode(tokens, skip_special_tokens=False)
        assert any([s in output for s in special_tokens])
