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
from typing import TYPE_CHECKING, Union

from genai_perf.exceptions import GenAIPerfException

Tokenizer = Union["PreTrainedTokenizer", "PreTrainedTokenizerFast"]
DEFAULT_TOKENIZER = "hf-internal-testing/llama-tokenizer"


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


def get_tokenizer(
    tokenizer_model: str,
) -> Tokenizer:
    """
    Download the tokenizer from Huggingface.co
    """
    try:
        # Silence tokenizer warning on first use
        with contextlib.redirect_stdout(
            io.StringIO()
        ) as stdout, contextlib.redirect_stderr(io.StringIO()) as stderr:
            from transformers import AutoTokenizer
            from transformers import logging as token_logger

            token_logger.set_verbosity_error()
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    except Exception as e:
        raise GenAIPerfException(e)

    # Disable add_bos_token so that llama tokenizer does not add bos token
    # (aka. beginning-of-sentence) to the beginning of every response
    # outputs, increasing the token count by 1 for each output response.
    # Note: The type is being ignored here, because not all tokenizers have
    # an add_bos_token variable.
    tokenizer.add_bos_token = False  # type: ignore

    return tokenizer
