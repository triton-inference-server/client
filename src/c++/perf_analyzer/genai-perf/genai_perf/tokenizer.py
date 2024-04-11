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

from genai_perf.exceptions import GenAIPerfException

# Silence tokenizer warning on import
with contextlib.redirect_stdout(io.StringIO()) as stdout, contextlib.redirect_stderr(
    io.StringIO()
) as stderr:
    from transformers import (
        AutoTokenizer,
        BatchEncoding,
        PreTrainedTokenizer,
        PreTrainedTokenizerFast,
    )
    from transformers import logging as token_logger
    from transformers.tokenization_utils_base import EncodingFast

    token_logger.set_verbosity_error()


DEFAULT_TOKENIZER = "hf-internal-testing/llama-tokenizer"


def get_tokenizer(
    tokenizer_model: str,
) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
    """
    Download the tokenizer from Huggingface.co
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    except Exception as e:
        raise GenAIPerfException(e)

    return tokenizer
