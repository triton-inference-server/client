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

from enum import Enum, auto


class ModelSelectionStrategy(Enum):
    ROUND_ROBIN = auto()
    RANDOM = auto()


class PromptSource(Enum):
    SYNTHETIC = auto()
    DATASET = auto()
    FILE = auto()

    def to_lowercase(self):
        return self.name.lower()


class OutputFormat(Enum):
    OPENAI_CHAT_COMPLETIONS = auto()
    OPENAI_COMPLETIONS = auto()
    OPENAI_EMBEDDINGS = auto()
    RANKINGS = auto()
    TENSORRTLLM = auto()
    VLLM = auto()

    def to_lowercase(self):
        return self.name.lower()


DEFAULT_STARTING_INDEX = 0
DEFAULT_LENGTH = 100
DEFAULT_TENSORRTLLM_MAX_TOKENS = 256
DEFAULT_BATCH_SIZE = 1
DEFAULT_RANDOM_SEED = 0
DEFAULT_PROMPT_TOKENS_MEAN = 550
DEFAULT_PROMPT_TOKENS_STDDEV = 0
DEFAULT_OUTPUT_TOKENS_MEAN = -1
DEFAULT_OUTPUT_TOKENS_STDDEV = 0
DEFAULT_NUM_PROMPTS = 100
