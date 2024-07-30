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

from collections import namedtuple
from unittest.mock import patch

from genai_perf.llm_inputs.dataset_retriever import DatasetRetriever
from genai_perf.llm_inputs.inputs_utils import ImageFormat, OutputFormat
from genai_perf.tokenizer import DEFAULT_TOKENIZER, get_tokenizer


class TestDatasetRetriever:

    @patch(
        "genai_perf.llm_inputs.synthetic_prompt_generator.SyntheticPromptGenerator.create_synthetic_prompt",
        side_effect=["prompt1", "prompt2", "prompt3"],
    )
    @patch(
        "genai_perf.llm_inputs.synthetic_image_generator.SyntheticImageGenerator.create_synthetic_image",
        side_effect=["image1", "image2", "image3"],
    )
    def test_from_synthetic_multi_modal(self, mock_prompts, mock_images):
        Data = namedtuple("Data", ["text_input", "image"])
        expected_data = [
            Data(text_input="prompt1", image="image1"),
            Data(text_input="prompt2", image="image2"),
            Data(text_input="prompt3", image="image3"),
        ]
        num_prompts = 3

        dataset = DatasetRetriever.from_synthetic(
            tokenizer=get_tokenizer(DEFAULT_TOKENIZER),
            prompt_tokens_mean=3,
            prompt_tokens_stddev=0,
            num_of_output_prompts=num_prompts,
            image_width_mean=5,
            image_width_stddev=0,
            image_height_mean=5,
            image_height_stddev=0,
            image_format=ImageFormat.PNG,
            output_format=OutputFormat.OPENAI_VISION,
        )

        assert len(dataset) == len(expected_data)

        for i, data in enumerate(expected_data):
            assert dataset[i] == {
                "text_input": data.text_input,
                "image": data.image,
            }
