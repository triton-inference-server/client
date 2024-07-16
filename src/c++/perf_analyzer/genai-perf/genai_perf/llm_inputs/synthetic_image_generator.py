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

import numpy as np
from genai_perf import utils
from genai_perf.llm_inputs.llm_inputs import ImageFormat
from PIL import Image


class SyntheticImageGenerator:
    """A simple synthetic image generator that generates multiple synthetic
    images from source image (either real or random noise).
    """

    def __init__(
        self,
        image_width_mean: int,
        image_height_mean: int,
        image_width_stddev: int,
        image_height_stddev: int,
        image_format: ImageFormat,
        random_seed: int,
    ):
        self._image_width_mean = image_width_mean
        self._image_height_mean = image_height_mean
        self._image_width_stddev = image_width_stddev
        self._image_height_stddev = image_height_stddev
        self._image_format = image_format
        self.rng = np.random.default_rng(seed=random_seed)

    def _sample_random_positive_integer(self, mean: int, stddev: int) -> int:
        while True:
            n = int(self.rng.normal(mean, stddev))
            if n > 0:
                break
        return n

    def _get_next_image(self):
        width = self._sample_random_positive_integer(
            self._image_width_mean, self._image_width_stddev
        )
        height = self._sample_random_positive_integer(
            self._image_height_mean, self._image_height_stddev
        )
        # (TMA-1994) support real images as source image
        shape = width, height, 3
        noise = self.rng.integers(0, 256, shape, dtype=np.uint8)
        return Image.fromarray(noise)

    def create_synthetic_image(self) -> str:
        image = self._get_next_image()
        img_base64 = utils.encode_image(image, self._image_format.name)
        return f"data:image/{self._image_format.name.lower()};base64,{img_base64}"
