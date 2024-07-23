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

import glob
import random
from enum import Enum, auto
from pathlib import Path
from typing import Optional

from genai_perf import utils
from PIL import Image


class ImageFormat(Enum):
    PNG = auto()
    JPEG = auto()


class SyntheticImageGenerator:
    """A simple synthetic image generator that generates multiple synthetic
    images from the source images.
    """

    @classmethod
    def create_synthetic_image(
        cls,
        image_width_mean: int,
        image_width_stddev: int,
        image_height_mean: int,
        image_height_stddev: int,
        image_format: Optional[ImageFormat] = None,
    ) -> str:
        """Generate base64 encoded synthetic image using the source images."""
        if image_format is None:
            image_format = random.choice(list(ImageFormat))
        width = cls._sample_random_positive_integer(
            image_width_mean, image_width_stddev
        )
        height = cls._sample_random_positive_integer(
            image_height_mean, image_height_stddev
        )

        image = cls._sample_source_image()
        image = image.resize(size=(width, height))

        img_base64 = utils.encode_image(image, image_format.name)
        return f"data:image/{image_format.name.lower()};base64,{img_base64}"

    @classmethod
    def _sample_source_image(cls):
        """Sample one image among the source images."""
        filepath = Path(__file__).parent.resolve() / "source_images" / "*"
        filenames = glob.glob(str(filepath))
        return Image.open(random.choice(filenames))

    @classmethod
    def _sample_random_positive_integer(cls, mean: int, stddev: int) -> int:
        n = int(abs(random.gauss(mean, stddev)))
        return n if n != 0 else 1  # avoid zero
