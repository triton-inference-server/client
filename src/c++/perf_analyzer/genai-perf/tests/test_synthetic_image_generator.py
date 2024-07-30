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

import base64
import random
from io import BytesIO

import pytest
from genai_perf.llm_inputs.synthetic_image_generator import (
    ImageFormat,
    SyntheticImageGenerator,
)
from PIL import Image


def decode_image(base64_string):
    _, data = base64_string.split(",")
    decoded_data = base64.b64decode(data)
    return Image.open(BytesIO(decoded_data))


@pytest.mark.parametrize(
    "expected_image_size",
    [
        (100, 100),
        (200, 200),
    ],
)
def test_different_image_size(expected_image_size):
    expected_width, expected_height = expected_image_size
    base64_string = SyntheticImageGenerator.create_synthetic_image(
        image_width_mean=expected_width,
        image_width_stddev=0,
        image_height_mean=expected_height,
        image_height_stddev=0,
        image_format=ImageFormat.PNG,
    )

    image = decode_image(base64_string)
    assert image.size == expected_image_size, "image not resized to the target size"


def test_negative_size_is_not_selected():
    # exception is raised, when PIL.Image.resize is called with negative values
    _ = SyntheticImageGenerator.create_synthetic_image(
        image_width_mean=-1,
        image_width_stddev=10,
        image_height_mean=-1,
        image_height_stddev=10,
        image_format=ImageFormat.PNG,
    )


@pytest.mark.parametrize(
    "width_mean, width_stddev, height_mean, height_stddev",
    [
        (100, 15, 100, 15),
        (123, 10, 456, 7),
    ],
)
def test_generator_deterministic(width_mean, width_stddev, height_mean, height_stddev):
    random.seed(123)
    img1 = SyntheticImageGenerator.create_synthetic_image(
        image_width_mean=width_mean,
        image_width_stddev=width_stddev,
        image_height_mean=height_mean,
        image_height_stddev=height_stddev,
        image_format=ImageFormat.PNG,
    )

    random.seed(123)
    img2 = SyntheticImageGenerator.create_synthetic_image(
        image_width_mean=width_mean,
        image_width_stddev=width_stddev,
        image_height_mean=height_mean,
        image_height_stddev=height_stddev,
        image_format=ImageFormat.PNG,
    )

    assert img1 == img2, "generator is nondererministic"


@pytest.mark.parametrize("image_format", [ImageFormat.PNG, ImageFormat.JPEG])
def test_base64_encoding_with_different_formats(image_format):
    img_base64 = SyntheticImageGenerator.create_synthetic_image(
        image_width_mean=100,
        image_width_stddev=100,
        image_height_mean=100,
        image_height_stddev=100,
        image_format=image_format,
    )

    # check prefix
    expected_prefix = f"data:image/{image_format.name.lower()};base64,"
    assert img_base64.startswith(expected_prefix), "unexpected prefix"

    # check image format
    data = img_base64[len(expected_prefix) :]
    img_data = base64.b64decode(data)
    img_bytes = BytesIO(img_data)
    image = Image.open(img_bytes)
    assert image.format == image_format.name
