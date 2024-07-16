import base64
from io import BytesIO

import numpy as np
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
    sut = SyntheticImageGenerator(
        image_width_mean=expected_width,
        image_height_mean=expected_height,
        image_width_stddev=0,
        image_height_stddev=0,
        image_format=ImageFormat.PNG,
        random_seed=1234,
    )

    base64_string = sut.create_synthetic_image()
    image = decode_image(base64_string)

    assert image.size == expected_image_size, "image not resized to the target size"


def test_negative_size_is_not_selected():
    sut = SyntheticImageGenerator(
        image_width_mean=-1,
        image_height_mean=-1,
        image_width_stddev=10,
        image_height_stddev=10,
        image_format=ImageFormat.PNG,
        random_seed=1234,
    )

    # exception is raised, when PIL.Image.resize is called with negative values
    sut.create_synthetic_image()


def test_generator_deterministic():
    IMAGE_SIZE = 100, 100
    STDDEV = 100, 100
    IMG_FORMAT = ImageFormat.PNG
    SEED = 44

    sut1 = SyntheticImageGenerator(*IMAGE_SIZE, *STDDEV, IMG_FORMAT, SEED)
    sut2 = SyntheticImageGenerator(*IMAGE_SIZE, *STDDEV, IMG_FORMAT, SEED)

    for _ in range(5):
        img1 = sut1.create_synthetic_image()
        img2 = sut2.create_synthetic_image()
        assert img1 == img2, "generator is nondererministic"


@pytest.mark.parametrize("image_format", [ImageFormat.PNG, ImageFormat.JPEG])
def test_base64_encoding_with_different_formats(image_format):
    IMAGE_SIZE = 100, 100
    STDDEV = 100, 100
    SEED = 44

    sut = SyntheticImageGenerator(*IMAGE_SIZE, *STDDEV, image_format, SEED)
    img_base64 = sut.create_synthetic_image()

    # check prefix
    expected_prefix = f"data:image/{image_format.name.lower()};base64,"
    assert img_base64.startswith(expected_prefix), "unexpected prefix"

    # check image format
    data = img_base64[len(expected_prefix) :]
    img_data = base64.b64decode(data)
    img_bytes = BytesIO(img_data)
    image = Image.open(img_bytes)
    assert image.format == image_format.name
