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


def test_random_image_format():
    random.seed(123)
    img1 = SyntheticImageGenerator.create_synthetic_image(
        image_width_mean=100,
        image_width_stddev=100,
        image_height_mean=100,
        image_height_stddev=100,
        image_format=None,
    )

    random.seed(456)
    img2 = SyntheticImageGenerator.create_synthetic_image(
        image_width_mean=100,
        image_width_stddev=100,
        image_height_mean=100,
        image_height_stddev=100,
        image_format=None,
    )

    # check prefix
    assert img1.startswith("data:image/png")
    assert img2.startswith("data:image/jpeg")
