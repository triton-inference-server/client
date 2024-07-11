import base64
from io import BytesIO

import numpy as np
import pytest
from genai_perf.llm_inputs.synthetic_image_generator import (
    ImageFormat,
    RandomFormatBase64Encoder,
    build_synthetic_image_generator,
    white_images_generator,
)
from PIL import Image


@pytest.mark.parametrize(
    "image_size",
    [
        (100, 100),
        (200, 200),
    ],
)
def test_different_image_size(image_size):
    sut = build_synthetic_image_generator(
        mean_size=image_size, dimensions_stddev=[0, 0]
    )

    image = next(sut)

    assert isinstance(image, Image.Image), "generator produces unexpected type of data"
    assert image.size == image_size, "image not resized to the target size"


def test_white_images_generator():
    sut = white_images_generator()

    image = next(sut)
    assert isinstance(image, Image.Image), "generator produces unexpected type of data"
    white_pixel = np.array([[[255, 255, 255]]])
    assert (np.array(image) == white_pixel).all(), "not all pixels are white"


@pytest.mark.parametrize("image_format", [ImageFormat.PNG, ImageFormat.JPEG])
def test_base64_encoding_with_different_formats(image_format):
    image = Image.new("RGB", (100, 100))
    sut = RandomFormatBase64Encoder(image_formats=[image_format])

    base64String = sut(image)

    base64prefix = f"data:image/{image_format.name.lower()};base64,"
    assert base64String.startswith(base64prefix), "unexpected prefix"
    data = base64String[len(base64prefix) :]

    # test if generator encodes to base64
    img_data = base64.b64decode(data)
    img_bytes = BytesIO(img_data)
    # test if an image is encoded
    image = Image.open(img_bytes)

    assert image.format == image_format.name
