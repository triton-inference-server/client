import base64
from io import BytesIO

import pytest
from genai_perf.llm_inputs.synthetic_image_generator import (
    Base64Encoder,
    ImageFormat,
    SyntheticImageGenerator,
)
from PIL import Image


def test_generating_images():
    sut = SyntheticImageGenerator()

    data = next(sut)

    assert isinstance(data, Image.Image), "generator produces unexpected type of data"


@pytest.mark.parametrize("image_format", [ImageFormat.PNG, ImageFormat.JPEG])
def test_base64_encoding_with_different_formats(image_format):
    image = Image.new("RGB", (100, 100))
    sut = Base64Encoder(image_format=image_format)

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
