import base64
from io import BytesIO

import pytest
from genai_perf.llm_inputs.synthetic_image_generator import (
    ImageEncoding,
    SyntheticImageGenerator,
)
from PIL import Image


def test_generating_images():
    sut = SyntheticImageGenerator(image_encodeing=ImageEncoding.PIL_IMAGE)

    data = next(sut)

    assert isinstance(data, Image.Image), "generator produces unexpected type of data"


def test_base64_encoding():
    sut = SyntheticImageGenerator(image_encodeing=ImageEncoding.BASE64)

    base64String = next(sut)

    base64prefix = "data:image/png;base64,"
    assert base64String.startswith(base64prefix), "unexpected prefix"
    data = base64String[len(base64prefix) :]

    # test if generator encodes to base64
    img_data = base64.b64decode(data)
    img_bytes = BytesIO(img_data)
    # test if an image is encoded
    Image.open(img_bytes)
