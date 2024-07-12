import base64
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from genai_perf.exceptions import GenAIPerfException
from genai_perf.llm_inputs.synthetic_image_generator import (
    ImageFormat,
    RandomFormatBase64Encoder,
    SyntheticImageGenerator,
    images_from_file_generator,
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
    sut = SyntheticImageGenerator(
        mean_size=image_size,
        dimensions_stddev=[0, 0],
        image_iterator=white_images_generator(),
    )

    image = next(sut)

    assert isinstance(image, Image.Image), "generator produces unexpected type of data"
    assert image.size == image_size, "image not resized to the target size"


def test_negative_size_is_not_selected():
    sut = SyntheticImageGenerator(
        mean_size=(-1, -1),
        dimensions_stddev=[10, 10],
        image_iterator=white_images_generator(),
    )

    # exception is raised, when PIL.Image.resize is called with negative values
    image = next(sut)


@patch("pathlib.Path.exists", return_value=False)
def test_images_from_file_raises_when_file_not_found(mock_exists):
    DUMMY_PATH = Path("dummy-image.png")
    sut = images_from_file_generator(DUMMY_PATH)

    with pytest.raises(GenAIPerfException):
        next(sut)


DUMMY_IMAGE = Image.new("RGB", (100, 100), color="blue")


@patch("pathlib.Path.exists", return_value=True)
@patch(
    "PIL.Image.open",
    return_value=DUMMY_IMAGE,
)
def test_images_from_file_generates_multiple_times(mock_file, mock_exists):
    DUMMY_PATH = Path("dummy-image.png")
    sut = images_from_file_generator(DUMMY_PATH)

    image = next(sut)
    mock_exists.assert_called_once()
    mock_file.assert_called_once_with(DUMMY_PATH)
    assert image == DUMMY_IMAGE, "unexpected image produced"

    image = next(sut)
    assert image == DUMMY_IMAGE, "unexpected image produced"


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
