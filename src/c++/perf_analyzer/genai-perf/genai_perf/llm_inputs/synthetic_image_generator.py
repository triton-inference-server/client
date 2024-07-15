import base64
from enum import Enum, auto
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple, cast

import numpy as np
from genai_perf.exceptions import GenAIPerfException
from PIL import Image


class ImageFormat(Enum):
    JPEG = auto()
    PNG = auto()


class RandomFormatBase64Encoder:
    def __init__(self, image_format: ImageFormat = ImageFormat.PNG):
        self.image_format = image_format

    def __call__(self, image):
        buffered = BytesIO()
        image.save(buffered, format=self.image_format.name)
        data = base64.b64encode(buffered.getvalue()).decode("utf-8")
        prefix = f"data:image/{self.image_format.name.lower()};base64"
        return f"{prefix},{data}"


def images_from_file_generator(image_path: Path):
    if not image_path.exists():
        raise GenAIPerfException(f"File not found: {image_path}")

    image = Image.open(image_path)
    while True:
        yield image


def white_images_generator():
    white_image = Image.new("RGB", (100, 100), color="white")
    while True:
        yield white_image


def build_synthetic_image_generator(
    mean_size: Tuple[int, int],
    dimensions_stddev: Tuple[int, int],
    image_path: Optional[Path] = None,
    image_format: ImageFormat = ImageFormat.PNG,
):
    if image_path is None:
        image_iterator = white_images_generator()
    else:
        image_path = cast(Path, image_path)
        image_iterator = images_from_file_generator(image_path)

    image_generator = SyntheticImageGenerator(
        mean_size=mean_size,
        dimensions_stddev=dimensions_stddev,
        image_iterator=image_iterator,
    )
    base64_encode = RandomFormatBase64Encoder(image_format)
    return (base64_encode(image) for image in image_generator)


class SyntheticImageGenerator:
    def __init__(
        self,
        mean_size,
        dimensions_stddev,
        image_iterator,
    ):
        self.image_iterator = image_iterator
        self.mean_size = mean_size
        self.dimensions_stddev = dimensions_stddev

    def __iter__(self):
        return self

    def _sample_random_positive_integer(self, mean: int, stddev: int) -> int:
        while True:
            n = int(np.random.normal(mean, stddev))
            if n > 0:
                break
        return n

    def random_resize(self, image):
        width = self._sample_random_positive_integer(
            self._image_width_mean, self._image_width_stddev
        )
        height = self._sample_random_positive_integer(
            self._image_height_mean, self._image_height_stddev
        )
        return image.resize((width, height))

    def __next__(self):
        image = next(self.image_iterator)
        image = self.random_resize(image)
        return image
