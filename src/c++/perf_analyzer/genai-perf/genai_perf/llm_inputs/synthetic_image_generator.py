import base64
from enum import Enum, auto
from io import BytesIO
from pathlib import Path
from typing import Generator, List, Optional, Tuple, cast

import numpy as np
from genai_perf.exceptions import GenAIPerfException
from PIL import Image


class ImageFormat(Enum):
    JPEG = auto()
    PNG = auto()


class Base64Encoder:
    def __init__(self, image_format: ImageFormat = ImageFormat.PNG):
        self.image_format = image_format

    def __call__(self, image):
        buffered = BytesIO()
        image.save(buffered, format=self.image_format.name)
        data = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/{self.image_format.name.lower()};base64,{data}"


def images_from_file_generator(image_path: Path) -> Generator[Image.Image, None, None]:
    if not image_path.exists():
        raise GenAIPerfException(f"File not found: {image_path}")

    image = Image.open(image_path)
    while True:
        yield image


def white_images_generator() -> Generator[Image.Image, None, None]:
    white_image = Image.new("RGB", (100, 100), color="white")
    while True:
        yield white_image


def build_synthetic_image_generator(
    image_width_mean: int,
    image_height_mean: int,
    image_width_stddev: int,
    image_height_stddev: int,
    image_path: Optional[Path] = None,
    image_format: ImageFormat = ImageFormat.PNG,
):
    if image_path is None:
        image_iterator = white_images_generator()
    else:
        image_path = cast(Path, image_path)
        image_iterator = images_from_file_generator(image_path)

    image_generator = SyntheticImageGenerator(
        image_width_mean=image_width_mean,
        image_height_mean=image_height_mean,
        image_width_stddev=image_width_stddev,
        image_height_stddev=image_height_stddev,
        image_iterator=image_iterator,
    )
    base64_encode = Base64Encoder(image_format)
    return (base64_encode(image) for image in image_generator)


class SyntheticImageGenerator:
    def __init__(
        self,
        image_width_mean: int,
        image_height_mean: int,
        image_width_stddev: int,
        image_height_stddev: int,
        image_iterator: Generator[Image.Image, None, None],
        rng: Optional[np.random.Generator] = None,
    ):
        self.image_iterator = image_iterator
        self._image_width_mean = image_width_mean
        self._image_height_mean = image_height_mean
        self._image_width_stddev = image_width_stddev
        self._image_height_stddev = image_height_stddev
        self.rng = rng or np.random.default_rng()

    def __iter__(self):
        return self

    def _sample_random_positive_integer(self, mean: int, stddev: int) -> int:
        while True:
            n = int(self.rng.normal(mean, stddev))
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
