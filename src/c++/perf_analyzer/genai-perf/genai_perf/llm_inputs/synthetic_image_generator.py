import base64
from enum import Enum, auto
from io import BytesIO
from typing import Optional, cast

import numpy as np
from PIL import Image


class ImageFormat(Enum):
    JPEG = auto()
    PNG = auto()


class SyntheticImageGenerator:
    def __init__(
        self,
        image_width_mean: int,
        image_height_mean: int,
        image_width_stddev: int,
        image_height_stddev: int,
        image_format: ImageFormat = ImageFormat.PNG,
        rng: Optional[np.random.Generator] = None,
    ):
        self._image_width_mean = image_width_mean
        self._image_height_mean = image_height_mean
        self._image_width_stddev = image_width_stddev
        self._image_height_stddev = image_height_stddev
        self.image_format = image_format
        self.rng = cast(np.random.Generator, rng or np.random.default_rng())

    def __iter__(self):
        return self

    def _sample_random_positive_integer(self, mean: int, stddev: int) -> int:
        while True:
            n = int(self.rng.normal(mean, stddev))
            if n > 0:
                break
        return n

    def _get_next_image(self):
        width = self._sample_random_positive_integer(
            self._image_width_mean, self._image_width_stddev
        )
        height = self._sample_random_positive_integer(
            self._image_height_mean, self._image_height_stddev
        )
        shape = width, height, 3
        noise = self.rng.integers(0, 256, shape, dtype=np.uint8)
        return Image.fromarray(noise)

    def _encode(self, image):
        buffered = BytesIO()
        image.save(buffered, format=self.image_format.name)
        data = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/{self.image_format.name.lower()};base64,{data}"

    def __next__(self) -> str:
        image = self._get_next_image()
        base64_string = self._encode(image)
        return base64_string
