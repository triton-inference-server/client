import base64
from enum import Enum, auto
from io import BytesIO

import numpy as np
from genai_perf.exceptions import GenAIPerfException
from PIL import Image


class ImageFormat(Enum):
    JPEG = auto()
    PNG = auto()


class RandomFormatBase64Encoder:
    def __init__(self, image_formats: ImageFormat = ImageFormat.PNG):
        self.image_formats = image_formats

    def __call__(self, image):
        choice = np.random.randint(len(self.image_formats))
        image_format = self.image_formats[choice]
        buffered = BytesIO()
        image.save(buffered, format=image_format.name)
        data = base64.b64encode(buffered.getvalue()).decode("utf-8")
        prefix = f"data:image/{image_format.name.lower()};base64"
        return f"{prefix},{data}"


class SyntheticImageGenerator:
    def __init__(
        self,
        mean_size,
        dimensions_stddev,
    ):
        self._image_iterator = self.white_images_iterator()
        self.mean_size = mean_size
        self.dimensions_stddev = dimensions_stddev

    def __iter__(self):
        return self

    def random_resize(self, image):
        new_size = np.random.normal(self.mean_size, self.dimensions_stddev)
        new_size = new_size.astype(int)
        return image.resize(new_size)

    def __next__(self):
        image = next(self._image_iterator)
        image = self.random_resize(image)
        return image

    def white_images_iterator(self):
        white_image = Image.new("RGB", (100, 100), color="white")
        while True:
            yield white_image
