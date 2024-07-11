import base64
from enum import Enum, auto
from io import BytesIO

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
        prefix = f"data:image/{self.image_format.name.lower()};base64"
        return f"{prefix},{data}"


class SyntheticImageGenerator:
    def __init__(
        self,
        image_format: ImageFormat = ImageFormat.PNG,
    ):
        self.image_format = image_format
        self._image_iterator = self.white_images_iterator()

    def __iter__(self):
        return self

    def __next__(self):
        image = next(self._image_iterator)
        return image

    def white_images_iterator(self):
        white_image = Image.new("RGB", (100, 100), color="white")
        while True:
            yield white_image
