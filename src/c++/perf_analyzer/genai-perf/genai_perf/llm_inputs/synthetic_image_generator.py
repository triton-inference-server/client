import base64
from enum import Enum, auto
from io import BytesIO

from genai_perf.exceptions import GenAIPerfException
from PIL import Image


class ImageEncoding(Enum):
    PIL_IMAGE = auto()
    BASE64 = auto()


class SyntheticImageGenerator:
    def __init__(self, image_encodeing: ImageEncoding):
        self.image_encodeing = image_encodeing
        self._image_iterator = self.white_images_iterator()

    def __iter__(self):
        return self

    def __next__(self):
        image = next(self._image_iterator)
        return self._encode(image)

    def white_images_iterator(self):
        white_image = Image.new("RGB", (100, 100), color="white")
        while True:
            yield white_image

    def _encode(self, image):
        if self.image_encodeing == ImageEncoding.PIL_IMAGE:
            return image
        elif self.image_encodeing == ImageEncoding.BASE64:
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            data = base64.b64encode(buffered.getvalue()).decode("utf-8")
            prefix = "data:image/png;base64"
            return f"{prefix},{data}"
        else:
            raise GenAIPerfException(
                f"Unexpected image_encodeing: {self.image_encodeing.name}"
            )
