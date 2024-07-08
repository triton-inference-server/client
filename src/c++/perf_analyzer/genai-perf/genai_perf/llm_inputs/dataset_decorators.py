import base64
from enum import Enum, auto
from io import BytesIO

from PIL import Image, ImageDraw


class UploadMethod(Enum):
    base64 = auto()


class DatasetDecorator:
    def process(self, generic_dataset):
        return generic_dataset


def make_a_snowman():
    # Create a blank image with white background
    img = Image.new("RGB", (600, 400), color="skyblue")
    d = ImageDraw.Draw(img)

    # Draw the snowman's body (three circles)
    body_color = "white"
    d.ellipse([200, 500, 400, 700], fill=body_color, outline="black")  # Bottom circle
    d.ellipse([225, 350, 375, 550], fill=body_color, outline="black")  # Middle circle
    d.ellipse([250, 200, 350, 400], fill=body_color, outline="black")  # Head circle

    # Draw the snowman's eyes
    eye_color = "black"
    d.ellipse([275, 250, 285, 260], fill=eye_color)  # Left eye
    d.ellipse([315, 250, 325, 260], fill=eye_color)  # Right eye

    # Draw the snowman's nose (carrot)
    nose_color = "orange"
    d.polygon([(300, 270), (300, 280), (340, 275)], fill=nose_color)  # Nose

    # Draw the snowman's mouth (smile)
    mouth_color = "black"
    d.arc([275, 290, 325, 310], start=0, end=180, fill=mouth_color)  # Smile

    # Draw the snowman's buttons
    d.ellipse([290, 420, 310, 440], fill=eye_color)  # Top button
    d.ellipse([290, 460, 310, 480], fill=eye_color)  # Middle button
    d.ellipse([290, 500, 310, 520], fill=eye_color)  # Bottom button

    # Draw the snowman's arms
    arm_color = "brown"
    d.line([225, 450, 150, 400], fill=arm_color, width=5)  # Left arm
    d.line([375, 450, 450, 400], fill=arm_color, width=5)  # Right arm

    return img


class ImageDecorator(DatasetDecorator):
    def __init__(self, upload_method):
        self.upload_method = upload_method

    def process(self, generic_dataset):
        snowman_image = make_a_snowman()
        for row in generic_dataset["rows"]:
            if isinstance(row["text_input"], str):
                row["text_input"] = [
                    dict(
                        type="text",
                        text=row["text_input"],
                    )
                ]

            row["text_input"].append(self.pack_image(snowman_image))

        return generic_dataset

    def pack_image(self, image):
        image_repr = None
        if self.upload_method == UploadMethod.base64:
            image_repr = self.encode_image(image)
        else:
            raise GenAIPerfException("unexpected upload_method")
        return dict(
            type="image_url",
            image_url=f"data:image/png;{self.upload_method.name},{image_repr}",
        )

    def encode_image(self, img: Image):
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
