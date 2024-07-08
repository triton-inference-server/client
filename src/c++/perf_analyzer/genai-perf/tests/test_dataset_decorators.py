import pytest
from genai_perf.llm_inputs.dataset_decorators import ImageDecorator, UploadMethod


@pytest.fixture()
def generic_dataset():
    return {
        "features": [{"name": "text_input"}],
        "rows": [{"text_input": "test prompt"}],
    }


def test_base64_images(generic_dataset):
    upload_method = UploadMethod.base64
    sut = ImageDecorator(upload_method=upload_method)

    updated_generic_dataset = sut.process(generic_dataset)

    prompt = updated_generic_dataset["rows"][0]["text_input"]

    assert isinstance(prompt, list), "The prompt is not converted to a list"

    for content in prompt:
        assert isinstance(content, dict), "The prompt element is not a dict"

    images = [content for content in prompt if content["type"] == "image_url"]
    assert len(images) == 1, "Image not added to the prompt"

    assert images[0]["image_url"]["url"].startswith(
        f"data:image/png;{upload_method.name},"
    )
