import base64
import json
from io import BytesIO

import pytest
from genai_perf.llm_inputs.nvcf_assets import NvcfUploader
from PIL import Image


def test_not_upload_text():
    DUMMY_API_KEY = "test api key"
    input_dataset = {
        "rows": [{"text_input": "small message"}],
    }
    sut = NvcfUploader(threshold_kbytes=0, nvcf_api_key=DUMMY_API_KEY)

    new_dataset = sut.upload_large_assets(input_dataset)

    assert (
        new_dataset == input_dataset
    ), "There is no row to upload - dataset should stay unchanged"


def generate_image(approx_kbytes):
    estimated_base64_ratio = 4 / 3  # Base64 encoding increases size by about 33%
    color_channels = 3
    npixels = approx_kbytes * 1000 / color_channels / estimated_base64_ratio
    width = height = int(npixels**0.5)
    img = Image.new("RGB", (width, height), color="white")
    buffered = BytesIO()
    img.save(buffered, format="BMP")  # BMP doesn't compress
    data = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/bmp;base64,{data}"


def test_threshold_applies_for_each_row_independently(requests_mock):
    DUMMY_API_KEY = "test api key"
    image_300kb = generate_image(approx_kbytes=300)
    image_200kb = generate_image(approx_kbytes=200)
    input_dataset = {
        "rows": [
            {
                "text_input": [
                    {"image_url": {"url": image_300kb}},
                ]
            },
            {
                "text_input": [
                    {"image_url": {"url": image_200kb}},
                ]
            },
        ]
    }

    sut = NvcfUploader(threshold_kbytes=400, nvcf_api_key=DUMMY_API_KEY)

    new_dataset = sut.upload_large_assets(input_dataset)

    rows = new_dataset["rows"]
    assert (
        rows[0]["text_input"][0]["image_url"]["url"] == image_300kb
    ), "300kb asset should not be uploaded"
    assert (
        rows[1]["text_input"][0]["image_url"]["url"] == image_200kb
    ), "200kb asset should not be uploaded"


@pytest.mark.parametrize(
    "threshold_kbytes",
    [100, 400],
)
def test_upload_images(requests_mock, threshold_kbytes):
    DUMMY_API_KEY = "test api key"
    DUMMY_ASSET_ID = "dummy asset id"
    DUMMY_UPLOAD_URL = "https://dummy-upload-url"
    NEW_ASSET_RESP = {
        "assetId": DUMMY_ASSET_ID,
        "uploadUrl": DUMMY_UPLOAD_URL,
        "contentType": "image/jpeg",
        "description": "test image",
    }
    image_300kb = generate_image(approx_kbytes=300)
    text_200kb = 200_000 * "!"
    input_dataset = {
        "rows": [
            {
                "text_input": [
                    {"text": text_200kb},
                    {"image_url": {"url": image_300kb}},
                ]
            }
        ]
    }

    sut = NvcfUploader(threshold_kbytes=threshold_kbytes, nvcf_api_key=DUMMY_API_KEY)

    requests_mock.post(
        "https://api.nvcf.nvidia.com/v2/nvcf/assets", json=NEW_ASSET_RESP
    )
    requests_mock.put(DUMMY_UPLOAD_URL)
    new_dataset = sut.upload_large_assets(input_dataset)

    prompts = new_dataset["rows"][0]["text_input"]
    assert "text" in prompts[0], "prompts order not preserved"
    assert prompts[0]["text"] == text_200kb, "text asset should not be uploaded"
    assert "image_url" in prompts[1], "prompts order not preserved"
    assert prompts[1]["image_url"]["url"] == f"data:image/bmp;asset_id,{DUMMY_ASSET_ID}"


def test_payload_is_closer_to_threshold(requests_mock):
    DUMMY_API_KEY = "test api key"
    DUMMY_ASSET_ID = "dummy asset id"
    DUMMY_UPLOAD_URL = "https://dummy-upload-url"
    NEW_ASSET_RESP = {
        "assetId": DUMMY_ASSET_ID,
        "uploadUrl": DUMMY_UPLOAD_URL,
        "contentType": "image/jpeg",
        "description": "test image",
    }
    image_300kb = generate_image(approx_kbytes=300)
    image_200kb = generate_image(approx_kbytes=200)
    input_dataset = {
        "rows": [
            {
                "text_input": [
                    {"image_url": {"url": image_300kb}},
                    {"image_url": {"url": image_200kb}},
                ]
            }
        ]
    }

    sut = NvcfUploader(nvcf_api_key=DUMMY_API_KEY, threshold_kbytes=400)

    requests_mock.post(
        "https://api.nvcf.nvidia.com/v2/nvcf/assets", json=NEW_ASSET_RESP
    )
    requests_mock.put(DUMMY_UPLOAD_URL)
    new_dataset = sut.upload_large_assets(input_dataset)

    prompts = new_dataset["rows"][0]["text_input"]
    assert (
        prompts[1]["image_url"]["url"] == f"data:image/bmp;asset_id,{DUMMY_ASSET_ID}"
    ), "smaller image should be uploaded"
    assert (
        prompts[0]["image_url"]["url"] == image_300kb
    ), "larger image should not be uploaded"


def test_upload_report(requests_mock):
    DUMMY_API_KEY = "test api key"
    DUMMY_ASSET_ID = "dummy asset id"
    DUMMY_UPLOAD_URL = "https://dummy-upload-url"
    NEW_ASSET_RESP = {
        "assetId": DUMMY_ASSET_ID,
        "uploadUrl": DUMMY_UPLOAD_URL,
        "contentType": "image/jpeg",
        "description": "test image",
    }
    image_300kb = generate_image(approx_kbytes=300)
    input_dataset = {
        "rows": [
            {
                "text_input": [
                    {"image_url": {"url": image_300kb}},
                ]
            }
        ]
    }

    sut = NvcfUploader(nvcf_api_key=DUMMY_API_KEY, threshold_kbytes=200)

    requests_mock.post(
        "https://api.nvcf.nvidia.com/v2/nvcf/assets", json=NEW_ASSET_RESP
    )
    requests_mock.put(DUMMY_UPLOAD_URL)
    sut.upload_large_assets(input_dataset)

    report = sut.get_upload_report()
    assert DUMMY_ASSET_ID in report, "file upload not recorded"
