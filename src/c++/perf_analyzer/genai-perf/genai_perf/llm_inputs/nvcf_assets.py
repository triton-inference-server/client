import base64
import json
import time
from pathlib import Path
from typing import Dict

import genai_perf.logging as logging
import requests


def greedy_fill(size_limit, sizes):
    remaining = size_limit
    selected = []
    for i, size in sorted(enumerate(sizes), key=lambda x: -x[1]):
        if size <= remaining:
            selected.append(i)
            remaining -= size
    return selected


NVCF_URL = "https://api.nvcf.nvidia.com/v2/nvcf"


class NvcfUploader:
    def __init__(self, threshold_kbytes: int, nvcf_api_key: str):
        self.threshold_kbytes = threshold_kbytes
        self._upload_report: Dict[str, float] = {}
        self._initialize_headers(nvcf_api_key)

    def _initialize_headers(self, nvcf_api_key):
        self._headers = {
            "Authorization": f"Bearer {nvcf_api_key}",
            "accept": "application/json",
            "content-type": "application/json",
        }

    def _add_upload_report_entry(self, asset_id, time_delta):
        self._upload_report[asset_id] = time_delta

    def get_upload_report(self):
        return self._upload_report.copy()

    def upload_large_assets(self, dataset: Dict):
        for row in dataset["rows"]:
            sizes, entries = self._find_uploadable(row)
            non_uploadable = self._calculate_data_size(row) - sum(sizes)
            payload_limit = max(0, self.threshold_kbytes * 1000 - non_uploadable)
            take = greedy_fill(payload_limit, sizes)
            upload = set(range(len(entries))) - set(take)
            for entry in (entries[i] for i in upload):
                self._upload_image(entry)
        return dataset

    def _calculate_data_size(self, data):
        return len(json.dumps(data))

    def _find_uploadable(self, row):
        found = zip(
            *(
                (self._calculate_data_size(entry), entry)
                for entry in row.get("text_input", {})
                if "image_url" in entry
            )
        )
        found = list(found)
        if not found:
            return [], []
        else:
            return found

    def _decode_base64_img_url(self, data):
        prefix, payload = data.split(";")
        _, img_format = prefix.split("/")
        _, img_base64 = payload.split(",")
        img = base64.b64decode(img_base64)
        return img_format, img

    def _upload_image_to_nvcf(self, data, img_format):
        json = {
            "contentType": f"image/{img_format}",
            "description": "GenAI-perf synthetic image",
        }
        new_asset_resp = requests.post(
            f"{NVCF_URL}/assets", headers=self._headers, json=json
        ).json()
        upload_headers = {
            "Content-Type": json["contentType"],
            "x-amz-meta-nvcf-asset-description": json["description"],
        }
        upload_resp = requests.put(
            new_asset_resp["uploadUrl"], headers=upload_headers, data=data
        )
        print(
            f"Uploaded asset {new_asset_resp['assetId']} with status {upload_resp.status_code}"
        )
        return new_asset_resp["assetId"]

    def _upload_image(self, data):
        img_format, img = self._decode_base64_img_url(data["image_url"]["url"])

        start_time = time.perf_counter()
        asset_id = self._upload_image_to_nvcf(img, img_format)
        data["image_url"]["url"] = f"data:image/{img_format};asset_id,{asset_id}"
        end_time = time.perf_counter()
        self._add_upload_report_entry(asset_id, end_time - start_time)
