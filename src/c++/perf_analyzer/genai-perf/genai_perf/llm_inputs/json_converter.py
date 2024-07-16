# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, List


class JSONConverter:
    """
    This class converts the dataset into a generic format that
    is agnostic of the data source.
    """

    @staticmethod
    def to_generic(dataset: List[Dict[str, Any]]) -> Dict:
        if isinstance(dataset, list) and len(dataset) > 0:
            if isinstance(dataset[0], dict):
                converted_data = []
                for item in dataset:
                    row_data = {
                        "text_input": item.get("text_input", ""),
                        "system_prompt": item.get("system_prompt", ""),
                        "response": item.get("response", ""),
                    }
                    converted_data.append(row_data)
                return {
                    "features": ["text_input", "system_prompt", "response"],
                    "rows": [{"row": item} for item in converted_data],
                }
            elif isinstance(dataset[0], str):
                # Assume dataset is a list of strings
                return {
                    "features": ["text_input"],
                    "rows": [{"row": {"text_input": item}} for item in dataset],
                }
            else:
                raise ValueError(
                    f"Dataset is not in a recognized format. Dataset: `{dataset}`"
                )
        else:
            raise ValueError(
                f"Dataset is empty or not in a recognized format. Dataset: `{dataset}`"
            )
