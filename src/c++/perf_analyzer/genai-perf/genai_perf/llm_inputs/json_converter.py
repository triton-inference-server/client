# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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
