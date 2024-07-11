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

import json
from pathlib import Path
from typing import Dict

from genai_perf.constants import DEFAULT_INPUT_DATA_JSON


class JSONWriter:
    @staticmethod
    def write_to_file(json_data: Dict, output_dir: Path) -> None:
        filename = output_dir / DEFAULT_INPUT_DATA_JSON
        with open(filename, "w") as f:
            f.write(json.dumps(json_data, indent=2))
