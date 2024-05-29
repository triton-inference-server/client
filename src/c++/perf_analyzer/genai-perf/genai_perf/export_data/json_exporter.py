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


import json
from enum import Enum
from typing import Dict

import genai_perf.logging as logging

DEFAULT_OUTPUT_DATA_JSON = "profile_export_genai_perf.json"

logger = logging.getLogger(__name__)


class JsonExporter:
    """
    A class to export the statistics and arg values in a json format.
    """

    def __init__(self, config: Dict):
        self._stats: Dict = config["stats"]
        self._args = dict(vars(config["args"]))
        self._extra_inputs = config["extra_inputs"]
        self._output_dir = config["artifact_dir"]
        self._stats_and_args: Dict = {}
        self._prepare_args_for_export()
        self._merge_stats_and_args()

    def export(self) -> None:
        filename = self._output_dir / DEFAULT_OUTPUT_DATA_JSON
        logger.info(f"Generating {filename}")
        with open(str(filename), "w") as f:
            f.write(json.dumps(self._stats_and_args, indent=2))

    def _prepare_args_for_export(self) -> None:
        del self._args["func"]
        del self._args["output_format"]
        self._args["profile_export_file"] = str(self._args["profile_export_file"])
        self._args["artifact_dir"] = str(self._args["artifact_dir"])
        for k, v in self._args.items():
            if isinstance(v, Enum):
                self._args[k] = v.name.lower()
        self._add_extra_inputs_to_args()

    def _add_extra_inputs_to_args(self) -> None:
        del self._args["extra_inputs"]
        self._args.update({"extra_inputs": self._extra_inputs})

    def _merge_stats_and_args(self) -> None:
        self._stats_and_args = dict(self._stats)
        self._stats_and_args.update({"input_config": self._args})
