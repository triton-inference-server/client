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
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import genai_perf.logging as logging

# Skip type checking to avoid mypy error
# Issue: https://github.com/python/mypy/issues/10632
import yaml  # type: ignore

logger = logging.getLogger(__name__)


def remove_sse_prefix(msg: str) -> str:
    prefix = "data: "
    if msg.startswith(prefix):
        return msg[len(prefix) :].strip()
    return msg.strip()


def load_yaml(filepath: Path) -> Dict[str, Any]:
    with open(str(filepath)) as f:
        configs = yaml.safe_load(f)
    return configs


def load_json(filepath: Path) -> Dict[str, Any]:
    with open(str(filepath), encoding="utf-8", errors="ignore") as f:
        content = f.read()
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            snippet = content[:200] + ("..." if len(content) > 200 else "")
            logger.error("Failed to parse JSON string: '%s'", snippet)
            raise


def remove_file(file: Path) -> None:
    if file.is_file():
        file.unlink()


def convert_option_name(name: str) -> str:
    return name.replace("_", "-")


def get_enum_names(enum: Type[Enum]) -> List:
    names = []
    for e in enum:
        names.append(e.name.lower())
    return names


def get_enum_entry(name: str, enum: Type[Enum]) -> Optional[Enum]:
    for e in enum:
        if e.name.lower() == name.lower():
            return e
    return None


def scale(value, factor):
    return value * factor
