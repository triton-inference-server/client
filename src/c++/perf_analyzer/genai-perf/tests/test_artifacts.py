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

from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pytest
from genai_perf.constants import DEFAULT_ARTIFACT_DIR
from genai_perf.main import create_artifacts_dirs


def test_create_artifacts_dirs(mocker):
    mock_makedirs = mocker.patch("os.makedirs")
    mock_args = Namespace(artifact_dir=Path(DEFAULT_ARTIFACT_DIR))
    create_artifacts_dirs(mock_args)
    mock_makedirs.assert_any_call(
        Path(DEFAULT_ARTIFACT_DIR), exist_ok=True
    ), f"Expected os.makedirs to be called with {DEFAULT_ARTIFACT_DIR} and exist_ok=True"
    mock_makedirs.assert_any_call(
        Path(Path(DEFAULT_ARTIFACT_DIR)) / "plots", exist_ok=True
    ), f"Expected os.makedirs to be called with {DEFAULT_ARTIFACT_DIR}/plots and exist_ok=True"
    assert mock_makedirs.call_count == 2
