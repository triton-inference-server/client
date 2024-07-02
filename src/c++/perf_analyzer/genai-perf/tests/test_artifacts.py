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

import pytest
from genai_perf.main import create_artifacts_dirs


@pytest.fixture
def mock_makedirs(mocker):
    return mocker.patch("os.makedirs")


def test_create_artifacts_dirs_custom_path(mock_makedirs):
    artifacts_dir_path = "/genai_perf_artifacts"
    mock_args = Namespace(artifact_dir=Path(artifacts_dir_path), generate_plots=True)
    create_artifacts_dirs(mock_args)
    mock_makedirs.assert_any_call(
        Path(artifacts_dir_path), exist_ok=True
    ), f"Expected os.makedirs to create artifacts directory inside {artifacts_dir_path} path."
    mock_makedirs.assert_any_call(
        Path(artifacts_dir_path) / "plots", exist_ok=True
    ), f"Expected os.makedirs to create plots directory inside {artifacts_dir_path}/plots path."
    assert mock_makedirs.call_count == 2


def test_create_artifacts_disable_generate_plots(mock_makedirs):
    artifacts_dir_path = "/genai_perf_artifacts"
    mock_args = Namespace(artifact_dir=Path(artifacts_dir_path))
    create_artifacts_dirs(mock_args)
    mock_makedirs.assert_any_call(
        Path(artifacts_dir_path), exist_ok=True
    ), f"Expected os.makedirs to create artifacts directory inside {artifacts_dir_path} path."
    assert mock_makedirs.call_count == 1
