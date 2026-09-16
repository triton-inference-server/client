# Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS' AND ANY
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

import os
import re
from itertools import chain

from hatchling.metadata.plugin.interface import MetadataHookInterface

# A pip requirements comment runs from an unquoted "#" at line start or
# preceded by whitespace, through end of line.
_COMMENT = re.compile(r"(^|\s)#.*$")


def _req_file(root, filename, folder="requirements"):
    """Read a pip requirements file into a list of PEP 508 requirements.

    Strips whole-line comments, trailing inline comments and blank lines.
    The setup.py this replaced dropped only lines starting with "#", so
    blank lines became empty strings and an inline comment such as
    "greenlet<3.4.0  # [TRI-964] ..." was passed through verbatim.
    setuptools quietly tolerated both; a PEP 621 dependency list does not.
    """
    with open(os.path.join(root, folder, filename)) as f:
        lines = (_COMMENT.sub("", line).strip() for line in f)
        return [line for line in lines if line]


class CustomMetadataHook(MetadataHookInterface):
    """Resolve tritonclient's version and dependencies at build time.

    The version comes from the VERSION environment variable set by
    build_wheel.py, and the dependency sets are read from requirements/
    so those files remain the single source of truth rather than being
    duplicated into pyproject.toml.
    """

    def update(self, metadata):
        if "VERSION" not in os.environ:
            raise Exception("envvar VERSION must be specified")
        metadata["version"] = os.environ["VERSION"]
        metadata["dependencies"] = _req_file(self.root, "requirements.txt")
        optional = {
            "grpc": _req_file(self.root, "requirements_grpc.txt"),
            "http": _req_file(self.root, "requirements_http.txt"),
            "cuda": _req_file(self.root, "requirements_cuda.txt"),
            "perf_analyzer": _req_file(self.root, "requirements_perf_analyzer.txt"),
        }
        optional["all"] = list(chain.from_iterable(optional.values()))
        metadata["optional-dependencies"] = optional
