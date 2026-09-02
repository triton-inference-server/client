#!/usr/bin/env python3

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

"""Unit tests for the tritonclient Python package layout.

Guards against shipping internal-only directories (e.g. ``tests``) as
top-level packages in the wheel that ``setup.py`` produces. Runs
``setup.py egg_info`` against the real ``setup.py`` and inspects the
generated ``top_level.txt``, so a regression in the ``find_packages``
call shows up as a test failure.
"""

import os
import subprocess
import sys
import tempfile
import unittest

LIBRARY_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


class PackageLayoutTest(unittest.TestCase):
    def _collect_top_level(self):
        with tempfile.TemporaryDirectory() as egg_dir:
            env = os.environ.copy()
            env.setdefault("VERSION", "0.0.0dev")
            result = subprocess.run(
                [sys.executable, "setup.py", "egg_info", "-e", egg_dir],
                cwd=LIBRARY_DIR,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                result.returncode,
                0,
                f"setup.py egg_info failed: {result.stderr}",
            )
            top_level = os.path.join(egg_dir, "tritonclient.egg-info", "top_level.txt")
            with open(top_level) as f:
                return [line.strip() for line in f if line.strip()]

    def test_tests_package_not_in_top_level(self):
        # The ``tests`` directory next to ``setup.py`` exists only to host
        # the in-tree unit tests; it must not ship as a top-level package
        # in the wheel published to PyPI.
        pkgs = self._collect_top_level()
        self.assertNotIn(
            "tests",
            pkgs,
            "setup.py is exporting the in-tree 'tests' package; "
            "exclude it via find_packages(exclude=['tests', 'tests.*']).",
        )

    def test_tritonclient_in_top_level(self):
        # Sanity check: the exclusion must not strip out the real client
        # packages.
        pkgs = self._collect_top_level()
        self.assertIn("tritonclient", pkgs)


if __name__ == "__main__":
    unittest.main()
