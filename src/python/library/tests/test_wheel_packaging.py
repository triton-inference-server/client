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

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

LIBRARY_DIR = Path(__file__).resolve().parents[1]


class TestWheelPackaging(unittest.TestCase):
    """
    Packaging checks for the tritonclient wheel.
    """

    def _build_wheel(self, dest_dir):
        env = os.environ.copy()
        env["VERSION"] = "0.0.0"
        subprocess.run(
            [
                sys.executable,
                "setup.py",
                "bdist_wheel",
                "--dist-dir",
                str(dest_dir / "dist"),
                "--bdist-dir",
                str(dest_dir / "bdist"),
            ],
            cwd=dest_dir,
            env=env,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        wheels = list((dest_dir / "dist").glob("tritonclient-*.whl"))
        self.assertEqual(len(wheels), 1, "expected exactly one tritonclient wheel")
        return wheels[0]

    def test_license_only_in_dist_info(self):
        """
        LICENSE.txt must ship in dist-info only.

        setuptools data_files with an empty destination installs the file at
        sys.prefix (the venv root), which is the extra LICENSE.txt reported in
        https://github.com/triton-inference-server/client/issues/858.
        """

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            shutil.copy(LIBRARY_DIR / "setup.py", tmp_path / "setup.py")
            shutil.copy(LIBRARY_DIR / "LICENSE.txt", tmp_path / "LICENSE.txt")
            shutil.copytree(LIBRARY_DIR / "requirements", tmp_path / "requirements")

            wheel = self._build_wheel(tmp_path)
            with zipfile.ZipFile(wheel) as archive:
                names = archive.namelist()

            license_names = [name for name in names if name.endswith("LICENSE.txt")]
            self.assertTrue(
                license_names, "wheel must include LICENSE.txt in dist-info"
            )
            for name in license_names:
                self.assertIn(
                    ".dist-info/",
                    name,
                    "LICENSE.txt must not be installed outside dist-info, found {}".format(
                        name
                    ),
                )


if __name__ == "__main__":
    unittest.main()
