#!/usr/bin/env python3

# Copyright 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import argparse
import os
import pathlib
import re

FLAGS = None
SKIP_EXTS = ("pt", "log", "png", "pdf", "ckpt", "csv", "json")

REPO_PATH_FROM_THIS_FILE = "../.."
SKIP_PATHS = (".git", "VERSION", "LICENSE")

COPYRIGHT_YEAR_RE = "Copyright( \\(c\\))? 20[1-9][0-9](-(20)?[1-9][0-9])?(,((20[2-9][0-9](-(20)?[2-9][0-9])?)|([2-9][0-9](-[2-9][0-9])?)))*,? NVIDIA CORPORATION( & AFFILIATES)?. All rights reserved."

COPYRIGHT = """

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

repo_abs_path = (
    pathlib.Path(__file__).parent.joinpath(REPO_PATH_FROM_THIS_FILE).resolve()
)

copyright_year_re = re.compile(COPYRIGHT_YEAR_RE)


def visit(path):
    if FLAGS.verbose:
        print("visiting " + path)

    for skip in SKIP_EXTS:
        if path.endswith("." + skip):
            if FLAGS.verbose:
                print("skipping due to extension: " + path)
            return True

    for skip in SKIP_PATHS:
        if str(pathlib.Path(path).resolve()).startswith(
            str(repo_abs_path.joinpath(skip).resolve())
        ):
            if FLAGS.verbose:
                print("skipping due to path prefix: " + path)
            return True

    with open(path, "r") as f:
        first_line = True
        line = None
        try:
            for fline in f:
                line = fline

                # Skip any '#!', '..', '<!--', or '{{/*' lines at the
                # start of the file
                if first_line:
                    first_line = False
                    if (
                        fline.startswith("#!")
                        or fline.startswith("..")
                        or fline.startswith("<!--")
                        or fline.startswith("{{/*")
                    ):
                        continue
                # Skip empty lines...
                if len(fline.strip()) != 0:
                    break
        except UnicodeDecodeError as ex:
            # If we get this exception on the first line then assume a
            # non-text file.
            if not first_line:
                raise ex
            if FLAGS.verbose:
                print("skipping binary file: " + path)
            return True
        if line is None:
            if FLAGS.verbose:
                print("skipping empty file: " + path)
            return True

        line = line.strip()

        # The next line must be the copyright line with a single year
        # or a year range. It is optionally allowed to have '# ' or
        # '// ' prefix.
        prefix = ""
        if line.startswith("# "):
            prefix = "# "
        elif line.startswith("// "):
            prefix = "// "
        elif not line.startswith(COPYRIGHT_YEAR_RE[0]):
            print(
                "incorrect prefix for copyright line, allowed prefixes '# ' or '// ', for "
                + path
                + ": "
                + line
            )
            return False

        # Check if the copyright year line matches the regex
        # and see if the year(s) are reasonable
        years = []

        copyright_row = line[len(prefix) :]
        if copyright_year_re.match(copyright_row):
            for year in (
                copyright_row.split(
                    "(c) " if "(c) " in copyright_row else "Copyright "
                )[1]
                .split(" NVIDIA ")[0]
                .split(",")
            ):
                if len(year) == 4:  # 2021
                    years.append(int(year))
                elif len(year) == 2:  # 21
                    years.append(int(year) + 2000)
                elif len(year) == 9:  # 2021-2022
                    years.append(int(year[0:4]))
                    years.append(int(year[5:9]))
                elif len(year) == 7:  # 2021-22
                    years.append(int(year[0:4]))
                    years.append(int(year[5:7]) + 2000)
                elif len(year) == 5:  # 21-23
                    years.append(int(year[0:2]) + 2000)
                    years.append(int(year[3:5]) + 2000)
        else:
            print("copyright year is not recognized for " + path + ": " + line)
            return False

        if years[0] > FLAGS.year:
            print(
                "copyright start year greater than current year for "
                + path
                + ": "
                + line
            )
            return False
        if years[-1] > FLAGS.year:
            print(
                "copyright end year greater than current year for " + path + ": " + line
            )
            return False
        for i in range(1, len(years)):
            if years[i - 1] >= years[i]:
                print("copyright years are not increasing for " + path + ": " + line)
                return False

        # Subsequent lines must match the copyright body.
        copyright_body = [
            l.rstrip() for i, l in enumerate(COPYRIGHT.splitlines()) if i > 0
        ]
        copyright_idx = 0
        for line in f:
            if copyright_idx >= len(copyright_body):
                break

            if len(prefix) == 0:
                line = line.rstrip()
            else:
                line = line.strip()

            if len(copyright_body[copyright_idx]) == 0:
                expected = prefix.strip()
            else:
                expected = prefix + copyright_body[copyright_idx]
            if line != expected:
                print("incorrect copyright body for " + path)
                print("  expected: '" + expected + "'")
                print("       got: '" + line + "'")
                return False
            copyright_idx += 1

        if copyright_idx != len(copyright_body):
            print(
                "missing "
                + str(len(copyright_body) - copyright_idx)
                + " lines of the copyright body"
            )
            return False

    if FLAGS.verbose:
        print("copyright correct for " + path)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        required=False,
        default=False,
        help="Enable verbose output",
    )
    parser.add_argument("-y", "--year", type=int, required=True, help="Copyright year")
    parser.add_argument(
        "paths", type=str, nargs="*", default=None, help="Directories or files to check"
    )
    FLAGS = parser.parse_args()

    if FLAGS.paths is None or len(FLAGS.paths) == 0:
        parser.print_help()
        exit(1)

    ret = True
    for path in FLAGS.paths:
        if not os.path.isdir(path):
            if not visit(path):
                ret = False
        else:
            for root, dirs, files in os.walk(path):
                for name in files:
                    if not visit(os.path.join(root, name)):
                        ret = False

    exit(0 if ret else 1)
