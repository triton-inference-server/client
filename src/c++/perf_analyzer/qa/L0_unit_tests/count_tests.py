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
import importlib
import inspect
import os
import sys

sys.path.insert(0, "../../")


def args():
    parser = argparse.ArgumentParser("test_counter")
    parser.add_argument("--path", help="Path to use for counting the tests", type=str)
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    number_of_tests = 0
    opt = args()
    path = opt.path

    for file_path in os.listdir(path):
        # All the test files start with "Test"
        if file_path.startswith("test_"):
            module_name = "tests." + file_path.split(".")[0]
            module = importlib.import_module(module_name)
            classes = inspect.getmembers(module, inspect.isclass)
            for class_tuple in classes:
                class_name = class_tuple[0]
                class_object = class_tuple[1]

                # All the test classes start with "Test"
                if class_name.startswith("Test"):
                    methods = inspect.getmembers(class_object, inspect.isroutine)
                    for method_tuple in methods:
                        method_name = method_tuple[0]
                        if method_name.startswith("test_"):
                            number_of_tests += 1

    # Print the number of tests
    print(number_of_tests)
