#!/bin/bash
# Copyright (c) 2020-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

EXPECTED_NUM_TESTS=`python3 count_tests.py --path ../../tests/`
source ../common/check_analyzer_results.sh
source ../common/util.sh

create_logs_dir "L0_unit_tests"
create_result_paths -export-path false -checkpoints false

RET=0

set +e
coverage run --branch --source=../../model_analyzer -m unittest discover -v -s ../../tests  -t ../../ > $ANALYZER_LOG 2>&1
if [ $? -ne 0 ]; then
    RET=1
else
    check_unit_test_results $ANALYZER_LOG $EXPECTED_NUM_TESTS
    if [ $? -ne 0 ]; then
        cat $ANALYZER_LOG
        echo -e "\n***\n*** Test Result Verification Failed\n***"
        RET=1
    fi
fi
set -e

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test PASSED\n***"
else
    cat $ANALYZER_LOG
    echo -e "\n***\n*** Test FAILED\n***"
fi

# Generate html files
echo `pwd`
coverage html --directory $LOGS_DIR/html/

exit $RET
