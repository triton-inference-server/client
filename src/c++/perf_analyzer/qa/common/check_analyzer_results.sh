#!/bin/bash
# Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

source ../common/util.sh

# Check Python unittest results.
function check_unit_test_results () {
    local log_file=$1
    local expected_num_tests=$2

    if [[ -z "$expected_num_tests" ]]; then
        echo "=== expected number of tests must be defined"
        return 1
    fi

    local num_failures=`cat $log_file | grep -E ".*total.*errors.*failures.*" | tail -n 1 | jq .failures`
    local num_tests=`cat $log_file | grep -E ".*total.*errors.*failures.*" | tail -n 1 | jq .total`
    local num_errors=`cat $log_file | grep -E ".*total.*errors.*failures.*" | tail -n 1 | jq .errors`

    # Number regular expression
    re='^[0-9]+$'

    if [[ $? -ne 0 ]] || ! [[ $num_failures =~ $re ]] || ! [[ $num_tests =~ $re ]] || \
     ! [[ $num_errors =~ $re ]]; then
        cat $log_file
        echo -e "\n***\n*** Test Failed: unable to parse test results\n***" >> $log_file
        return 1
    fi
    if [[ $num_errors != "0" ]] || [[ $num_failures != "0" ]] || [[ $num_tests -ne $expected_num_tests ]]; then
        cat $log_file
        echo -e "\n***\n*** Test Failed: Expected $expected_num_tests test(s), $num_tests test(s) executed, $num_errors test(s) had error, and $num_failures test(s) failed. \n***" >> $log_file
        return 1
    fi

    return 0
}

# Check the table(s) by calling check_log_table_row_column and check_csv_table_row_column
# See if the logic generalizes to the specify testing needs
# Call check_log_table_row_column and check_csv_table_row_column in util.sh directly if not
function check_table_row_column() {
    local inference_log_file=$1
    local gpu_log_file=$2
    local server_log_file=$3
    local inference_csv_file=$4
    local gpu_csv_file=$5
    local server_csv_file=$6

    local inference_expected_num_columns=$7
    local inference_expected_num_rows=$8
    local gpu_expected_num_columns=$9
    local gpu_expected_num_rows=${10}
    local server_expected_num_columns=${11}
    local server_expected_num_rows=${12}

    local ret=0

    if [ "$inference_log_file" != "" ]; then
        check_log_table_row_column $inference_log_file $inference_expected_num_columns $inference_expected_num_rows "Models\ \(Inference\):"
        if [ $? -ne 0 ]; then
            echo -e "\n***\n*** Test Output Verification Failed for $inference_log_file.\n***"
            ret=1
        fi
    fi
    if [ "$gpu_log_file" != "" ]; then
        check_log_table_row_column $gpu_log_file $gpu_expected_num_columns $gpu_expected_num_rows "Models\ \(GPU\ Metrics\):"
        if [ $? -ne 0 ]; then
            echo -e "\n***\n*** Test Output Verification Failed for $gpu_log_file.\n***"
            ret=1
        fi
    fi
    if [ "$server_log_file" != "" ]; then
        check_log_table_row_column $server_log_file $server_expected_num_columns $server_expected_num_rows "Server\ Only:"
        if [ $? -ne 0 ]; then
            echo -e "\n***\n*** Test Output Verification Failed for $server_log_file.\n***"
            ret=1
        fi
    fi

    if [ "$inference_csv_file" != "" ]; then
        check_csv_table_row_column $inference_csv_file $inference_expected_num_columns $inference_expected_num_rows "Model"
        if [ $? -ne 0 ]; then
            echo -e "\n***\n*** Test Output Verification Failed for $inference_csv_file.\n***"
            ret=1
        fi
    fi
    if [ "$gpu_csv_file" != "" ]; then
        check_csv_table_row_column $gpu_csv_file $gpu_expected_num_columns $gpu_expected_num_rows "Model"
        if [ $? -ne 0 ]; then
            echo -e "\n***\n*** Test Output Verification Failed for $gpu_csv_file.\n***"
            ret=1
        fi
    fi
    if [ "$server_csv_file" != "" ]; then
        check_csv_table_row_column $server_csv_file $server_expected_num_columns $server_expected_num_rows "Model"
        if [ $? -ne 0 ]; then
            echo -e "\n***\n*** Test Output Verification Failed for $server_csv_file.\n***"
            ret=1
        fi
    fi

    return $ret
}
