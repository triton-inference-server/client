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

import unittest
from unittest.mock import patch

from model_analyzer.record.record_aggregator import RecordAggregator
from model_analyzer.record.types.gpu_utilization import GPUUtilization
from model_analyzer.record.types.perf_latency_p99 import PerfLatencyP99
from model_analyzer.record.types.perf_throughput import PerfThroughput

from .common import test_result_collector as trc


class TestRecordAggregatorMethods(trc.TestResultCollector):
    def tearDown(self):
        patch.stopall()

    def test_insert(self):
        record_aggregator = RecordAggregator()

        self.assertEqual(record_aggregator.total(), 0)

        throughput_record = PerfThroughput(5)
        record_aggregator.insert(throughput_record)

        # Assert record is added
        self.assertEqual(record_aggregator.total(), 1)

    def test_record_types(self):
        record_aggregator = RecordAggregator()

        throughput_record = PerfThroughput(5)
        record_aggregator.insert(throughput_record)

        self.assertEqual(record_aggregator.record_types()[0], PerfThroughput)

    def test_filter_records_default(self):
        record_aggregator = RecordAggregator()

        # insert throughput record and check its presence
        throughput_record = PerfThroughput(5)
        record_aggregator.insert(throughput_record)

        # Get the record
        retrieved_records = record_aggregator.filter_records().get_records()
        retrieved_throughput = retrieved_records[PerfThroughput][0]

        self.assertIsInstance(
            retrieved_throughput,
            PerfThroughput,
            msg="Record types do not match after filter_records",
        )

        self.assertEqual(
            retrieved_throughput.value(),
            throughput_record.value(),
            msg="Values do not match after filter_records",
        )

    def test_filter_records_filtered(self):
        record_aggregator = RecordAggregator()

        # Test for malformed inputs
        with self.assertRaises(Exception):
            record_aggregator.filter_records(filters=[(lambda x: False)])
        with self.assertRaises(Exception):
            record_aggregator.filter_records(
                record_types=[None, None], filters=[(lambda x: False)]
            )

        # Insert 3 throughputs
        record_aggregator.insert(PerfThroughput(5))
        record_aggregator.insert(PerfThroughput(1))
        record_aggregator.insert(PerfThroughput(10))

        # Test get with filters
        retrieved_records = record_aggregator.filter_records(
            record_types=[PerfThroughput], filters=[(lambda v: v.value() >= 5)]
        ).get_records()

        # Should return 2 records
        self.assertEqual(len(retrieved_records[PerfThroughput]), 2)
        retrieved_values = [
            record.value() for record in retrieved_records[PerfThroughput]
        ]
        self.assertIn(5, retrieved_values)
        self.assertIn(10, retrieved_values)

        # Insert 2 Latency records
        record_aggregator.insert(PerfLatencyP99(3))
        record_aggregator.insert(PerfLatencyP99(6))

        # Test get with multiple headers
        retrieved_records = record_aggregator.filter_records(
            record_types=[PerfLatencyP99, PerfThroughput],
            filters=[(lambda v: v.value() == 3), (lambda v: v.value() < 5)],
        ).get_records()

        retrieved_values = {
            record_type: [record.value() for record in retrieved_records[record_type]]
            for record_type in [PerfLatencyP99, PerfThroughput]
        }

        self.assertEqual(len(retrieved_records[PerfLatencyP99]), 1)
        self.assertIn(3, retrieved_values[PerfLatencyP99])

        self.assertEqual(len(retrieved_records[PerfThroughput]), 1)
        self.assertIn(1, retrieved_values[PerfThroughput])

    def test_groupby(self):
        record_aggregator = RecordAggregator()
        # Insert 3 throughputs
        record_aggregator.insert(PerfThroughput(5, timestamp=0))
        record_aggregator.insert(PerfThroughput(1, timestamp=1))
        record_aggregator.insert(PerfThroughput(10, timestamp=1))

        def groupby_criteria(record):
            return record.timestamp()

        records = record_aggregator.groupby([PerfThroughput], groupby_criteria)
        self.assertEqual(list(records[PerfThroughput]), [0, 1])
        self.assertEqual(
            list(records[PerfThroughput].values()),
            [PerfThroughput(5.0), PerfThroughput(10.0)],
        )

        records = record_aggregator.groupby([PerfThroughput], groupby_criteria)
        self.assertEqual(list(records[PerfThroughput]), [0, 1])
        self.assertEqual(
            list(records[PerfThroughput].values()),
            [PerfThroughput(5.0), PerfThroughput(10.0)],
        )

    def test_aggregate(self):
        record_aggregator = RecordAggregator()

        # Insert 10 records
        for i in range(10):
            record_aggregator.insert(PerfThroughput(i))

        for i in range(10):
            record_aggregator.insert(GPUUtilization(i))
        # Aggregate them with max, min and average
        max_vals = record_aggregator.aggregate(record_types=[PerfThroughput])
        avg_vals = record_aggregator.aggregate(record_types=[GPUUtilization])

        self.assertEqual(
            max_vals[PerfThroughput],
            PerfThroughput(9),
            msg="Aggregation failed with max",
        )

        self.assertEqual(
            avg_vals[GPUUtilization],
            GPUUtilization(4.5),
            msg="Aggregation failed with max",
        )


if __name__ == "__main__":
    unittest.main()
