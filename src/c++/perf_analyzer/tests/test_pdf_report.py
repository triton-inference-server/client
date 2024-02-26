#!/usr/bin/env python3

# Copyright 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from unittest.mock import MagicMock, patch

from model_analyzer.reports.pdf_report import PDFReport

from .common import test_result_collector as trc


class TestPDFReportMethods(trc.TestResultCollector):
    """
    Tests the methods of the PDFReport class
    """

    def setUp(self):
        self.maxDiff = None
        self.report = PDFReport()

    def tearDown(self):
        patch.stopall()

    def test_write_report(self):
        with patch(
            "model_analyzer.reports.pdf_report.pdfkit", MagicMock()
        ) as pdfkit_mock:
            self.report.add_title("Test PDF Report")
            self.report.add_subheading("Throughput vs. Latency")
            test_paragraph = (
                "This is a test paragraph with a lot to say."
                " There is more than one line in this paragraph."
            )
            self.report.add_paragraph(test_paragraph, font_size=14)
            self.report.write_report("test_report_filename")

            expected_report_body = (
                "<html><head><style></style></head><body><center><h1>Test PDF Report</h1></center><h3>Throughput vs. Latency</h3>"
                f'<div style="font-size:14"><p>{test_paragraph}</p></div>'
                "</body></html>"
            )
            pdfkit_mock.from_string.assert_called_with(
                expected_report_body, "test_report_filename", options={"quiet": ""}
            )


if __name__ == "__main__":
    unittest.main()
