#!/usr/bin/env python3

# Copyright 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import base64
import unittest
from unittest.mock import mock_open, patch

from model_analyzer.reports.html_report import HTMLReport
from model_analyzer.result.result_table import ResultTable

from .common import test_result_collector as trc


class TestHTMLReportMethods(trc.TestResultCollector):
    """
    Tests the methods of the HTMLReport class
    """

    def setUp(self):
        self.maxDiff = None
        self.report = HTMLReport()

    def tearDown(self):
        patch.stopall()

    def test_add_title(self):
        self.report.add_title("Test HTML Report")
        expected_report_body = "<html><head><style></style></head><body><center><h1>Test HTML Report</h1></center></body></html>"
        self.assertEqual(self.report.document(), expected_report_body)

    def test_add_subheading(self):
        # Add one subheading
        self.report.add_subheading("Throughput vs. Latency")
        expected_report_body = "<html><head><style></style></head><body><h3>Throughput vs. Latency</h3></body></html>"
        self.assertEqual(self.report.document(), expected_report_body)

        # Add another subheading
        self.report.add_subheading("GPU Memory vs. Latency")
        expected_report_body = (
            "<html><head><style></style></head><body><h3>Throughput vs. Latency</h3>"
            "<h3>GPU Memory vs. Latency</h3></body></html>"
        )
        self.assertEqual(self.report.document(), expected_report_body)

    def test_add_image(self):
        with patch(
            "model_analyzer.reports.html_report.open",
            mock_open(read_data=bytes(">:(".encode("ascii"))),
        ):
            self.report.add_images(["test_image_file"], ["test_caption"])
        img_content = base64.b64encode(bytes(">:(".encode("ascii"))).decode("ascii")
        expected_report_body = (
            f'<html><head><style></style></head><body><center><div><div class="image" style="float:center;width:100%"><img src="data:image/png;base64,{img_content}"'
            ' style="width:100%"><center><div style="font-weight:bold;font-size:12;padding-bottom:20px">'
            "test_caption</div></center></div></div></center></body></html>"
        )
        self.assertEqual(self.report.document(), expected_report_body)

    def test_add_paragraph(self):
        test_paragraph = (
            "This is a test paragraph with a lot to say."
            " There is more than one line in this paragraph."
        )

        # Default font size
        self.report.add_paragraph(test_paragraph)
        expected_report_body = (
            '<html><head><style></style></head><body><div style="font-size:14">'
            f"<p>{test_paragraph}</p></div></body></html>"
        )
        self.assertEqual(self.report.document(), expected_report_body)

        # custom font size
        self.report.add_paragraph(test_paragraph, font_size=20)
        expected_report_body = (
            '<html><head><style></style></head><body><div style="font-size:14">'
            f'<p>{test_paragraph}</p></div><div style="font-size:20">'
            f"<p>{test_paragraph}</p></div></body></html>"
        )
        self.assertEqual(self.report.document(), expected_report_body)

    def test_add_table(self):
        result_table = ResultTable(["header1", "header2"])
        # Try empty table
        self.report.add_table(table=result_table)
        table_style = "border: 1px solid black;border-collapse: collapse;text-align: center;width: 80%;padding: 5px 10px;font-size: 11pt"
        expected_report_body = (
            "<html><head><style></style></head><body>"
            f'<center><table style="{table_style}">'
            "<tr>"
            f'<th style="{table_style}">header1</th>'
            f'<th style="{table_style}">header2</th>'
            "</tr>"
            "</table></center>"
            "</body></html>"
        )
        self.assertEqual(self.report.document(), expected_report_body)

        # Fill table
        for i in range(2):
            result_table.insert_row_by_index([f"value{i}1", f"value{i}2"])

        # Table has 5 rows
        self.report.add_table(table=result_table)
        expected_report_body = (
            "<html><head><style></style></head><body>"
            f'<center><table style="{table_style}">'
            "<tr>"
            f'<th style="{table_style}">header1</th>'
            f'<th style="{table_style}">header2</th>'
            "</tr>"
            "</table></center>"
            f'<center><table style="{table_style}">'
            "<tr>"
            f'<th style="{table_style}">header1</th>'
            f'<th style="{table_style}">header2</th>'
            "</tr>"
            "<tr>"
            f'<td style="{table_style}">value01</td>'
            f'<td style="{table_style}">value02</td>'
            "</tr>"
            "<tr>"
            f'<td style="{table_style}">value11</td>'
            f'<td style="{table_style}">value12</td>'
            "</tr>"
            "</table></center>"
            "</body></html>"
        )

        self.assertEqual(self.report.document(), expected_report_body)

    def test_write_report(self):
        self.report.add_title("Test HTML Report")
        self.report.add_subheading("Throughput vs. Latency")
        test_paragraph = (
            "This is a test paragraph with a lot to say."
            " There is more than one line in this paragraph."
        )
        self.report.add_paragraph(test_paragraph, font_size=14)

        expected_report_body = (
            "<html><head><style></style></head><body><center><h1>Test HTML Report</h1></center><h3>Throughput vs. Latency</h3>"
            f'<div style="font-size:14"><p>{test_paragraph}</p></div>'
            "</body></html>"
        )

        with patch("builtins.open", mock_open()) as mocked_file:
            self.report.write_report("test_report_filename")
        mocked_file.assert_called_once_with(f"test_report_filename", "w")
        mocked_file().write.assert_called_once_with(f"{expected_report_body}")


if __name__ == "__main__":
    unittest.main()
