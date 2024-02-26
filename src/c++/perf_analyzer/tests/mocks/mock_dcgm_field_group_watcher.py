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

import time
from collections import defaultdict
from unittest.mock import MagicMock

from .mock_dcgm_agent import MockDCGMAgent

TEST_RECORD_VALUE = 2.4


class MockDCGMFieldGroupWatcherHelper:
    """
    Mock of the DCGMFieldGroupWatcher class
    """

    def __init__(
        self,
        handle,
        group_id,
        field_group,
        operation_mode,
        update_freq,
        max_keep_age,
        max_keep_samples,
        start_timestamp,
    ):
        """
        handle : dcgm_handle
            DCGM handle from dcgm_agent.dcgmInit()
        groupId : int
            a DCGM group ID returned from dcgm_agent.dcgmGroupCreate
        fieldGroup : int
            DcgmFieldGroup() instance to watch fields for
        operationMode : dcgm_structs.DCGM_OPERATION_MODE
            a dcgm_structs.DCGM_OPERATION_MODE_? constant for if the host
            engine is running in lock step or auto mode
        updateFreq : float
            how often to update each field in usec
        maxKeepAge : int
            how long DCGM should keep values for in seconds
        maxKeepSamples : int
            is the maximum number of samples DCGM should ever cache for each
            field
        startTimestamp : int
            a base timestamp we should start from when first reading
            values. This can be used to resume a previous instance of a
            DcgmFieldGroupWatcher by using its _nextSinceTimestamp. 0=start
            with all cached data
        """

        self._handle = handle
        self._group_id = group_id.value
        self._field_group = field_group
        self._operation_mode = operation_mode
        self._update_freq = update_freq
        self._max_keep_age = max_keep_age
        self._max_keep_samples = max_keep_samples
        self._start_timestamp = start_timestamp
        self.values = defaultdict(lambda: defaultdict(MagicMock))

    def GetMore(self):
        """
        This function performs a single iteration of monitoring
        """

        group_name = list(MockDCGMAgent.device_groups)[self._group_id]
        device_group = MockDCGMAgent.device_groups[group_name]
        field_group_name = list(MockDCGMAgent.field_groups)[self._field_group]

        for device in device_group:
            for field in MockDCGMAgent.field_groups[field_group_name]:
                # Sample Record
                record = MagicMock()
                record.value = TEST_RECORD_VALUE
                record.ts = int(time.time() * 1e6)
                if not isinstance(self.values[device][field].values, list):
                    self.values[device][field].values = [record]
                else:
                    self.values[device][field].values.append(record)
