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

from unittest.mock import MagicMock

import model_analyzer.monitor.dcgm.dcgm_structs as structs
from model_analyzer.model_analyzer_exceptions import TritonModelAnalyzerException

TEST_PCI_BUS_ID = "00000000:6A:00.0"
TEST_UUID = "dddddddd-bbbb-aaaa-cccc-ffffffffffff"


class MockDCGMAgent:
    device_groups = {}
    field_groups = {}
    devices = [{"pci_bus_id": TEST_PCI_BUS_ID, "uuid": TEST_UUID}]

    @staticmethod
    def dcgmInit():
        """
        Mock dcgmInit method
        """

        pass

    @staticmethod
    def dcgmStartEmbedded(op_mode):
        """
        Mock dcgmStartEmbedded

        Parameters
        ----------
        op_mode : int
            DCGM operational mode
        """

        pass

    @staticmethod
    def dcgmGetAllSupportedDevices(dcgm_handle):
        """
        Mock dcgmGetAllSupportedDevices

        Parameters
        ----------
        dcgm_handle : ctypes.c_void_p
            A DCGM Handle object
        """

        return list(range(0, len(MockDCGMAgent.devices)))

    @staticmethod
    def dcgmGetDeviceAttributes(dcgm_handle, device):
        """
        Mock dcgmGetDeviceAttributes

        Parameters
        ----------
        dcgm_handle : ctypes.c_void_p
            A DCGM Handle object
        device : int
            device id

        Returns
        -------
        MagicMock
            A MagicMock containing the device attributes
        """

        gpu_device = MockDCGMAgent.devices[device]
        device_attribute = MagicMock()
        device_attribute.identifiers.pciBusId = gpu_device["pci_bus_id"]
        device_attribute.identifiers.uuid = gpu_device["uuid"]
        return device_attribute

    @staticmethod
    def dcgmGroupCreate(dcgm_handle, type_name, name):
        """
        Mock dcgmGroupCreate

        Parameters
        ----------
        dcgm_handle : ctypes.c_void_p
            A DCGM Handle object
        type_name : int
            Group type
        name : str
            Device group name

        Returns
        -------
        int
            Returns the group id

        Raises
        ------
        KeyError
            If the group already exists it raises a KeyError
        """

        if not (name in MockDCGMAgent.device_groups):
            if type_name == structs.DCGM_GROUP_EMPTY:
                MockDCGMAgent.device_groups[name] = []
                group_id = MagicMock()
                group_id.value = list(MockDCGMAgent.device_groups).index(name)
                return group_id
        else:
            raise KeyError

    @staticmethod
    def dcgmGroupAddDevice(dcgm_handle, group_id, gpu_device_id):
        """
        Mock dcgmGroupAddDevice

        Parameters
        ----------
        dcgm_handle : ctypes.c_void_p
            A DCGM Handle object
        group_id : int
            Group type
        gpu_device_id : int
            GPU device id

        Raises
        ------
        KeyError
            If the group does not exist
        """

        group_id = group_id.value
        if group_id >= len(list(MockDCGMAgent.device_groups)):
            raise KeyError

        device_group_name = list(MockDCGMAgent.device_groups)[group_id]
        device_group = MockDCGMAgent.device_groups[device_group_name]

        if gpu_device_id in device_group:
            raise TritonModelAnalyzerException(
                f"GPU device {gpu_device_id} already exists in the device group"
            )

        device_group.append(gpu_device_id)

    @staticmethod
    def dcgmFieldGroupCreate(dcgm_handle, fields, name):
        """
        Mock dcgmFieldGroupCreate

        Parameters
        ----------
        dcgm_handle : ctypes.c_void_p
            A DCGM Handle object
        fields : list
            List of ints containing the fields to be monitored
        name : str
            Group name

        Returns
        -------
        int
            Returns the group id

        Raises
        ------
        KeyError
            If the group already exists it raises a KeyError
        """

        if not (name in MockDCGMAgent.field_groups):
            MockDCGMAgent.field_groups[name] = fields
            group_id = MagicMock()
            group_id.value = list(MockDCGMAgent.field_groups).index(name)
            return group_id
        else:
            raise KeyError

    @staticmethod
    def dcgmShutdown():
        """
        Mock dcgmShutdown
        """

        MockDCGMAgent.device_groups = {}
        MockDCGMAgent.field_groups = {}
