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

import itertools


class MockRunConfig:
    """
    Mock class that only contains the important values from the model_config and
    perf_config from a ModelRunConfig object
    """

    def load_from_model_run_config(self, model_run_config):
        """Populate from a ModelRunConfig object"""
        model_config = model_run_config.model_config().get_config()
        perf_config = model_run_config.perf_config()
        self._load_from_model_and_perf_config(model_config, perf_config)

    def load_from_dict(self, dict):
        """Populate from a dictionary"""
        self._config = dict

    def get_config_tuple(self):
        """
        Convert and return the data from a dict into a tuple
        of (key,value,key,value,...) where the keys are in a
        sorted order
        """
        list_of_key_value_pairs = [
            (y, self._config[y])
            for y in sorted(self._config.keys())
            if self._config[y] != None
        ]
        out_tuple = tuple(item for x in list_of_key_value_pairs for item in x)
        return out_tuple

    def _load_from_model_and_perf_config(self, model_config, perf_config):
        """
        Given a model configuration and perf configuration, extract and store
        the key bits of information
        """
        max_batch_size = None
        if model_config.get("max_batch_size") is not None:
            max_batch_size = model_config["max_batch_size"]

        instances = None
        kind = None
        if model_config.get("instance_group") is not None:
            instances = model_config["instance_group"][0]["count"]
            kind = model_config["instance_group"][0]["kind"]

        dynamic_batching = None
        max_queue_delay = None
        if model_config.get("dynamic_batching") is not None:
            dynamic_batching = 0
            max_queue_delay = model_config["dynamic_batching"].get(
                "max_queue_delay_microseconds"
            )

        batch_size = perf_config.__getitem__("batch-size")
        concurrency = perf_config.__getitem__("concurrency-range")

        self._config = {
            "kind": kind,
            "batch_sizes": batch_size,
            "batching": dynamic_batching,
            "concurrency": concurrency,
            "instances": instances,
            "max_batch_size": max_batch_size,
            "max_queue_delay": max_queue_delay,
        }


class MockRunConfigs:
    """
    This class holds a list of MockRunConfig
    """

    def __init__(self):
        self._configs = []

    def get_num_configs(self) -> int:
        """Returns the number of configs"""
        return len(self._configs)

    def get_configs_set(self) -> set:
        """Returns a set of the configs"""
        configs_set = {config.get_config_tuple() for config in self._configs}
        return configs_set

    def add_from_model_run_config(self, config):
        """Add a single config from a ModelRunConfig"""

        mock_run_config = MockRunConfig()
        mock_run_config.load_from_model_run_config(config)
        self._configs.append(mock_run_config)

    def add_from_dict(self, config):
        """Add a single config from a dict"""

        mock_run_config = MockRunConfig()
        mock_run_config.load_from_dict(config)
        self._configs.append(mock_run_config)

    def populate_from_ranges(self, ranges):
        """
        Given a dict of key-to-list, create the set of ModelRunConfigs based on
        the full cartesian product of each dict in the list

        For example, passing in
        [{ A: [1,2], B: [3,4], C:[0] },  { A: [5], B: [8,9], C: [10] }]
        would create configs with the following values:
          - A=1, B=3, C=0
          - A=1, B=4, C=0
          - A=2, B=3, C=0
          - A=2, B=4, C=0
          - A=5, B=8, C=10
          - A=5, B=9, C=10
        """
        for ranges_dict in ranges:
            ranges_keys = list(sorted(ranges_dict.keys()))
            value_lists = []
            for key in ranges_keys:
                value_lists.append(ranges_dict[key])
            configs = list(itertools.product(*value_lists))

            for value in configs:
                config_dict = {
                    ranges_keys[i]: value[i] for i in range(len(ranges_keys))
                }
                self.add_from_dict(config_dict)
