# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import json
import yaml

import genai_perf.logging as logging
from genai_perf.export_data.exporter_config import ExporterConfig


DEFAULT_OUTPUT_DATA_YAML = "profile_export_genai_perf.yaml"
SCHEMA_TYPE = "measurement"
BENCHMARK_NAME = "abc"

logger = logging.getLogger(__name__)

class YamlExporter:
    """
    A class to export the statistics and arg values in a yaml format.
    """

    def __init__(self,config: ExporterConfig):
        self._benchmark_metadata = {}
        self._benchmark_measurement = dict(config.stats)
        self._benchmark_config = dict(vars(config.args))
        self._extra_inputs = dict(config.extra_inputs)
        self._output_dir = config.artifact_dir
        self._benchmark_data: Dict = {}
        self._prepare_benchmark_metadata_for_export()
        self._prepare__benchmark_config_for_export()
        self._prepare__benchmark_measurement_for_export()
        self._prepare_benchmark_data()

    def export(self) -> None:
        filename = self._output_dir / DEFAULT_OUTPUT_DATA_YAML
        logger.info(f"Generating {filename}")
        with open(str(filename), "w") as f:
            yaml.safe_dump(self._benchmark_data, f, indent=2, sort_keys=False)

    def _prepare_benchmark_metadata_for_export(self) -> None:
        self._benchmark_metadata.update({"schema_type":SCHEMA_TYPE})
        self._benchmark_metadata.update({"benchmark_name":BENCHMARK_NAME})

    def _prepare_benchmark_config_for_export(self) -> None:
        config_to_exclude = ['model_selection_strategy', 'batch_size', 'formatted_model_name', 'output_format','func','extra_inputs']
        for config in config_to_exclude:
            self._benchmark_config.pop(config,None)
        self._benchmark_config['model'] = ', '.join(self._benchmark_config['model'])
        self._benchmark_config['profile_export_file'] = str(self._benchmark_config['profile_export_file'])
        self._benchmark_config['artifact_dir'] = str(self._benchmark_config['artifact_dir'])
        self._benchmark_config['prompt_source'] = self._benchmark_config['prompt_source'].name.lower()
        self._add_extra_inputs_to__benchmark_config()

    def _add_extra_inputs_to__benchmark_config(self) -> None:
        self._benchmark_config.update({"extra_inputs":self._extra_inputs})

    def _prepare__benchmark_measurement_for_export(self) -> None:
        self._benchmark_measurement['output_token_throughput_per_request']['unit'] = 'queries/sec'
        self._convert_measurement_ms_to_ns()
        self._rename_metrics()

    def _convert_measuremnt_ms_to_ns(self) -> None:
        conversion_factor = 1000000
        metrics_to_convert = ['request_latency','time_to_first_token','inter_token_latency']
        for metric in metrics_to_convert:
            for key in self.__benchmark_measurement[metric]:
                if key == 'unit':
                    self._benchmark_measurement[metric][key] = 'ns'
                else:
                    self._benchmark_measurement[metric][key] = self._benchmark_measurement[metric][key] * conversion_factor

    def _rename_metrics(self) -> None:
        #rename stats to match the provided yaml
        metric_name_mapping = {
            'request_throughput':'request_throughput_per_sec',
            'request_latency': 'request_latency_ns',
            'time_to_first_token': 'time_to_first_token_ns',
            'inter_token_latency': 'inter_token_latency_ns',
            'output_token_throughput': 'output_token_throughput_per_sec',
            'output_sequence_length': 'num_output_tokens',
            'input_sequence_length' : 'num_input_tokens'
        }
        self._benchmark_measurement = {metric_name_mapping.get(original_name, original_name): refactored_name for original_name, refactored_name in self._benchmark_measurement.items()}

    def _prepare_benchmark_data(self) -> None:
        self._benchmark_data = dict(self._benchmark_metadata)
        self._benchmark_data.update({"benchmark_config":self._benchmark_config})
        self._benchmark_data.update({"benchmark_measurements": self._benchmark_measurement})

            
