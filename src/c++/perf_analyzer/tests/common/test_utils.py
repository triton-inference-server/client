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

import os
from typing import Tuple, Union

from model_analyzer.cli.cli import CLI
from model_analyzer.config.input.config_command_profile import ConfigCommandProfile
from model_analyzer.config.input.config_command_report import ConfigCommandReport
from model_analyzer.config.input.config_defaults import (
    DEFAULT_BATCH_SIZES,
    DEFAULT_CLIENT_PROTOCOL,
    DEFAULT_MEASUREMENT_MODE,
    DEFAULT_MONITORING_INTERVAL,
    DEFAULT_OUTPUT_MODEL_REPOSITORY,
    DEFAULT_TRITON_GRPC_ENDPOINT,
    DEFAULT_TRITON_HTTP_ENDPOINT,
    DEFAULT_TRITON_INSTALL_PATH,
    DEFAULT_TRITON_LAUNCH_MODE,
    DEFAULT_TRITON_METRICS_URL,
)
from model_analyzer.config.run.model_run_config import ModelRunConfig
from model_analyzer.config.run.run_config import RunConfig
from model_analyzer.constants import SECONDS_TO_MILLISECONDS_MULTIPLIER
from model_analyzer.perf_analyzer.perf_config import PerfAnalyzerConfig
from model_analyzer.record.metrics_manager import MetricsManager
from model_analyzer.result.constraint_manager import ConstraintManager
from model_analyzer.result.result_manager import ResultManager
from model_analyzer.result.run_config_measurement import RunConfigMeasurement
from model_analyzer.result.run_config_result import RunConfigResult
from model_analyzer.state.analyzer_state_manager import AnalyzerStateManager
from model_analyzer.triton.model.model_config import ModelConfig
from model_analyzer.triton.model.model_config_variant import ModelConfigVariant
from tests.mocks.mock_config import MockConfig

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def evaluate_mock_config(
    args: list, yaml_str: str, subcommand: str = "profile"
) -> Union[ConfigCommandProfile, ConfigCommandReport]:
    """
    Return a ConfigCommandReport/Analyze/Profile created from the fake CLI
    'args' list and fake config.yaml contents 'yaml_str'
    """
    yaml_content = convert_to_bytes(yaml_str)
    mock_config = MockConfig(args, yaml_content)
    mock_config.start()

    if subcommand == "report":
        config = ConfigCommandReport()
    else:
        config = ConfigCommandProfile()

    cli = CLI()
    cli.add_subcommand(cmd=subcommand, help="", config=config)
    cli.parse()
    mock_config.stop()
    return config


def load_single_model_result_manager() -> Tuple[ResultManager, ConfigCommandProfile]:
    """
    Return a ResultManager with the single model test checkpoint loaded, as well
    as the ConfigCommandProfile used to fake the profile step
    """
    dir_path = f"{ROOT_DIR}/single-model-ckpt/"
    yaml_str = "profile_models: add_sub"
    return _load_result_manager_helper(dir_path, yaml_str)


def load_multi_model_result_manager() -> Tuple[ResultManager, ConfigCommandProfile]:
    """
    Return a ResultManager with the multi model test checkpoint loaded, as well
    as the ConfigCommandProfile used to fake the profile step
    """
    dir_path = f"{ROOT_DIR}/multi-model-ckpt/"
    yaml_str = "profile_models: resnet50_libtorch,vgg19_libtorch"
    return _load_result_manager_helper(dir_path, yaml_str)


def load_ensemble_result_manager() -> Tuple[ResultManager, ConfigCommandProfile]:
    """
    Return a ResultManager with the ensemble model test checkpoint loaded, as well
    as the ConfigCommandProfile used to fake the profile step
    """
    dir_path = f"{ROOT_DIR}/ensemble-ckpt/"
    yaml_str = "profile_models: ensemble_add_sub"
    return _load_result_manager_helper(dir_path, yaml_str)


def load_bls_result_manager() -> Tuple[ResultManager, ConfigCommandProfile]:
    """
    Return a ResultManager with the BLS model test checkpoint loaded, as well
    as the ConfigCommandProfile used to fake the profile step
    """
    dir_path = f"{ROOT_DIR}/bls-ckpt/"
    yaml_str = "profile_models: bls"
    return _load_result_manager_helper(dir_path, yaml_str)


def load_request_rate_result_manager() -> Tuple[ResultManager, ConfigCommandProfile]:
    """
    Return a ResultManager with the request rate model test checkpoint loaded, as well
    as the ConfigCommandProfile used to fake the profile step
    """
    dir_path = f"{ROOT_DIR}/request-rate-ckpt/"
    yaml_str = f"""
        request_rate_search_enable: true
        profile_models: add_sub
    """
    return _load_result_manager_helper(dir_path, yaml_str)


def _load_result_manager_helper(dir_path: str, yaml_str: str):
    args = [
        "model-analyzer",
        "profile",
        "-f",
        "config.yml",
        "--checkpoint-directory",
        dir_path,
        "--export-path",
        dir_path,
        "--model-repository",
        ".",
        "--run-config-profile-models-concurrently-enable",
        "--run-config-search-mode",
        "quick",
    ]
    config = evaluate_mock_config(args, yaml_str, subcommand="profile")
    state_manager = AnalyzerStateManager(config=config, server=None)
    state_manager.load_checkpoint(checkpoint_required=True)

    result_manager = ResultManager(
        config=config,
        state_manager=state_manager,
        constraint_manager=ConstraintManager(config=config),
    )
    return result_manager, config


def convert_to_bytes(string):
    """
    Converts string into bytes and ensures minimum length requirement
    for compatibility with unpack function called in usr/lib/python3.8/gettext.py

    Parameters
    ----------
    string: str
    """
    if len(string) > 4:
        return bytes(string, "utf-8")
    else:
        return bytes(string + "    ", "utf-8")


def convert_non_gpu_metrics_to_data(non_gpu_metric_values):
    """
    Non GPU data will be a dict whose keys and values are
    a list of Records

    Parameters
    ----------
    non_gpu_metric_values: dict of non-gpu metrics
    """

    non_gpu_data = []
    non_gpu_metric_tags = list(non_gpu_metric_values.keys())

    for i, metric in enumerate(MetricsManager.get_metric_types(non_gpu_metric_tags)):
        non_gpu_data.append(metric(value=non_gpu_metric_values[non_gpu_metric_tags[i]]))

    return non_gpu_data


def convert_gpu_metrics_to_data(gpu_metric_values):
    """
    GPU data will be a dict whose keys are gpu_ids and values
    are lists of Records

    Parameters
    ----------
    gpu_metric_values: dict of gpu metrics
    """
    gpu_data = {}
    for gpu_uuid, metrics_values in gpu_metric_values.items():
        gpu_data[gpu_uuid] = []
        gpu_metric_tags = list(metrics_values.keys())
        for i, gpu_metric in enumerate(
            MetricsManager.get_metric_types(gpu_metric_tags)
        ):
            gpu_data[gpu_uuid].append(
                gpu_metric(value=metrics_values[gpu_metric_tags[i]])
            )

    return gpu_data


def convert_avg_gpu_metrics_to_data(avg_gpu_metric_values):
    """
    Avg GPU data will be a dict of Records

    Parameters
    ----------
    gpu_metric_values: dict of gpu metrics
    """
    avg_gpu_data = {}
    avg_gpu_metric_tags = list(avg_gpu_metric_values.keys())

    for i, avg_gpu_metric in enumerate(
        MetricsManager.get_metric_types(avg_gpu_metric_tags)
    ):
        avg_gpu_data[avg_gpu_metric_tags[i]] = avg_gpu_metric(
            value=avg_gpu_metric_values[avg_gpu_metric_tags[i]]
        )

    return avg_gpu_data


def construct_perf_analyzer_config(
    model_name="my-model",
    output_file_name="my-model-results.csv",
    batch_size=DEFAULT_BATCH_SIZES,
    concurrency=1,
    request_rate=None,
    launch_mode=DEFAULT_TRITON_LAUNCH_MODE,
    client_protocol=DEFAULT_CLIENT_PROTOCOL,
    perf_analyzer_flags=None,
):
    """
    Constructs a Perf Analyzer Config

    Parameters
    ----------
    model_name: str
        The name of the model
    output_file_name: str
        The name of the output file
    batch_size: int
        The batch size for this PA configuration
    concurrency: int
        The concurrency value for this PA configuration
    request_rate: int
        The request rate value for this PA configuration
    launch_mode: str
        The launch mode for this PA configuration
    client_protocol: str
        The client protocol for this PA configuration
    perf_analyzer_flags: dict
        A dict of any additional PA flags to be set

    Returns
    -------
    PerfAnalyzerConfig
        constructed with all of the above data.
    """

    pa_config = PerfAnalyzerConfig()
    pa_config._options["-m"] = model_name
    pa_config._options["-f"] = output_file_name
    pa_config._options["-b"] = batch_size

    if request_rate:
        pa_config._args["request-rate-range"] = request_rate
    else:
        pa_config._args["concurrency-range"] = concurrency

    pa_config._args["measurement-mode"] = DEFAULT_MEASUREMENT_MODE

    pa_config.update_config(perf_analyzer_flags)

    if launch_mode == "c_api":
        pa_config._args["service-kind"] = "triton_c_api"
        pa_config._args["triton-server-directory"] = DEFAULT_TRITON_INSTALL_PATH
        pa_config._args["model-repository"] = DEFAULT_OUTPUT_MODEL_REPOSITORY
    else:
        pa_config._args["collect-metrics"] = "True"
        pa_config._args["metrics-url"] = DEFAULT_TRITON_METRICS_URL
        pa_config._args["metrics-interval"] = (
            SECONDS_TO_MILLISECONDS_MULTIPLIER * DEFAULT_MONITORING_INTERVAL
        )
        pa_config._options["-i"] = client_protocol
        if client_protocol == "http":
            pa_config._options["-u"] = DEFAULT_TRITON_HTTP_ENDPOINT
        else:
            pa_config._options["-u"] = DEFAULT_TRITON_GRPC_ENDPOINT

    return pa_config


def construct_run_config(
    model_name: str, model_config_variant_name: str, pa_config_name: str
) -> RunConfig:
    """
    Constructs a Perf Analyzer Config
    """

    model_config_dict = {"name": model_name}
    model_config = ModelConfig.create_from_dictionary(model_config_dict)
    model_config_variant = ModelConfigVariant(model_config, model_config_variant_name)

    perf_config = PerfAnalyzerConfig()
    perf_config.update_config({"model-name": pa_config_name})
    mrc = ModelRunConfig(model_name, model_config_variant, perf_config)
    rc = RunConfig({})
    rc.add_model_run_config(mrc)
    return rc


def construct_run_config_measurement(
    model_name,
    model_config_names,
    model_specific_pa_params,
    gpu_metric_values,
    non_gpu_metric_values,
    constraint_manager=None,
    metric_objectives=None,
    model_config_weights=None,
):
    """
    Construct a RunConfig measurement from the given data

    Parameters
    ----------
    model_name: str
        The name of the model that generated this result
    model_config_names: list of str
        A list of Model Config names that generated this result
    model_specific_pa_params: list of dict
        A list (one per model config) of dict's of PA parameters that change
        between models in a multi-model run
    gpu_metric_values: dict
        Keys are gpu id, values are dict
        The dict where keys are gpu based metric tags, values are the data
    non_gpu_metric_values: list of dict
        List of (one per model config) dict's where keys are non gpu perf metrics, values are the data
    metric_objectives: list of RecordTypes
        A list of metric objectives (one per model config) used to compare measurements
    model_config_weights: list of ints
        A list of weights (one per model config) used to bias measurement results between models

    Returns
    -------
    RunConfigMeasurement
        constructed with all of the above data.
    """

    gpu_data = convert_gpu_metrics_to_data(gpu_metric_values)

    model_variants_name = "".join(model_config_names)
    rc_measurement = RunConfigMeasurement(model_variants_name, gpu_data)

    non_gpu_data = [
        convert_non_gpu_metrics_to_data(non_gpu_metric_value)
        for non_gpu_metric_value in non_gpu_metric_values
    ]

    for index, model_config_name in enumerate(model_config_names):
        rc_measurement.add_model_config_measurement(
            model_config_name=model_config_name,
            model_specific_pa_params=model_specific_pa_params[index],
            non_gpu_data=non_gpu_data[index],
        )

    if model_config_weights:
        rc_measurement.set_model_config_weighting(model_config_weights)

    if metric_objectives:
        rc_measurement.set_metric_weightings(metric_objectives)

    if constraint_manager:
        rc_measurement.set_constraint_manager(constraint_manager=constraint_manager)

    return rc_measurement


def construct_run_config_result(
    avg_gpu_metric_values,
    avg_non_gpu_metric_values_list,
    comparator,
    constraint_manager=None,
    value_step=1,
    model_name="test_model",
    model_config_names=["test_model"],
    run_config=None,
):
    """
    Takes a dictionary whose values are average
    metric values, constructs artificial data
    around these averages, and then constructs
    a result from this data.

    Parameters
    ----------
    avg_gpu_metric_values: dict
        The dict where keys are gpu based metric tags
        and values are the average values around which
        we want data
    avg_non_gpu_metric_values: list of dict
        Per model list of:
            keys are non gpu perf metrics, values are their
            average values.
    value_step: int
        The step value between two adjacent data values.
        Can be used to control the max/min of the data
        distribution in the construction result
    comparator: RunConfigResultComparator
        The comparator used to compare measurements/results
    model_name: str
        The name of the model that generated this result
    model_config: ModelConfig
        The model config used to generate this result
    """

    num_vals = 10

    # Construct a result
    run_config_result = RunConfigResult(
        model_name=model_name,
        run_config=run_config,
        comparator=comparator,
        constraint_manager=constraint_manager,
    )

    # Get dict of list of metric values
    gpu_metric_values = {}
    for gpu_uuid, metric_values in avg_gpu_metric_values.items():
        gpu_metric_values[gpu_uuid] = {
            key: list(
                range(
                    val - value_step * num_vals, val + value_step * num_vals, value_step
                )
            )
            for key, val in metric_values.items()
        }

    non_gpu_metric_values_list = []
    for avg_non_gpu_metric_values in avg_non_gpu_metric_values_list:
        non_gpu_metric_values_list.append(
            {
                key: list(
                    range(
                        val - value_step * num_vals,
                        val + value_step * num_vals,
                        value_step,
                    )
                )
                for key, val in avg_non_gpu_metric_values.items()
            }
        )

    # Construct measurements and add them to the result
    for i in range(2 * num_vals):
        gpu_metrics = {}
        for gpu_uuid, metric_values in gpu_metric_values.items():
            gpu_metrics[gpu_uuid] = {
                key: metric_values[key][i] for key in metric_values
            }

        non_gpu_metrics = []
        for non_gpu_metric_values in non_gpu_metric_values_list:
            non_gpu_metrics.append(
                {key: non_gpu_metric_values[key][i] for key in non_gpu_metric_values}
            )

        run_config_result.add_run_config_measurement(
            construct_run_config_measurement(
                model_name=model_name,
                model_config_names=model_config_names,
                model_specific_pa_params=[
                    {"batch_size": 1, "concurrency": 1}
                    for model_config_name in model_config_names
                ],
                gpu_metric_values=gpu_metrics,
                non_gpu_metric_values=non_gpu_metrics,
                metric_objectives=comparator._metric_weights,
            )
        )

    return run_config_result


def construct_constraint_manager(yaml_config_str=None):
    """
    Returns a ConstraintManager object for Test cases

    Parameters
    ----------
    yaml_config_str: optional str
        yaml config string with at least one profile model name and optional constraints
    """
    args = ["model-analyzer", "profile", "-f", "config.yml", "-m", "."]

    if not yaml_config_str:
        yaml_config_str = """
            profile_models:
              test_model
        """

    config = evaluate_mock_config(args, yaml_config_str, subcommand="profile")

    return ConstraintManager(config)


def default_encode(obj):
    if isinstance(obj, bytes):
        return obj.decode("utf-8")
    elif hasattr(obj, "to_dict"):
        return obj.to_dict()
    else:
        return obj.__dict__
