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


from genai_perf.metrics import Metrics


class ExporterConfig:
    def __init__(self):
        self._stats = None
        self._metrics = None
        self._args = None
        self._extra_inputs = None
        self._artifact_dir = None
        self._benchmark_duration = None

    @property
    def stats(self):
        return self._stats

    @stats.setter
    def stats(self, stats_value):
        self._stats = stats_value

    @property
    def metrics(self):
        return self._metrics

    @metrics.setter
    def metrics(self, metrics: Metrics):
        self._metrics = metrics

    @property
    def args(self):
        return self._args

    @args.setter
    def args(self, args_value):
        self._args = args_value

    @property
    def extra_inputs(self):
        return self._extra_inputs

    @extra_inputs.setter
    def extra_inputs(self, extra_inputs_value):
        self._extra_inputs = extra_inputs_value

    @property
    def artifact_dir(self):
        return self._artifact_dir

    @artifact_dir.setter
    def artifact_dir(self, artifact_dir_value):
        self._artifact_dir = artifact_dir_value

    @property
    def benchmark_duration(self):
        return self._benchmark_duration

    @benchmark_duration.setter
    def benchmark_duration(self, duration):
        if duration <= 0:
            raise ValueError("Benchmark duration cannot be non-positive")
        self._benchmark_duration = duration