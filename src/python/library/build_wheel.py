#!/usr/bin/env python3
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

import argparse
import os
import pathlib
import re
import shutil
import subprocess
from distutils.dir_util import copy_tree
from tempfile import mkstemp

def fail_if(p, msg):
    if p:
        print('error: {}'.format(msg), file=sys.stderr)
        sys.exit(1)

def mkdir(path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

def touch(path):
    pathlib.Path(path).touch()

def cpdir(src, dest):
    copy_tree(src, dest, preserve_symlinks=1)

def sed(pattern, replace, source, dest=None):
    fin = open(source, 'r')
    if dest:
        fout = open(dest, 'w')
    else:
        fd, name = mkstemp()
        fout = open(name, 'w')

    for line in fin:
        out = re.sub(pattern, replace, line)
        fout.write(out)

    fin.close()
    fout.close()
    if not dest:
        shutil.copyfile(name, source)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dest-dir',
        type=str,
        required=True,
        help=
        'Destination directory.'
    )
    parser.add_argument('--linux',
                        action="store_true",
                        required=False,
                        help='Include linux specific artifacts.')
    parser.add_argument(
        '--perf-analyzer',
        type=str,
        required=False,
        default=None,
        help='perf-analyzer path.')

    FLAGS = parser.parse_args()

    FLAGS.triton_version = None
    with open('TRITON_VERSION', "r") as vfile:
        FLAGS.triton_version = vfile.readline().strip()

    FLAGS.whl_dir = os.path.join(FLAGS.dest_dir, 'wheel')

    print("=== Building in: {}".format(os.getcwd()))
    print("=== Using builddir: {}".format(FLAGS.whl_dir))
    print("Adding package files")

    mkdir(os.path.join(FLAGS.whl_dir, 'tritonclient'))
    touch(os.path.join(FLAGS.whl_dir, 'tritonclient/__init__.py'))

    # Needed for backwards-compatibility; remove when moving
    # completely to the new structure.
    if os.path.isdir('tritonclientutils'):
        cpdir('tritonclientutils', os.path.join(FLAGS.whl_dir, 'tritonclientutils'))
    if os.path.isdir('tritonhttpclient'):
        cpdir('tritonhttpclient', os.path.join(FLAGS.whl_dir, 'tritonhttpclient'))
    if os.path.isdir('tritongrpcclient'):
        cpdir('tritongrpcclient', os.path.join(FLAGS.whl_dir, 'tritongrpcclient'))
    if FLAGS.linux:
        if os.path.isdir('tritonshmutils'):
            cpdir('tritonshmutils', os.path.join(FLAGS.whl_dir, 'tritonshmutils'))

    if os.path.isdir('tritonclient/grpc'):
        cpdir('tritonclient/grpc', os.path.join(FLAGS.whl_dir, 'tritonclient/grpc'))
        shutil.copyfile("../_deps/repo-common-build/protobuf/model_config_pb2.py",
                 os.path.join(FLAGS.whl_dir, 'tritonclient/grpc/model_config_pb2.py'))
        shutil.copyfile("../_deps/repo-common-build/protobuf/grpc_service_pb2.py",
                 os.path.join(FLAGS.whl_dir, 'tritonclient/grpc/service_pb2.py'))
        shutil.copyfile("../_deps/repo-common-build/protobuf/grpc_service_pb2_grpc.py",
                 os.path.join(FLAGS.whl_dir, 'tritonclient/grpc/service_pb2_grpc.py'))

        # Use 'sed' command to fix protoc compiled imports (see
        # https://github.com/google/protobuf/issues/1491).
        for fl in ('model_config_pb2.py', 'service_pb2.py'):
            sed("^import ([^ ]*)_pb2 as ([^ ]*)$",
                "from tritonclient.grpc import \\1_pb2 as \\2",
                os.path.join(FLAGS.whl_dir, 'tritonclient', 'grpc', fl))

        sed("^import grpc_([^ ]*)_pb2 as ([^ ]*)$",
            "from tritonclient.grpc import \\1_pb2 as \\2",
            os.path.join(FLAGS.whl_dir, 'tritonclient/grpc/service_pb2_grpc.py'))

    if os.path.isdir('tritonclient/http'):
        cpdir('tritonclient/http', os.path.join(FLAGS.whl_dir, 'tritonclient/http'))

    mkdir(os.path.join(FLAGS.whl_dir, 'tritonclient/utils'))
    shutil.copyfile("tritonclient/utils/__init__.py",
                    os.path.join(FLAGS.whl_dir, 'tritonclient/utils/__init__.py'))

    if FLAGS.linux:
        cpdir('tritonclient/utils/shared_memory',
              os.path.join(FLAGS.whl_dir, 'tritonclient/utils/shared_memory'))
        shutil.copyfile('tritonclient/utils/libcshm.so',
                        os.path.join(FLAGS.whl_dir,
                                     'tritonclient/utils/shared_memory/libcshm.so'))
        if (os.path.exists('tritonclient/utils/libccudashm.so') and
            os.path.exists('tritonclient/utils/cuda_shared_memory/__init__.py')):
            cpdir('tritonclient/utils/cuda_shared_memory',
                  os.path.join(FLAGS.whl_dir, 'tritonclient/utils/cuda_shared_memory'))
            shutil.copyfile('tritonclient/utils/libccudashm.so',
                            os.path.join(FLAGS.whl_dir,
                                         'tritonclient/utils/cuda_shared_memory/libccudashm.so'))

        # Copy the pre-compiled perf_analyzer binary
        if FLAGS.perf_analyzer is not None:
            # The permission bits need to be copied to along with the executable
            shutil.copy(FLAGS.perf_analyzer, os.path.join(FLAGS.whl_dir, 'perf_analyzer'))

            # Create a symbolic link for backwards compatibility
            if not os.path.exists(os.path.join(FLAGS.whl_dir, 'perf_client')):
                os.symlink('perf_analyzer', os.path.join(FLAGS.whl_dir, 'perf_client'))

    shutil.copyfile('LICENSE.txt', os.path.join(FLAGS.whl_dir, 'LICENSE.txt'))
    shutil.copyfile('setup.py', os.path.join(FLAGS.whl_dir, 'setup.py'))
    cpdir('requirements', os.path.join(FLAGS.whl_dir, 'requirements'))

    os.chdir(FLAGS.whl_dir)
    print("=== Building wheel")
    if FLAGS.linux:
        if os.uname().machine == "aarch64":
            platform_name = "manylinux2014_aarch64"
        else:
            platform_name = "manylinux1_x86_64"
        args = ['python3', 'setup.py', 'bdist_wheel', '--plat-name', platform_name]
    else:
        args = ['python3', 'setup.py', 'bdist_wheel']

    wenv = os.environ.copy()
    wenv["VERSION"] = FLAGS.triton_version
    p = subprocess.Popen(args, env=wenv)
    p.wait()
    fail_if(p.returncode != 0, 'setup.py failed')

    cpdir('dist', FLAGS.dest_dir)

    print("=== Output wheel file is in: {}".format(FLAGS.dest_dir))
    touch(os.path.join(FLAGS.dest_dir, 'stamp.whl'))
