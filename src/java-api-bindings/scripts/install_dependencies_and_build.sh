#!/bin/bash
# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
set -x

# Install jdk and maven
TRITON_HOME="/opt/tritonserver"
mkdir -p ${TRITON_HOME}
MAVEN_VERSION=${MAVEN_VERSION:="3.8.4"}
cd ${TRITON_HOME}
apt update && apt install -y openjdk-11-jdk
wget https://archive.apache.org/dist/maven/maven-3/${MAVEN_VERSION}/binaries/apache-maven-${MAVEN_VERSION}-bin.tar.gz
tar zxvf apache-maven-${MAVEN_VERSION}-bin.tar.gz
MAVEN_PATH=${TRITON_HOME}/apache-maven-${MAVEN_VERSION}/bin/mvn

# Copy tritonserver.h
mkdir -p /opt/tritonserver/lib/
cd ${TRITON_HOME}
CORE_BRANCH=${CORE_BRANCH:="https://github.com/triton-inference-server/core.git"}
CORE_BRANCH_TAG=${CORE_BRANCH_TAG:="main"}
git clone --single-branch --depth=1 -b ${CORE_BRANCH_TAG} ${CORE_BRANCH}
cp -r core/include /opt/tritonserver/include

# Clone JavaCPP-presets, build java bindings and copy jar to /opt/tritonserver 
JAVACPP_BRANCH=${JAVACPP_BRANCH:="https://github.com/bytedeco/javacpp-presets.git"}
JAVACPP_BRANCH_TAG=${JAVACPP_BRANCH_TAG:="master"}
git clone --single-branch --depth=1 -b ${JAVACPP_BRANCH_TAG} ${JAVACPP_BRANCH}
cd javacpp-presets
${MAVEN_PATH} clean install --projects .,tritonserver
${MAVEN_PATH} clean install -f platform --projects ../tritonserver/platform -Djavacpp.platform=linux-x86_64

mkdir -p /workspace/install/java-api-bindings
cp ${TRITON_HOME}/javacpp-presets/tritonserver/platform/target/tritonserver-platform-*shaded.jar /workspace/install/java-api-bindings/tritonserver-java-bindings.jar
rm -rf ${TRITON_HOME}

set +x
