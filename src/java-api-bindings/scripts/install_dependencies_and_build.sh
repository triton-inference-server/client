#!/bin/bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
USAGE="
usage: install_dependencies_and_build.sh [options]

Installs Maven, Java JDK and builds Tritonserver Java bindings
-h|--help                         Shows usage
-t|--triton-home                  Expected Trition library location, default is: /opt/tritonserver
-b|--build-home                   Expected build location, default is: /tmp/build
-v|--maven-version                Maven version, default is: "3.8.4"
-c|--core-tag                     Tag for core repo, defaut is: "main"
-j|--jar-install-path             Path to install the bindings .jar
--javacpp-branch                  Javacpp-presets git path, default is https://github.com/bytedeco/javacpp-presets.git
--javacpp-tag                     Javacpp-presets branch tag, default "master"
--enable-developer-tools-server  Include C++ bindings from developer_tools repository
"

# Get all options:
OPTS=$(getopt -l ht:b:v:c:j:,help,triton-home,build-home:,maven-version:,core-tag:,jar-install-path:,javacpp-branch:,javacpp-tag:,enable-developer-tools-server -- "$@")

TRITON_HOME="/opt/tritonserver"
BUILD_HOME="/tmp/build"
MAVEN_VERSION="3.8.4"
export MAVEN_PATH=${BUILD_HOME}/apache-maven-${MAVEN_VERSION}/bin/mvn
CORE_BRANCH_TAG="main"
TOOLS_BRANCH=${TOOLS_BRANCH:="https://github.com/triton-inference-server/developer_tools.git"}
TOOLS_BRANCH_TAG=${TOOLS_BRANCH_TAG:="main"}
JAVACPP_BRANCH=${JAVACPP_BRANCH:="https://github.com/bytedeco/javacpp-presets.git"}
JAVACPP_BRANCH_TAG=${JAVACPP_BRANCH_TAG:="master"}
CMAKE_VERSION=${CMAKE_VERSION:="3.21.1"}
export JAR_INSTALL_PATH="/workspace/install/java-api-bindings"

for OPTS; do
    case "$OPTS" in
        -h|--help)
        printf "%s\\n" "$USAGE"
        exit
        ;;
        -t|--triton-home)
        TRITON_HOME=$2
        echo "Triton home set to: ${TRITON_HOME}"
        shift 2
        ;;
        -b|--build-home)
        BUILD_HOME=$2
        shift 2
        echo "Build home set to: ${BUILD_HOME}"
        ;;
        -v|--maven-version)
        MAVEN_VERSION=$2
        echo "Maven version is set to: ${MAVEN_VERSION}"
        shift 2
        ;;
        -c|--core-tag) 
        CORE_BRANCH_TAG=$2
        echo "Tritonserver core branch is set to: ${CORE_BRANCH_TAG}"
        shift 2
        ;;
        -j|--jar-install-path) 
        JAR_INSTALL_PATH=$2
        echo "Bindings jar will be set to: ${JAR_INSTALL_PATH}"
        shift 2
        ;;
        --javacpp-branch) 
        JAVACPP_BRANCH=$2
        echo "Javacpp-presets branch set to: ${JAVACPP_BRANCH}"
        shift 2
        ;;
        --javacpp-tag) 
        JAVACPP_BRANCH_TAG=$2
        echo "Javacpp-presets branch tag set to: ${JAVACPP_BRANCH_TAG}"
        shift 2
        ;;
        --enable-developer-tools-server) 
        INCLUDE_DEVELOPER_TOOLS_SERVER=1
        echo "Including developer tools server C++ bindings"
        ;;
    esac
done
set -x

# Install jdk and maven
rm -r ${BUILD_HOME}
mkdir -p ${BUILD_HOME}
cd ${BUILD_HOME}
apt update && apt install -y openjdk-11-jdk
wget https://archive.apache.org/dist/maven/maven-3/${MAVEN_VERSION}/binaries/apache-maven-${MAVEN_VERSION}-bin.tar.gz
tar zxvf apache-maven-${MAVEN_VERSION}-bin.tar.gz
export PATH=$PATH:$PWD/apache-maven-${MAVEN_VERSION}/bin/

# Build static libraries and copy wrapper includes to triton home
# Copy necessary tritonserver .h files so the bindings can be generated
mkdir -p ${TRITON_HOME}/lib/
cd ${BUILD_HOME}
CORE_BRANCH=${CORE_BRANCH:="https://github.com/triton-inference-server/core.git"}
git clone --single-branch --depth=1 -b ${CORE_BRANCH_TAG} ${CORE_BRANCH}
cp -r core/include ${TRITON_HOME}/include

if [ ${INCLUDE_DEVELOPER_TOOLS_SERVER} -eq 1 ]; then
    # install cmake and rapidjson
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | \
        gpg --dearmor - |  \
        tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null && \
        apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal main' && \
        apt-get update && \
        apt-get install -y --no-install-recommends \
            cmake-data=${CMAKE_VERSION}-0kitware1ubuntu20.04.1 \
            cmake=${CMAKE_VERSION}-0kitware1ubuntu20.04.1 \
            rapidjson-dev

    git clone --single-branch --depth=1 -b ${TOOLS_BRANCH_TAG} ${TOOLS_BRANCH} 
    cd developer_tools/server
    rm -r build 
    mkdir build && cd build
    cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install -DTRITON_BUILD_TEST=ON -DTRITON_ENABLE_EXAMPLES=ON -DTRITON_BUILD_STATIC_LIBRARY=OFF .. 
    make -j"$(grep -c ^processor /proc/cpuinfo)" install
    # Copy dynamic library to triton home
    cp ${BUILD_HOME}/developer_tools/server/build/install/lib/libtritondevelopertoolsserver.so ${TRITON_HOME}/lib/.

    BUILD_INCLUDE_REPO=${BUILD_HOME}/developer_tools/server/include/triton/developer_tools
    BUILD_SRC_REPO=${BUILD_HOME}/developer_tools/server/src
    TRITON_INCLUDE_REPO=${TRITON_HOME}/include/triton/developer_tools
    rm -r ${TRITON_INCLUDE_REPO}
    mkdir -p ${TRITON_INCLUDE_REPO}/src
    cp ${BUILD_INCLUDE_REPO}/common.h ${TRITON_INCLUDE_REPO}/.
    cp ${BUILD_INCLUDE_REPO}/generic_server_wrapper.h ${TRITON_INCLUDE_REPO}/.
    cp ${BUILD_SRC_REPO}/infer_requested_output.h ${TRITON_INCLUDE_REPO}/src/.
    cp ${BUILD_SRC_REPO}/tracer.h ${TRITON_INCLUDE_REPO}/src/.
fi
# Clone JavaCPP-presets, build java bindings and copy jar to /opt/tritonserver 
cd ${BUILD_HOME}
git clone --single-branch --depth=1 -b ${JAVACPP_BRANCH_TAG} ${JAVACPP_BRANCH}
cd javacpp-presets
# remove usage of C++ bindings
if [ ${INCLUDE_DEVELOPER_TOOLS_SERVER} -ne 1 ]; then
    sed -i 's/, "tritondevelopertoolsserver"//' tritonserver/src/main/java/org/bytedeco/tritonserver/presets/tritonserver.java
    sed -i 's/, "common.h", "generic_server_wrapper.h"//' tritonserver/src/main/java/org/bytedeco/tritonserver/presets/tritonserver.java
    sed -i 's/, "/opt/tritonserver/include/triton/developer_tools", "/opt/tritonserver/include/triton/developer_tools/src"' tritonserver/src/main/java/org/bytedeco/tritonserver/presets/tritonserver.java
fi
${MAVEN_PATH} clean install --projects .,tritonserver
${MAVEN_PATH} clean install -f platform --projects ../tritonserver/platform -Djavacpp.platform=linux-x86_64

# Copy over the jar to a specific location
rm -r ${JAR_INSTALL_PATH}
mkdir -p ${JAR_INSTALL_PATH}
cp ${BUILD_HOME}/javacpp-presets/tritonserver/platform/target/tritonserver-platform-*shaded.jar ${JAR_INSTALL_PATH}/tritonserver-java-bindings.jar

rm -r ${BUILD_HOME}/javacpp-presets/ ${BUILD_HOME}/core ${BUILD_HOME}/developer_tools 
rm -r /root/.m2/repository

set +x
