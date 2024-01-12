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
-c|--core-tag                     Tag for core repo, default is: "main"
-j|--jar-install-path             Path to install the bindings .jar
--javacpp-branch                  Javacpp-presets git path, default is https://github.com/bytedeco/javacpp-presets.git
--javacpp-tag                     Javacpp-presets branch tag, default "master"
--enable-developer-tools-server   Include C++ bindings from developer_tools repository
--keep-build-dependencies         Keep build dependencies instead of deleting
"

# Get all options:
OPTS=$(getopt -l ht:b:v:c:j:,help,triton-home,build-home:,maven-version:,core-tag:,jar-install-path:,javacpp-branch:,javacpp-tag:,enable-developer-tools-server,keep-build-dependencies -- "$@")

TRITON_HOME="/opt/tritonserver"
BUILD_HOME="/tmp/build"
MAVEN_VERSION="3.8.4"
export MAVEN_PATH=${BUILD_HOME}/apache-maven-${MAVEN_VERSION}/bin/mvn
TRITON_CORE_REPO_TAG=${TRITON_CORE_REPO_TAG:="main"}
JAVACPP_BRANCH=${JAVACPP_BRANCH:="https://github.com/bytedeco/javacpp-presets.git"}
JAVACPP_BRANCH_TAG=${JAVACPP_BRANCH_TAG:="master"}
CMAKE_VERSION=${CMAKE_VERSION:="3.21.1"}
export JAR_INSTALL_PATH="/workspace/install/java-api-bindings"
# Note: True != 0 and False == 0
export INCLUDE_DEVELOPER_TOOLS_SERVER=0
KEEP_BUILD_DEPENDENCIES=1

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
        export MAVEN_PATH=${BUILD_HOME}/apache-maven-${MAVEN_VERSION}/bin/mvn
        shift 2
        echo "Build home set to: ${BUILD_HOME}"
        ;;
        -v|--maven-version)
        MAVEN_VERSION=$2
        export MAVEN_PATH=${BUILD_HOME}/apache-maven-${MAVEN_VERSION}/bin/mvn
        echo "Maven version is set to: ${MAVEN_VERSION}"
        shift 2
        ;;
        -c|--core-tag)
        TRITON_CORE_REPO_TAG=$2
        echo "Tritonserver core branch is set to: ${TRITON_CORE_REPO_TAG}"
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
        export INCLUDE_DEVELOPER_TOOLS_SERVER=1
        echo "Including developer tools server C++ bindings"
        ;;
        --keep-build-dependencies)
        KEEP_BUILD_DEPENDENCIES=0
        echo "Including developer tools server C++ bindings"
        ;;
    esac
done
set -x

if [ ${INCLUDE_DEVELOPER_TOOLS_SERVER} -ne 0 ]; then
    # install cmake and rapidjson
    apt-get update && apt-get install -y gpg wget && \
        wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | \
            gpg --dearmor - |  \
            tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null && \
        . /etc/os-release && \
        echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ $UBUNTU_CODENAME main" | \
        tee /etc/apt/sources.list.d/kitware.list >/dev/null && \
        apt-get update && \
        apt-get install -y --no-install-recommends cmake=3.27.7* cmake-data=3.27.7* rapidjson-dev
fi

# Install jdk and maven
mkdir -p ${BUILD_HOME}
cd ${BUILD_HOME}
apt update && apt install -y openjdk-11-jdk
wget https://archive.apache.org/dist/maven/maven-3/${MAVEN_VERSION}/binaries/apache-maven-${MAVEN_VERSION}-bin.tar.gz
tar zxvf apache-maven-${MAVEN_VERSION}-bin.tar.gz
export PATH=$PATH:$PWD/apache-maven-${MAVEN_VERSION}/bin/

# Clone JavaCPP-presets, build java bindings and copy jar to /opt/tritonserver
cd ${BUILD_HOME}
git clone --single-branch --depth=1 -b ${JAVACPP_BRANCH_TAG} ${JAVACPP_BRANCH}
cd javacpp-presets

# Remove developer_tools/server related build
if [ ${INCLUDE_DEVELOPER_TOOLS_SERVER} -eq 0 ]; then
    rm -r tritonserver/src/gen
    rm tritonserver/src/main/java/org/bytedeco/tritonserver/presets/tritondevelopertoolsserver.java
fi

mvn clean install --projects .,tritonserver
mvn clean install -f platform --projects ../tritonserver/platform -Djavacpp.platform=linux-x86_64

# Copy over the jar to a specific location
mkdir -p ${JAR_INSTALL_PATH}
cp ${BUILD_HOME}/javacpp-presets/tritonserver/platform/target/tritonserver-platform-*shaded.jar ${JAR_INSTALL_PATH}/tritonserver-java-bindings.jar

if [ ${KEEP_BUILD_DEPENDENCIES} -eq 1 ]; then
    rm -r ${BUILD_HOME}/javacpp-presets/
    rm -r /root/.m2/repository
fi

set +x
