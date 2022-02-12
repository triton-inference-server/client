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


# Install jdk and maven
export TRITON_HOME="/opt/tritonserver"
cd ${TRITON_HOME}
apt update && apt install -y openjdk-11-jdk
wget https://archive.apache.org/dist/maven/maven-3/3.8.4/binaries/apache-maven-3.8.4-bin.tar.gz
tar zxvf apache-maven-3.8.4-bin.tar.gz
export PATH=${TRITON_HOME}/apache-maven-3.8.4/bin:$PATH
echo "javacpp branch ${JAVACPP_BRANCH} and tag ${JAVACPP_BRANCH_TAG}"
# Clone JavaCPP-presets, build java bindings and copy jar to /opt/tritonserver 
export JAVACPP_BRANCH=${JAVACPP_BRANCH:="https://github.com/bytedeco/javacpp-presets.git"}
export JAVACPP_BRANCH_TAG=${JAVACPP_BRANCH_TAG:="master"}
git clone --single-branch --depth=1 -b ${JAVACPP_BRANCH_TAG} ${JAVACPP_BRANCH}
cd javacpp-presets
mvn clean install --projects .,tritonserver > install.txt
mvn clean install -f platform --projects ../tritonserver/platform -Djavacpp.platform=linux-x86_64 > platform_install.txt
cd tritonserver/build
mvn clean install -Djavacpp.platform=linux-x86_64
cp target/tritonserver-uber-*.jar ${TRITON_HOME}/tritonserver-java-bindings.jar
