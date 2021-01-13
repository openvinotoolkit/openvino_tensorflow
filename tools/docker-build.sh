#!  /bin/bash

# ==============================================================================
#  Copyright 2020 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ==============================================================================

# Script environment variables:
#
# NGRAPH_TF_VERSION        nGraph TF bridge version 
# DEVEL_DOCKERFILE         Dockerfile to use to build the devel container
# DEVEL_IMAGE              Image name and tag for the devel container
# NGRAPH_TF_BUILD_OPTIONS  Additional build options to use for the devel container
# NGRAPH_TF_DOCKERFILE     Dockerfile to use to build nGraph TF container
# NGRAPH_TF_IMAGE          Image name and tag for the nGraph TF container

NGRAPH_TF_VERSION=r0.23
DEVEL_DOCKERFILE='Dockerfile.ubuntu18.04.devel'
DEVEL_IMAGE='ngraph-bridge:devel-${NGRAPH_TF_VERSION}'
NGRAPH_TF_DOCKERFILE='Dockerfile.ubuntu18.04.install'
NGRAPH_TF_IMAGE='ngraph-bridge:${NGRAPH_TF_VERSION}'
NGRAPH_TF_BUILD_OPTIONS=$1

echo "docker-build.sh is building the following:"
echo "    Devel Dockerfile:            ${DEVEL_DOCKERFILE}"
echo "    Devel Image name/tag:        ${DEVEL_IMAGE}"
echo "    nGraph TF build options:     ${NGRAPH_TF_BUILD_OPTIONS}"
echo "    nGraph TF Dockerfile:        ${NGRAPH_TF_DOCKERFILE}"
echo "    nGraph TF Image name/tag:    ${NGRAPH_TF_IMAGE}"

# If proxy settings are detected in the environment, make sure they are
# included on the docker-build command-line.  This mirrors a similar system
# in the Makefile.

if [ ! -z "${http_proxy}" ] ; then
    DOCKER_HTTP_PROXY="--build-arg http_proxy=${http_proxy}"
else
    DOCKER_HTTP_PROXY=' '
fi

if [ ! -z "${https_proxy}" ] ; then
    DOCKER_HTTPS_PROXY="--build-arg https_proxy=${https_proxy}"
else
    DOCKER_HTTPS_PROXY=' '
fi

# Pass through any build options
if [ -z "${NGRAPH_TF_BUILD_OPTIONS}" ] ; then
    NGRAPH_TF_BUILD_OPTIONS="--use_prebuilt_tensorflow --disable_cpp_api"
fi
NGRAPH_TF_BUILD_OPTIONS="--build-arg build_options='${NGRAPH_TF_BUILD_OPTIONS}'"

# Build base docker image to get the build environment for ngraph and TF
dbuild_cmd="docker build --rm=true \
 ${DOCKER_HTTP_PROXY} ${DOCKER_HTTPS_PROXY} ${NGRAPH_TF_BUILD_OPTIONS} \
 -f=${DEVEL_DOCKERFILE} -t=${DEVEL_IMAGE} ."

echo "Docker build command for devel image: ${dbuild_cmd}"
eval $dbuild_cmd
dbuild_result=$?

if [ $dbuild_result = 0 ] ; then
    echo ' '
    echo "Successfully created devel docker image ${DEVEL_IMAGE}"
else
    echo ' '
    echo "Base docker image build reported an error (exit code ${dbuild_result})"
    exit 1
fi

# Use the base docker image to run the build_ngtf.py script and install ngraph and TF
dbuild_cmd="docker build --rm=true \
 ${DOCKER_HTTP_PROXY} ${DOCKER_HTTPS_PROXY} \
 --build-arg base_image=${DEVEL_IMAGE} \
 -f=${NGRAPH_TF_DOCKERFILE} -t=${NGRAPH_TF_IMAGE} ."

echo "Docker build command for nGraph TF image: ${dbuild_cmd}"
$dbuild_cmd
dbuild_result=$?

if [ $dbuild_result = 0 ] ; then
    echo ' '
    echo "Successfully created docker image with nGraph TF installed: ${NGRAPH_TF_IMAGE}"
else
    echo ' '
    echo "Docker image with ngtf build reported an error (exit code ${dbuild_result})"
    exit 1
fi
