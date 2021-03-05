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
# BASE_DOCKERFILE      Dockerfile to use to build the base env container
# BASE_IMAGE_NAME      Image name for the base env container
# BASE_IMAGE_TAG       Image tag for the base env container
# OVTF_DOCKERFILE      Dockerfile to use to build/install OVTF container
# OVTF_IMAGE_NAME      Image name for the OVTF container
# OVTF_IMAGE_TAG       Image tag for the OVTF container

# Set vars for the base image
BASE_DOCKERFILE='Dockerfile.ubuntu18.04'
BASE_IMAGE_NAME='openvino_tensorflow'
BASE_IMAGE_TAG='devel'

# Set vars for the ovtf image
OVTF_DOCKERFILE='Dockerfile.ubuntu18.04.install'
OVTF_IMAGE_NAME='openvino_tensorflow'
OVTF_IMAGE_TAG='ovtf'
OVTF_BUILD_OPTIONS=$1

echo "docker_build_and_install_ovtf is building the following:"
echo "    Base Dockerfile:             ${BASE_DOCKERFILE}"
echo "    Base Image name/tag:         ${BASE_IMAGE_NAME}:${BASE_IMAGE_TAG}"
echo "    nGraph TF Dockerfile:        ${OVTF_DOCKERFILE}"
echo "    nGraph TF Image name/tag:    ${OVTF_IMAGE_NAME}:${OVTF_IMAGE_TAG}"
echo "    nGraph TF build options:     ${OVTF_BUILD_OPTIONS}"

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

# Build base docker image to get the build environment for ngraph and TF
dbuild_cmd="docker build --rm=true \
${DOCKER_HTTP_PROXY} ${DOCKER_HTTPS_PROXY} \
-f=${BASE_DOCKERFILE} -t=${BASE_IMAGE_NAME}:${BASE_IMAGE_TAG} ."

echo "Docker build command for base image: ${dbuild_cmd}"
$dbuild_cmd
dbuild_result=$?

if [ $dbuild_result = 0 ] ; then
    echo ' '
    echo "Successfully created base docker image ${BASE_IMAGE_NAME}:${BASE_IMAGE_TAG}"
else
    echo ' '
    echo "Base docker image build reported an error (exit code ${dbuild_result})"
    exit 1
fi

# Pass through any build options
if [ ! -z "${OVTF_BUILD_OPTIONS}" ] ; then
    DOCKER_OVTF_BUILD_OPTIONS="--build-arg ovtf_build_options='${OVTF_BUILD_OPTIONS}'"
else
    DOCKER_OVTF_BUILD_OPTIONS=' '
fi

# Use the base docker image to run the build_ovtf.py script and install ngraph and TF
dbuild_cmd="docker build --rm=true \
${DOCKER_HTTP_PROXY} ${DOCKER_HTTPS_PROXY} ${DOCKER_OVTF_BUILD_OPTIONS} \
--build-arg base_image=${BASE_IMAGE_NAME}:${BASE_IMAGE_TAG} \
-f=${OVTF_DOCKERFILE} -t=${OVTF_IMAGE_NAME}:${OVTF_IMAGE_TAG} ."

echo "Docker build command for nGraph TF image: ${dbuild_cmd}"
alias dbuild=$dbuild_cmd
dbuild
dbuild_result=$?

if [ $dbuild_result = 0 ] ; then
    echo ' '
    echo "Successfully created docker image with ovtf installed: ${OVTF_IMAGE_NAME}:${OVTF_IMAGE_TAG}"
else
    echo ' '
    echo "Docker image with ovtf build reported an error (exit code ${dbuild_result})"
    exit 1
fi
