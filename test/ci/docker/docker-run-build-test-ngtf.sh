#!  /bin/bash

# ==============================================================================
#  Copyright 2018-2020 Intel Corporation
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

# Script parameters:
#
# $1 ImageID    Required: ID of the ngtf_bridge_ci docker image to use
#
# Script environment variable parameters:
#
# NG_TF_BUILD_OPTIONS  Optional: additional build options for build_ngtf.py
# NG_TF_TEST_OPTIONS   Optional: additional test options for test_ngtf.py
# NG_TF_TEST_PLAIDML   Optional: run additional testing on PlaidML backend
#
# General environment variables that are passed through to the docker container:
#
# NGRAPH_TF_BACKEND
# PLAIDML_EXPERIMENTAL
# PLAIDML_DEVICE_IDS

set -e  # Fail on any command with non-zero exit

IMAGE_ID="${1}"
if [ -z "${IMAGE_ID}" ] ; then
    echo 'Please provide an image version as the first parameter'
    exit 1
fi

# Check if we have a docker ID of image:ID, or just ID
# Use || true to make sure the exit code is always zero, so that the script is
# not killed if ':' is not found
long_ID=`echo ${IMAGE_ID} | grep ':' || true`

# If we have just ID, then IMAGE_CLASS AND IMAGE_ID have
# already been set above
#
# Handle case where we have image:ID
if [ ! -z "${long_ID}" ] ; then
    IMAGE_CLASS=` echo ${IMAGE_ID} | sed -e 's/:[^:]*$//' `
    IMAGE_ID=` echo ${IMAGE_ID} | sed -e 's/^[^:]*://' `
fi

# Find the top-level bridge directory, so we can mount it into the docker
# container
bridge_dir="$(realpath ../../..)"

bridge_mountpoint='/home/dockuser/ngraph-tf'
tf_mountpoint='/home/dockuser/tensorflow'

# Set up a bunch of volume mounts
volume_mounts="-v ${bridge_dir}:${bridge_mountpoint}"

# Set up optional environment variables
optional_env=''
if [ ! -z "${NG_TF_BUILD_OPTIONS}" ] ; then
  optional_env="${optional_env} --env NG_TF_BUILD_OPTIONS=${NG_TF_BUILD_OPTIONS}"
fi
if [ ! -z "${NG_TF_TEST_OPTIONS}" ] ; then
  optional_env="${optional_env} --env NG_TF_TEST_OPTIONS=${NG_TF_TEST_OPTIONS}"
fi
if [ ! -z "${NG_TF_TEST_PLAIDML}" ] ; then
  optional_env="${optional_env} --env NG_TF_TEST_PLAIDML=${NG_TF_TEST_PLAIDML}"
fi

# Set up passthrough environment variables
if [ ! -z "${NGRAPH_TF_BACKEND}" ] ; then
  optional_env="${optional_env} --env NGRAPH_TF_BACKEND=${NGRAPH_TF_BACKEND}"
fi
if [ ! -z "${PLAIDML_EXPERIMENTAL}" ] ; then
  optional_env="${optional_env} --env PLAIDML_EXPERIMENTAL=${PLAIDML_EXPERIMENTAL}"
fi
if [ ! -z "${PLAIDML_DEVICE_IDS}" ] ; then
  optional_env="${optional_env} --env PLAIDML_DEVICE_IDS=${PLAIDML_DEVICE_IDS}"
fi

set -u  # No unset variables after this point

RUNASUSER_SCRIPT="${bridge_mountpoint}/test/ci/docker/docker-scripts/run-as-user.sh"
BUILD_SCRIPT="${bridge_mountpoint}/test/ci/docker/docker-scripts/run-build-test-ngtf.sh"

# If proxy settings are detected in the environment, make sure they are
# included on the docker-build command-line.  This mirrors a similar system
# in the Makefile.

if [ ! -z "${http_proxy}" ] ; then
    DOCKER_HTTP_PROXY="--env http_proxy=${http_proxy}"
else
    DOCKER_HTTP_PROXY=' '
fi

if [ ! -z "${https_proxy}" ] ; then
    DOCKER_HTTPS_PROXY="--env https_proxy=${https_proxy}"
else
    DOCKER_HTTPS_PROXY=' '
fi

drun_cmd="docker run --rm \
    --env RUN_UID=$(id -u) \
    --env RUN_CMD=${BUILD_SCRIPT} \
    ${optional_env} \
    ${DOCKER_HTTP_PROXY} ${DOCKER_HTTPS_PROXY} \
    ${volume_mounts} \
    ${IMAGE_CLASS}:${IMAGE_ID} ${RUNASUSER_SCRIPT}"

echo "Docker build command: ${drun_cmd}"
$drun_cmd
