#!  /bin/bash

# ==============================================================================
# Copyright (C) 2021 Intel Corporation
 
# SPDX-License-Identifier: Apache-2.0
# ==============================================================================

# Script parameters:
#
# $1 ImageID    Required: ID of the ngtf_bridge_ci docker image to use

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

bridge_mountpoint='/home/dockuser/openvino-tf'
tf_mountpoint='/home/dockuser/tensorflow'

# Set up a bunch of volume mounts
volume_mounts="-v ${bridge_dir}:${bridge_mountpoint}"

set -u  # No unset variables after this point

RUNASUSER_SCRIPT="${bridge_mountpoint}/test/ci/docker/docker-scripts/run-as-user.sh"
BUILD_SCRIPT="${bridge_mountpoint}/test/ci/docker/docker-scripts/run-check-code-format.sh"

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
    ${DOCKER_HTTP_PROXY} ${DOCKER_HTTPS_PROXY} \
    ${volume_mounts} \
    ${IMAGE_CLASS}:${IMAGE_ID} ${RUNASUSER_SCRIPT}"

echo "Docker build command: ${drun_cmd}"
$drun_cmd
