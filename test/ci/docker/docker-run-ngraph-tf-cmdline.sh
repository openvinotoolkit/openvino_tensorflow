#!  /bin/bash

# ==============================================================================
#  Copyright 2018 Intel Corporation
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

# Script command line parameters:
#
# $1 ImageID    Required: ID of the ngtf_bridge_ci docker image to use
# $2 Command    Required: Command to use to run the model
#                         (single-quoted, so the whole command is one string)
#
# Script environment variable parameters:
#
# NG_TF_MODELS_REPO  Optional: Directory that models repo is clone into
# NG_TF_TRAINED      Optional: Directory that pretrained models are in
# NG_TF_DATASET      Optional: Dataset to prepare for run
# NG_TF_LOG_ID       Optional: String to included in name of log


set -e  # Fail on any command with non-zero exit

IMAGE_ID="${1}"
if [ -z "${IMAGE_ID}" ] ; then  # Required ImageID command-line parameter
    ( >&2 echo 'Please provide an image version as the first argument' )
    exit 1
fi

CMDLINE="${2}"
if [ -z "${CMDLINE}" ] ; then  # Required Command command-line parameter
    ( >&2 echo "Second parameter must be a single-quoted command to run in the docker container")
    exit 1
fi

if [ -z "${NG_TF_MODELS_REPO}" ] ; then
    NG_TF_MODELS_REPO=''  # Make sure this is set, for use below
fi

if [ -z "${NG_TF_TRAINED}" ] ; then
    NG_TF_TRAINED=''  # Make sure this is set, for use below
fi

if [ -z "${NG_TF_DATASET}" ] ; then
    NG_TF_DATASET=''  # Make sure this is set, for use below
fi

if [ -z "${NG_TF_LOG_ID}" ] ; then
    NG_TF_LOG_ID=''  # Make sure this is set, for use below
fi

export PYTHON_VERSION_NUMBER='2'  # Build for Python 2 by default

# Note that the docker image must have been previously built using the
# make-docker-ngraph-tf-ci.sh script (in the same directory as this script).
#
IMAGE_CLASS='ngraph_tf_ci'
# IMAGE_ID set from 1st parameter, above

# Set up optional volume mounts
volume_mounts='-v /dataset:/dataset'
if [ ! -z "${NG_TF_MODELS_REPO}" ] ; then
  volume_mounts="${volume_mounts} -v ${NG_TF_MODELS_REPO}:/home/dockuser/ngraph-models"
fi
if [ -z "${NG_TF_TRAINED}" ] ; then
  volume_mounts="${volume_mounts} -v /aipg_trained_dataset:/aipg_trained_dataset"
  volume_mounts="${volume_mounts} -v /aipg_trained_dataset:/trained_dataset"
else
  volume_mounts="${volume_mounts} -v ${NG_TF_TRAINED}:/aipg_trained_dataset"
  volume_mounts="${volume_mounts} -v ${NG_TF_TRAINED}:/trained_dataset"
fi

# Set up optional environment variables
optional_env=''
if [ ! -z "${NG_TF_DATASET}" ] ; then
  optional_env="${optional_env} --env NG_TF_DATASET=${NG_TF_DATASET}"
fi
if [ ! -z "${NG_TF_LOG_ID}" ] ; then
  optional_env="${optional_env} --env NG_TF_LOG_ID=${NG_TF_LOG_ID}"
fi


# Find the top-level bridge directory, so we can mount it into the docker
# container
bridge_dir="$(realpath ../../..)"
bridge_mountpoint='/home/dockuser/bridge'
volume_mounts="-v ${bridge_dir}:${bridge_mountpoint} ${volume_mounts}"

RUNASUSER_SCRIPT="${bridge_mountpoint}/test/ci/docker/docker-scripts/run-as-user.sh"
BUILD_SCRIPT="${bridge_mountpoint}/test/ci/docker/docker-scripts/run-ngraph-tf-cmdline.sh"

# If proxy settings are detected in the environment, make sure they are
# included on the docker-build command-line.  This mirrors a similar system
# in the Makefile.

if [ ! -z "${http_proxy}" ] ; then
    docker_http_proxy="--env http_proxy=${http_proxy}"
else
    docker_http_proxy=' '
fi

if [ ! -z "${https_proxy}" ] ; then
    docker_https_proxy="--env https_proxy=${https_proxy}"
else
    docker_https_proxy=' '
fi

set -x  # Show the docker command being run
docker run --rm \
       --env RUN_UID="$(id -u)" \
       --env RUN_CMD="${BUILD_SCRIPT}" \
       --env HOST_HOSTNAME="${HOSTNAME}" \
       --env CMDLINE="${CMDLINE}" \
       ${optional_env} \
       ${docker_http_proxy} ${docker_https_proxy} \
       ${volume_mounts} \
       "${IMAGE_CLASS}:${IMAGE_ID}" "${RUNASUSER_SCRIPT}"
