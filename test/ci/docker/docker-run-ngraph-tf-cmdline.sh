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
# NG_TF_CONFIG       Optional: If defined, the value will be sourced at the
#                              beginning of the script.  This allows other
#                              environment variables (below) to be easily
#                              encapsulated in a config script.
#
# NG_TF_RUN_TYPE     Optional: "ngraph" or "reference", default=ngraph
# NG_TF_MODELS_REPO  Optional: Directory that models repo is clone into
# NG_TF_TRAINED      Optional: Directory that pretrained models are in
# NG_TF_DATASET      Optional: Dataset to prepare for run
# NG_TF_LOG_ID       Optional: String to included in name of log
# NG_TF_PY_VERSION   Optional: Set python major version ("2" or "3", default=2)
# NG_TF_WHEEL_NGRAPH Optional: Name of ngraph wheel to install
# NG_TF_WHEEL_TF     Optional: Name of TensorFlow wheel (for ngraph) to install
# NG_TF_WHEEL_TF_REF Optional: Name of reference TF wheel to install (no ngraph)


set -e  # Fail on any command with non-zero exit

if [ ! -z "${NG_TF_CONFIG}" ] ; then
    if [ ! -f "${NG_TF_CONFIG}" ] ; then
        ( >&2 echo "Could not find config file ${NG_TF_CONFIG}" )
    fi
    echo "Loading config from file ${NG_TF_CONFIG}:"
    cat "${NG_TF_CONFIG}"
    source "${NG_TF_CONFIG}"
fi

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

# Set defaults

if [ -z "${NG_TF_RUN_TYPE}" ] ; then
    NG_TF_RUN_TYPE='ngraph'  # Default is to run with ngraph
fi

if [ -z "${NG_TF_PY_VERSION}" ] ; then
    NG_TF_PY_VERSION='2'  # Default is Python 2
fi

# Note that the docker image must have been previously built using the
# make-docker-ngraph-tf-ci.sh script (in the same directory as this script).
#
case "${NG_TF_PY_VERSION}" in
    2)
        IMAGE_CLASS='ngraph_tf_ci_py2'
        ;;
    3)
        IMAGE_CLASS='ngraph_tf_ci_py3'
        ;;
    *)
        echo 'NG_TF_PY_VERSION must be set to "2", "3", or left unset (default is "2")'
        exit 1
        ;;
esac
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
if [ ! -z "${NG_TF_RUN_TYPE}" ] ; then
  optional_env="${optional_env} --env NG_TF_RUN_TYPE=${NG_TF_RUN_TYPE}"
fi
if [ ! -z "${NG_TF_DATASET}" ] ; then
  optional_env="${optional_env} --env NG_TF_DATASET=${NG_TF_DATASET}"
fi
if [ ! -z "${NG_TF_LOG_ID}" ] ; then
  optional_env="${optional_env} --env NG_TF_LOG_ID=${NG_TF_LOG_ID}"
fi
if [ ! -z "${NG_TF_PY_VERSION}" ] ; then
  optional_env="${optional_env} --env NG_TF_PY_VERSION=${NG_TF_PY_VERSION}"
fi
if [ ! -z "${NG_TF_WHEEL_NGRAPH}" ] ; then
  optional_env="${optional_env} --env NG_TF_WHEEL_NGRAPH=${NG_TF_WHEEL_NGRAPH}"
fi
if [ ! -z "${NG_TF_WHEEL_TF}" ] ; then
  optional_env="${optional_env} --env NG_TF_WHEEL_TF=${NG_TF_WHEEL_TF}"
fi
if [ ! -z "${NG_TF_WHEEL_TF_REF}" ] ; then
  optional_env="${optional_env} --env NG_TF_WHEEL_TF_REF=${NG_TF_WHEEL_TF_REF}"
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
