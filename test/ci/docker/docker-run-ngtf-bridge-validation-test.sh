#!  /bin/bash

# Script parameters:
#
# $1 ImageID    Required: ID of the ngtf_bridge_ci docker image to use
# $2 PythonVer  Optional: version of Python to build with (default: 2)

set -e  # Fail on any command with non-zero exit

if [ -z "${2}" ] ; then
    export PYTHON_VERSION_NUMBER='2'  # Build for Python 2 by default
else
    export PYTHON_VERSION_NUMBER="${3}"
fi

script='run-tf-ngraph-validation-test.sh'

# Note that the docker image must have been previously built using the
# make-docker-tf-ngraph-base.sh script (in the same directory as this script).
#
IMAGE_CLASS='ngtf_bridge_ci'
IMAGE_ID="${1}"
if [ -z "${IMAGE_ID}" ] ; then
    echo 'Please provide an image version as the only argument'
    exit 1
fi

dataset_dir='/dataset'

docker_dataset='/home/dockuser/dataset'

# Find the top-level bridge directory, so we can mount it into the docker
# container
bridge_dir="$(realpath ../../..)"

bridge_mountpoint='/home/dockuser/bridge'

RUNASUSER_SCRIPT="${bridge_mountpoint}/test/ci/docker/docker-scripts/run-as-user.sh"
BUILD_SCRIPT="${bridge_mountpoint}/test/ci/docker/docker-scripts/run-ngtf-bridge-validation-test.sh"

# NOTE: a pass-through mechanism is provided for the following environment
#       variables:
#
#           TF_NG_MODEL_DATASET  Model and dataset to run
#           TF_NG_ITERATIONS     Number of iterations (aka steps) to run (mlp)
#           TF_NG_EPOCHS         Number of epochs to run (resnet)
#           TF_NG_COMPARE_TO     JSON file of reference results to compare to
#           TF_NG_DO_NOT_RUN     If defined and not empty, does not run pytest

# TEST_NG_MODEL_DATASET *must* be defined to run any validation test in docker
if [ -z "${TF_NG_MODEL_DATASET}" ] ; then
    ( >&2 echo "TF_NG_MODEL_DATASET must be set to the model-dataset to run" )
    exit
fi

docker run --rm \
       --env RUN_UID="$(id -u)" \
       --env RUN_CMD="${BUILD_SCRIPT}" \
       --env HOST_HOSTNAME="${HOSTNAME}" \
       --env TF_NG_MODEL_DATASET="${TF_NG_MODEL_DATASET}" \
       --env TF_NG_ITERATIONS="${TF_NG_ITERATIONS}" \
       --env TF_NG_EPOCHS="${TF_NG_EPOCHS}" \
       --env TF_NG_COMPARE_TO="${TF_NG_COMPARE_TO}" \
       --env TF_NG_DO_NOT_RUN="${TF_NG_DO_NOT_RUN}" \
       --env http_proxy=http://proxy-us.intel.com:911 \
       --env https_proxy=https://proxy-us.intel.com:911 \
       -v "${dataset_dir}:${docker_dataset}" \
       -v "${bridge_dir}:${bridge_mountpoint}" \
       "${IMAGE_CLASS}:${IMAGE_ID}" "${RUNASUSER_SCRIPT}"

