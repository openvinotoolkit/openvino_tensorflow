#!  /bin/bash

# Script parameters:
#
# $1 ImageID    Required: ID of the ngtf_bridge_ci docker image to use
# $2 NGTFdir    Required: ngraph-tensorflow directory to test
# $3 PythonVer  Optional: version of Python to build with (default: 2)

set -e  # Fail on any command with non-zero exit

ngtf_dir="${2}"
if [ ! -d "${ngtf_dir}" ] ; then
    echo 'Please provide the name of the ngraph-tensorflow directory you want to build as the 2nd argument'
    exit 1
fi

if [ -z "${3}" ] ; then
    export PYTHON_VERSION_NUMBER='2'  # Build for Python 2 by default
else
    export PYTHON_VERSION_NUMBER="${3}"
fi

# The docker image ID is currently just the git SHA of this cloned repo.
# We need this ID to know which docker image to run with.
# Note that the docker image must have been previously built using the
# make-docker-tf-ngraph-base.sh script (in the same directory as this script).
#
IMAGE_NAME='ngtf_bridge_ci'
IMAGE_ID="${1}"
if [ -z "${IMAGE_ID}" ] ; then
    echo 'Please provide an image version as the only argument'
    exit 1
fi

set -u  # No unset variables from this point on

# Find the top-level bridge directory, so we can mount it into the docker
# container
bridge_dir="$(realpath ../../..)"

bridge_mountpoint='/home/dockuser/bridge'
ngtf_mountpoint='/home/dockuser/ngtf'

# If the plugin has not been specified, assume the plugin built by
# docker-run-ngtf-bridge-build.sh
if [ -z "${USER_PLUGIN_PATH:-}" ] ; then
    export USER_PLUGIN_PATH="${bridge_mountpoint}/BUILD-BRIDGE/src/libngraph_plugin.so"
fi

RUNASUSER_SCRIPT="${bridge_mountpoint}/test/ci/docker/docker-scripts/run-as-user.sh"
TESTING_SCRIPT="${bridge_mountpoint}/test/ci/docker/docker-scripts/run-ngtf-bridge-ci-testing.sh"

# docker run --rm \
docker run \
       --env RUN_UID="$(id -u)" \
       --env RUN_CMD="${TESTING_SCRIPT}" \
       --env http_proxy=http://proxy-us.intel.com:911 \
       --env https_proxy=https://proxy-us.intel.com:911 \
       --env PYTHON_VERSION_NUMBER="${PYTHON_VERSION_NUMBER}" \
       --env USER_PLUGIN_PATH="${USER_PLUGIN_PATH}" \
       -v "${bridge_dir}:${bridge_mountpoint}" \
       -v "${ngtf_dir}:${ngtf_mountpoint}" \
       "${IMAGE_NAME}:${IMAGE_ID}" "${RUNASUSER_SCRIPT}"
