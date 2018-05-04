#!  /bin/bash

# Script parameters:
#
# $1 ImageID    Required: ID of the ngtf_bridge_ci docker image to use
# $2 NGTFdir    Required: ngraph-tensorflow directory to build
# $3 PythonVer  Optional: version of Python to build with (default: 2)

set -e  # Fail on any command with non-zero exit

ngtf_dir="${2}"
if [ ! -d "${ngtf_dir}" ] ; then
    echo 'Please provide the name of the ngraph-tensorflow directory you want to build, as the 2nd parameter'
    exit 1
fi

if [ -z "${3}" ] ; then
    export PYTHON_VERSION_NUMBER='2'  # Build for Python 2 by default
else
    export PYTHON_VERSION_NUMBER="${3}"
fi

# Note that the docker image must have been previously built using the
# make-docker-ngtf-bridge-ci.sh script (in the same directory as this script).
#
IMAGE_CLASS='ngtf_bridge_ci'
IMAGE_ID="${1}"
if [ -z "${IMAGE_ID}" ] ; then
    echo 'Please provide an image version as the only argument'
    exit 1
fi

set -u  # No unset variables after this point

# Find the top-level bridge directory, so we can mount it into the docker
# container
bridge_dir="$(realpath ../../..)"

bridge_mountpoint='/home/dockuser/bridge'
ngtf_mountpoint='/home/dockuser/ngtf'

RUNASUSER_SCRIPT="${bridge_mountpoint}/test/ci/docker/docker-scripts/run-as-user.sh"
BUILD_SCRIPT="${bridge_mountpoint}/test/ci/docker/docker-scripts/run-ngtf-mkldnn-build.sh"

docker run --rm \
       --env RUN_UID="$(id -u)" \
       --env RUN_CMD="${BUILD_SCRIPT}" \
       --env http_proxy=http://proxy-us.intel.com:911 \
       --env https_proxy=https://proxy-us.intel.com:911 \
       --env PYTHON_VERSION_NUMBER="${PYTHON_VERSION_NUMBER}" \
       -v "${bridge_dir}:${bridge_mountpoint}" \
       -v "${ngtf_dir}:${ngtf_mountpoint}" \
       "${IMAGE_CLASS}:${IMAGE_ID}" "${RUNASUSER_SCRIPT}"


