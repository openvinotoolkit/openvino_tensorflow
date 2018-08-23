#!  /bin/sh

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

# First parameter *must* be the IMAGE_ID
#
# Additional parameters are passed to docker-build command
#
# Within .intel.com, appropriate proxies are applied
#
# Script environment variable parameters:
#
# NG_TF_PY_VERSION   Optional: Set python major version ("2" or "3", default=2)

set -e  # Fail on any command with non-zero exit

IMAGE_ID="${1}"
if [ -z "${IMAGE_ID}" ] ; then
    echo 'Please provide an image version as the only argument'
    exit 1
fi

# Set defaults

if [ -z "${NG_TF_PY_VERSION}" ] ; then
    NG_TF_PY_VERSION='2'  # Default is Python 2
fi

# If there are more parameters, which are intended to be directly passed to
# the "docker build ..." command-line, then shift off the IMAGE_NAME
if [ "x${2}" = 'x' ] ; then
    shift
fi

case "${NG_TF_PY_VERSION}" in
    2)
        DOCKER_FILE='Dockerfile.ngraph-tf-ci-py2'
        IMAGE_NAME='ngraph_tf_ci_py2'
        ;;
    3)
        DOCKER_FILE='Dockerfile.ngraph-tf-ci-py3'
        IMAGE_NAME='ngraph_tf_ci_py3'
        ;;
    *)
        echo 'NG_TF_PY_VERSION must be set to "2", "3", or left unset (default is "2")'
        exit 1
        ;;
esac

set -u  # No unset variables after this point

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

# Context is the maint-jenkins directory, to avoid including all of
# ngraph-tensorflow-1.3 in the context.
#
# The $@ allows us to pass command-line options easily to docker build.
# Note that a "shift" is done above to remove the IMAGE_ID from the cmd line.
#
docker build  --rm=true \
       ${DOCKER_HTTP_PROXY} ${DOCKER_HTTPS_PROXY} \
       $@ \
       -f="${DOCKER_FILE}"  -t="${IMAGE_NAME}:${IMAGE_ID}"   .
