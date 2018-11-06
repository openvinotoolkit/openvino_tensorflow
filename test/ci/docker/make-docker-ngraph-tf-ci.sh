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

# Script parameters:
#
# $1 ImageID    Required: ID of the ngtf_bridge_ci docker image to use
# $2 ImageType  Optional: Type of docker image to build
#
# Additional parameters are passed to docker-build command
#
# Within .intel.com, appropriate proxies are applied
#
# Script environment variable parameters:
#
# NG_TF_PY_VERSION   Optional: Set python major version ("2" or "3", default=2)

DIR_DOCKERFILES='dockerfiles'
PRE_DOCKERFILE='Dockerfile.ngraph_tf.'
PRE_NAME='ngraph_tf_'

syntax() {
    cmd=`basename ${0}`
    echo "Syntax: ${cmd} DTag [ ImageType [ docker-run-options] ]"
    echo "where:"
    echo "    DTag         Required: tag for the docker image (as in DRepo:DTag)"
    echo "    ImageType    Optional: which type of image to create"
    echo 'Additional parameters are passed directly to the "docker run" command'
    echo ' '
    echo "Image types available:"
    ls -1 "${DIR_DOCKERFILES}" | grep "${PRE_DOCKERFILE}" | sed -e "s/${PRE_DOCKERFILE}//"
}  # syntax()

if [ "${1}" = '-h' -o "${1}" = '--help' -o "${1}" = '' ] ; then
    syntax
    exit 0
fi

set -e  # Fail on any command with non-zero exit

IMAGE_ID="$1"
if [ -z "${IMAGE_ID}" ] ; then  # Parameter 1 is REQUIRED
    echo 'Please provide an image version as the only argument'
    exit 1
else
    shift 1  # We found parameter one, remove it from $@
fi

IMAGE_TYPE="$1"  # Second parameter has been shifted to be first parameter
if [ -z "${IMAGE_TYPE}" ] ; then  # PARAMETER 2 is OPTIONAL
    IMAGE_TYPE='default'
else
    shift 1  # We found parameter two, remove it from $@
fi

# We accomodate both the old-style dockerfile names and a new
# scalable naming, where IMAGE_TYPE matches the extension
case "${IMAGE_TYPE}" in
    default)
        DOCKER_FILE='Dockerfile.ngraph-tf-ci-py2'
        IMAGE_NAME='ngraph_tf_ci_py2'
        ;;
    default_py27)
        DOCKER_FILE='Dockerfile.ngraph-tf-ci-py2'
        IMAGE_NAME='ngraph_tf_ci_py2'
        ;;
    default_py35)
        DOCKER_FILE='Dockerfile.ngraph-tf-ci-py3'
        IMAGE_NAME='ngraph_tf_ci_py3'
        ;;
    *)  # Look for a dockerfile with an extension matching IMAGE_TYPE
        if [ -f "${DIR_DOCKERFILES}/${PRE_DOCKERFILE}${IMAGE_TYPE}" ] ; then
            DOCKER_FILE="${PRE_DOCKERFILE}${IMAGE_TYPE}"
            IMAGE_NAME="${PRE_NAME}${IMAGE_TYPE}"
        else
            echo "Dockerfile ${DIR_DOCKERFILES}/${PRE_DOCKERFILE}${IMAGE_TYPE} not found for image type ${IMAGE_TYPE}"
            exit 1
        fi
        ;;
esac


# The NG_TF_PY_VERSION takes precedence over the optional IMAGE_TYPE parameter,
# because NG_TF_PY_VERSION existed first and we need to maintain backward
# compatibility (for now)
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
        # Do nothing if NG_TF_PY_VERSION is explicitly not set
        ;;
esac

set -u  # No unset variables after this point

# Show in log what is being build
echo "make-docker-ngraph-tf-ci is building the following:"
echo "    Image Type         IMAGE_TYPE: ${IMAGE_TYPE}"
echo "    Dockerfile        DOCKER_FILE: ${DOCKER_FILE}"
echo "    Docker Repository  IMAGE_NAME: ${IMAGE_NAME}"
echo "    Docker Tag           IMAGE_ID: ${IMAGE_ID}"

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
dbuild_cmd="docker build  --rm=true \
            ${DOCKER_HTTP_PROXY} ${DOCKER_HTTPS_PROXY} \
            $@ \
            -f=${DIR_DOCKERFILES}/${DOCKER_FILE}  -t=${IMAGE_NAME}:${IMAGE_ID}  ."
echo "Docker build command: ${dbuild_cmd}"
$dbuild_cmd
dbuild_result=$?

if [ $dbuild_result = 0 ] ; then
    echo ' '
    echo "Successfully created docker image ${IMAGE_NAME}:${IMAGE_ID}"
else
    echo ' '
    echo "Docker image build reported an error (exit code ${dbuild_result})"
fi
