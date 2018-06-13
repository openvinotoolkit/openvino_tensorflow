#!  /bin/sh

# First parameter *must* be the IMAGE_ID
#
# Additional parameters are passed to docker-build command
#
# Within .intel.com, appropriate proxies are applied

set -e  # Fail on any command with non-zero exit

DOCKER_FILE='Dockerfile.ngraph-tf-ci'

IMAGE_NAME='ngraph_tf_ci'
IMAGE_ID="${1}"
if [ -z "${IMAGE_ID}" ] ; then
    echo 'Please provide an image version as the only argument'
    exit 1
fi

# If there are more parameters, which are intended to be directly passed to
# the "docker build ..." command-line, then shift off the IMAGE_NAME
if [ "x${2}" = 'x' ] ; then
    shift
fi

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
