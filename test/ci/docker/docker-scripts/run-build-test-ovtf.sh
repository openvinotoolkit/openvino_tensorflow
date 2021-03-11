#!  /bin/bash

# ==============================================================================
# Copyright (C) 2021 Intel Corporation
 
# SPDX-License-Identifier: Apache-2.0
# ==============================================================================

# This script is designed to be called from within a docker container.
# It is installed into a docker image.  It will not run outside the container.
#
# Optional parameters, passed in environment variables:
#
# OV_TF_BUILD_OPTIONS  Command-line options for build_ovtf.py
# OV_TF_TEST_OPTIONS   Command-line options for test_ovtf.py (CPU-backend)
# OV_TF_TEST_PLAIDML   If defined and non-zero, also run PlaidML-CPU unit-tests
#
# General environment variables that are passed through to the docker container:
#
# OPENVINO_TF_BACKEND
# PLAIDML_EXPERIMENTAL
# PLAIDML_DEVICE_IDS

set -e  # Make sure we exit on any command that returns non-zero
set -o pipefail # Make sure cmds in pipe that are non-zero also fail immediately

if [ -z "${OV_TF_BUILD_OPTIONS}" ] ; then
    export OV_TF_BUILD_OPTIONS=''
fi
if [ -z "${OV_TF_TEST_OPTIONS}" ] ; then
    export OV_TF_TEST_OPTIONS=''
fi
if [ -z "${OV_TF_TEST_PLAIDML}" ] ; then
    export OV_TF_TEST_PLAIDML=''
fi

# Report on any recognized passthrough variables
if [ ! -z "${OPENVINO_TF_BACKEND}" ] ; then
    echo "OPENVINO_TF_BACKEND=${OPENVINO_TF_BACKEND}"
fi
if [ ! -z "${PLAIDML_EXPERIMENTAL}" ] ;  then
    echo "PLAIDML_EXPERIMENTAL=${PLAIDML_EXPERIMENTAL}"
fi
if [ ! -z "${PLAIDML_DEVICES_IDS}" ] ; then
    echo "PLAIDML_DEVICE_IDS=${PLAIDML_DEVICE_IDS}"
fi

set -u  # No unset variables from this point

# Set up some important known directories
bridge_dir='/home/dockuser/ngraph-tf'
bbuild_dir="${bridge_dir}/build"
tf_dir="${bbuild_dir}/tensorflow"
ci_dir="${bridge_dir}/test/ci/docker"
venv_dir="/tmp/venv_code_check"
ngraph_wheel_dir="${bbuild_dir}"

# HOME is expected to be /home/dockuser.  See script run-as-user.sh, which
# sets this up.

echo "In $(basename ${0}):"
echo ''
echo "  bridge_dir=${bridge_dir}"
echo "  bbuild_dir=${bbuild_dir}"
echo "  tf_dir=${tf_dir}"
echo "  ci_dir=${ci_dir}"
echo "  venv_dir=${venv_dir}"
echo "  ngraph_wheel_dir=${ngraph_wheel_dir}"
echo ''
echo "  HOME=${HOME}"
echo ''
echo "  OV_TF_BUILD_OPTIONS=${OV_TF_BUILD_OPTIONS}"
echo "  OV_TF_TEST_OPTIONS=${OV_TF_TEST_OPTIONS}"
echo "  OV_TF_TEST_PLAIDML=${OV_TF_TEST_PLAIDML}"

# Do some up-front checks, to make sure necessary directories are in-place and
# build directories are not-in-place

if [ -d "${bbuild_dir}" ] ; then
    ( >&2 echo '***** Error: *****' )
    ( >&2 echo "Bridge build directory already exists -- please remove it before calling this script: ${bbuild_dir}" )
    exit 1
fi

if [ -d "${venv_dir}" ] ; then
    ( >&2 echo '***** Error: *****' )
    ( >&2 echo "Virtual-env build directory already exists -- please remove it before calling this script: ${venv_dir}" )
    exit 1
fi

# Make sure the Bazel cache is in /tmp, as docker images have too little space
# in the root filesystem, where /home (and $HOME/.cache) is.  Even though we
# may not be using the Bazel cache in the builds (in docker), we do this anyway
# in case we decide to turn the Bazel cache back on.
echo "Adjusting bazel cache to be located in /tmp/bazel-cache"
rm -fr "$HOME/.cache"
mkdir /tmp/bazel-cache
ln -s /tmp/bazel-cache "$HOME/.cache"

xtime="$(date)"
echo  ' '
echo  "===== Checking gcc, python and OS version at ${xtime} ====="
echo  ' '

echo 'gcc is being run from:'
which gcc

echo ' '
echo 'gcc verison is:'
gcc --version

echo 'g++ is being run from:'
which g++

echo ' '
echo 'g++ version is:'
g++ --version

if [ -f /usr/bin/python ] ; then
    echo ' '
    echo 'python version is:'
    python --version
else
    echo '/usr/bin/python was not found'
fi

if [ -f /usr/bin/python2 ] ; then
    echo ' '
    echo 'python2 version is:'
    python2 --version
else
    echo '/usr/bin/python2 was not found'
fi

if [ -f /usr/bin/python3 ] ; then
    echo ' '
    echo 'python3 version is:'
    python3 --version
else
    echo '/usr/bin/python3 was not found'
fi

echo ' '
echo 'virtualenv version is:'
virtualenv --version

echo ' '
echo 'pip version is:'
pip --version

echo ' '
echo 'Ubuntu version is:'
cat /etc/os-release

xtime="$(date)"
echo  ' '
echo  "===== Run build_ovtf.py at ${xtime} ====="
echo  ' '

cd "${bridge_dir}"
echo "Running: ./build_ovtf.py ${OV_TF_BUILD_OPTIONS}"
./build_ovtf.py ${OV_TF_BUILD_OPTIONS}
exit_code=$?
echo "Exit status for build_ovtf.py is ${exit_code}"

xtime="$(date)"
echo  ' '
echo  "===== Run test_ovtf.py at ${xtime} ====="
echo  ' '

cd "${bridge_dir}"
echo "Running: ./test_ovtf.py ${OV_TF_TEST_OPTIONS}"
./test_ovtf.py ${OV_TF_TEST_OPTIONS}
exit_code=$?
echo "Exit status for test_ovtf.py is ${exit_code}"

if [ ! -z "${OV_TF_TEST_PLAIDML}" ] ; then
    xtime="$(date)"
    echo  ' '
    echo  "===== Run test_ovtf.py with PlaidML at ${xtime} ====="
    echo  ' '

    OPENVINO_TF_BACKEND=PLAIDML

    echo "OPENVINO_TF_BACKEND=${OPENVINO_TF_BACKEND}"

    cd "${bridge_dir}"
    echo "Running: ./test_ovtf.py --plaidml_unit_tests_enable"
    ./test_ovtf.py --plaidml_unit_tests_enable
    exit_code=$?
    echo "Exit status for test_ovtf.py --plaidml_unit_tests_enable is ${exit_code}"
fi

xtime="$(date)"
echo ' '
echo "===== Completed Tensorflow Build and Test at ${xtime} ====="
echo ' '
