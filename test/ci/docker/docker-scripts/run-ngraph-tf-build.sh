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

# This script is designed to be called from within a docker container.
# It is installed into a docker image.  It will not run outside the container.

set -e  # Make sure we exit on any command that returns non-zero
set -u  # No unset variables
set -o pipefail # Make sure cmds in pipe that are non-zero also fail immediately


# Default is Python 2, but can override with NG_TF_PY_VERSION env. variable
export PYTHON_VERSION_NUMBER="${NG_TF_PY_VERSION}"
if [ -z "${PYTHON_VERSION_NUMBER}" ] ; then
    PYTHON_VERSION_NUMBER=2
fi
export PYTHON_BIN_PATH="/usr/bin/python$PYTHON_VERSION_NUMBER"

# Set up some important known directories
bridge_dir='/home/dockuser/ngraph-tf'
bbuild_dir="${bridge_dir}/BUILD-BRIDGE"
tf_dir='/home/dockuser/tensorflow'
ci_dir="${bridge_dir}/test/ci/docker"
dataset_dir='/dataset'
trained_dir='/trained_dataset'
venv_dir="/tmp/venv_python${PYTHON_VERSION_NUMBER}"
ngraph_dist_dir="${bbuild_dir}/ngraph/ngraph_dist"
ngraph_wheel_dir="${bbuild_dir}/python/dist"
libngraph_so="${bbuild_dir}/src/libngraph_bridge.so"
libngraph_dist_dir="${bridge_dir}/libngraph_dist"  # Directory to save plugin artifacts in
libngraph_tarball="${bridge_dir}/libngraph_dist.tgz"  # Tarball artifact to send to Artifactory
imagenet_dataset="${dataset_dir}/Imagenet_Validation"
trained_resnet50_model="${trained_dir}/ngraph_tensorflow/fully_trained/resnet50"

# HOME is expected to be /home/dockuser.  See script run-as-user.sh, which
# sets this up.

# Set up a directory to build the wheel in, so we know where to grab
# the wheel from later.  If the directory already exists, remove it.
export WHEEL_BUILD_DIR="${tf_dir}/BUILD_WHEEL"

echo "In $(basename ${0}):"
echo ''
echo "  bridge_dir=${bridge_dir}"
echo "  bbuild_dir=${bbuild_dir}"
echo "  tf_dir=${tf_dir}"
echo "  ci_dir=${ci_dir}"
echo "  dataset_dir=${dataset_dir}"
echo "  trained_dir=${trained_dir}"
echo "  venv_dir=${venv_dir}"
echo "  ngraph_dist_dir=${ngraph_dist_dir}"
echo "  ngraph_wheel_dir=${ngraph_wheel_dir}"
echo "  libngraph_so=${libngraph_so}"
echo "  libngraph_dist_dir=${libngraph_dist_dir}"
echo "  libngraph_tarball=${libngraph_tarball}"
echo "  imagenet_dataset=${imagenet_dataset}"
echo "  trained_resnet50_model=${trained_resnet50_model}"
echo ''
echo "  HOME=${HOME}"
echo "  PYTHON_VERSION_NUMBER=${PYTHON_VERSION_NUMBER}"
echo "  PYTHON_BIN_PATH=${PYTHON_BIN_PATH}"
echo "  WHEEL_BUILD_DIR=${WHEEL_BUILD_DIR}  (Used by maint/build-install-tf.sh)"

# Do some up-front checks, to make sure necessary directories are in-place and
# build directories are not-in-place

if [ -d "${WHEEL_BUILD_DIR}" ] ; then
    ( >&2 echo '***** Error: *****' )
    ( >&2 echo "Wheel build directory already exists -- please remove it before calling this script: ${WHEEL_BUILD_DIR}" )
    exit 1
fi

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

if [ -d "${libngraph_dist_dir}" ] ; then
    ( >&2 echo '***** Error: *****' )
    ( >&2 echo "libngraph_dist directory already exists -- please remove it before calling this script: ${libngraph_dist_dir}" )
    exit 1
fi

if [ -f "${libngraph_tarball}" ] ; then
    ( >&2 echo '***** Error: *****' )
    ( >&2 echo "libngraph distribution directory already exists -- please remove it before calling this script: ${libngraph_tarball}" )
    exit 1
fi

if [ ! -d "${dataset_dir}" ] ; then
    ( >&2 echo '***** Error: *****' )
    ( >&2 echo "Datset directory ${dataset_dir} does not seem to be mounted inside the Docker container" )
    exit 1
fi

if [ ! -d "${imagenet_dataset}" ] ; then
    ( >&2 echo '***** Error: *****' )
    ( >&2 echo "The validation dataset for ImageNet does not seem to be found: ${imagenet_dataset}" )
    exit 1
fi

if [ ! -d "${trained_resnet50_model}" ] ; then
    ( >&2 echo '***** Error: *****' )
    ( >&2 echo "The pretrained model for resnet50 CI testing does not seem to be found: ${trained_resnet50_model}" )
    exit 1
fi

xtime="$(date)"
echo  ' '
echo  "===== Checking gcc and OS version at ${xtime} ====="
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

echo ' '
echo 'Ubuntu version is:'
cat /etc/os-release

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
echo  "===== Setting Up Virtual Environment for Tensorflow Wheel at ${xtime} ====="
echo  ' '

# Make sure the bash shell prompt variables are set, as virtualenv crashes
# if PS2 is not set.
PS1='prompt> '
PS2='prompt-more> '
virtualenv --system-site-packages -p "${PYTHON_BIN_PATH}" "${venv_dir}"
source "${venv_dir}/bin/activate"

xtime="$(date)"
echo  ' '
echo  "===== Configuring Tensorflow Build at ${xtime} ====="
echo  ' '

export CC_OPT_FLAGS="-march=native"
export USE_DEFAULT_PYTHON_LIB_PATH=1
export TF_ENABLE_XLA=0

export TF_NEED_MKL=0
export TF_DOWNLOAD_MKL=0

export TF_NEED_JEMALLOC=1
export TF_NEED_GCP=0
export TF_NEED_HDFS=0
export TF_NEED_VERBS=0
export TF_NEED_OPENCL=0
export TF_NEED_CUDA=0
export TF_NEED_MPI=0

cd "${tf_dir}"
./configure

xtime="$(date)"
echo  ' '
echo  "===== Starting Tensorflow Binaries Build at ${xtime} ====="
echo  ' '

cd "${tf_dir}"

cmd="bazel build --config=opt --verbose_failures //tensorflow/tools/pip_package:build_pip_package"
echo "Running bazel command: ${cmd}"
${cmd}

xtime="$(date)"
echo  ' '
echo  "===== Starting Tensorflow Wheel Build at ${xtime} ====="
echo  ' '

bazel-bin/tensorflow/tools/pip_package/build_pip_package "${WHEEL_BUILD_DIR}"

xtime="$(date)"
echo  ' '
echo  "===== Installing Tensorflow Wheel at ${xtime} ====="
echo  ' '

set -x
cd "${tf_dir}"
declare WHEEL_FILE="$(find "${WHEEL_BUILD_DIR}" -name '*.whl')"
# If installing into the OS, use:
# sudo --preserve-env --set-home pip install --ignore-installed ${PIP_INSTALL_EXTRA_ARGS:-} "${WHEEL_FILE}"
# Here we are installing into a virtual environment, so DO NOT USE SUDO!!!
pip install -U "${WHEEL_FILE}"
set +x

xtime="$(date)"
echo  ' '
echo  "===== Starting Tensorflow C++ Library Build at ${xtime} ====="
echo  ' '

cd "${tf_dir}"
cmd="bazel build --config=opt //tensorflow:libtensorflow_cc.so"
echo "Running bazel command: ${cmd}"
${cmd}

xtime="$(date)"
echo  ' '
echo  "===== Starting nGraph TensorFlow Bridge Build at ${xtime} ====="
echo  ' '

cd "${bridge_dir}"

mkdir "${bbuild_dir}"
cd "${bbuild_dir}"
cmd="cmake -DUNIT_TEST_ENABLE=TRUE -DTF_SRC_DIR=${tf_dir} .."
echo "Running cmake command: ${cmd}"
${cmd}

cmd="make -j16"
echo "Running make command: ${cmd}"
${cmd}

cmd="make install"
echo "Running make command: ${cmd}"
${cmd}

cmd="make -j16 gtest_ngtf"
echo "Running make command: ${cmd}"
${cmd}

xtime="$(date)"
echo  ' '
echo  "===== Creating libngraph_dist.tgz at ${xtime} ====="
echo  ' '

cd "${bridge_dir}"

if [ ! -d "${ngraph_dist_dir}" ] ; then
    ( >&2 echo '***** Error: *****' )
    ( >&2 echo "ngraph_dist directory does not exist -- this likely indicates a build failure: ${ngraph_dist_dir}" )
    exit 1
fi

if [ ! -f "${libngraph_so}" ] ; then
    ( >&2 echo '***** Error: *****' )
    ( >&2 echo "libngraph_bridge.so file does not exist -- this likely indicatesa build failure: ${libngraph_so}" )
    exit 1
fi

set -x

# Create the directory for libngraph distibution
mkdir "${libngraph_dist_dir}"

# Copy the ngraph wheel into the build directory, for easy access in Jenkins
cp "${ngraph_wheel_dir}"/*.whl "${bridge_dir}"

# Copy the ngraph_dist directory into the libngraph distribution directory
cp -r "${ngraph_dist_dir}" "${libngraph_dist_dir}/ngraph_dist"

# Copy the libngraph_bridge.so file into the libngraph distribution directory
cp "${libngraph_so}" "${libngraph_dist_dir}"

pwd
ls -l

# We use the directory name only here because tar (understandably) does not
# like an absolute path (to avoid making non-portable tarballs)
tar czf "${libngraph_tarball}" libngraph_dist

set +x

xtime="$(date)"
echo  ' '
echo  "===== Installing nGraph Wheel at ${xtime} ====="
echo  ' '

echo 'md5sum of wheel that was built above, and will be installed:'
cd "${bridge_dir}"
md5sum ngraph*.whl

cd "${bridge_dir}"
pip install ngraph*.whl

xtime="$(date)"
echo  ' '
echo  "===== Run Bridge CI Test Scripts at ${xtime} ====="
echo  ' '

cd "${bbuild_dir}/test"
export NGRAPH_IMAGENET_DATASET="${imagenet_dataset}"
export NGRAPH_TRAINED_MODEL="${trained_resnet50_model}"
"${bridge_dir}/test/ci/run-premerge-ci-checks.sh"

xtime="$(date)"
echo  ' '
echo  "===== Run Sanity Check for Plugins at ${xtime} ====="
echo  ' '

cd "${bridge_dir}/test"

python install_test.py

xtime="$(date)"
echo  ' '
echo  "===== Deactivating the Virtual Environment at ${xtime} ====="
echo  ' '

deactivate

xtime="$(date)"
echo ' '
echo "===== Completed Tensorflow Build and Test at ${xtime} ====="
echo ' '
