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

# Script Parameters:
#
# $1  OPTIONAL TensorFlow directory. If defined, which directory to build
#     the tensorflow wheel and library in.
#
# Environment Variables:
#
# NG_TF_PY_VERSION  Python version to use for build
#
# NOTES ABOUT THE MAC BUILD:
#
# - This script was developed for and has been tested with MacOS High
#   Sierra (10.13.6).  It has not been tested with earlier versions of
#   MacOS.
#
# - This script is designed to use with the "pyenv" command, which controls
#   which python versions are installed.  "pyenv" is the recommended Python
#   version manager to use with "homebrew", as "homebrew" cannot easily
#   support installing earlier minor versions of Python like 3.5.6 (when
#   3.7 is the "latest").  As of this writing, TensorFlow does not build
#   with Python 3.7, so we need to be able to install (and manage) older
#   Python versions like 3.5.6, which MacOS does not provide.
#
# - This script assumes that "homebrew" has been installed and enough
#   packages have been brew-installed (as per the TensorFlow MacOS build
#   instructions).  Homebrew was used because the TensorFlow project
#   recommends it for MacOS builds.

set -e  # Make sure we exit on any command that returns non-zero
set -o pipefail # Make sure cmds in pipe that are non-zero also fail immediately

if [ -z "$1" ] ; then
  echo "TensorFlow directory was not specified as the script parameter."
  echo "The TensorFlow wheel will therefore be downloaded using pip, and not built."
  tf_dir=''  # tf_dir is empty, i.e. -z "$tf_dir" is true
else
  tf_dir="$1"
fi

# Default is Python 2, but can override with NG_TF_PY_VERSION env. variable
export PYTHON_VERSION_NUMBER="${NG_TF_PY_VERSION}"
if [ -z "${PYTHON_VERSION_NUMBER}" ] ; then
    export PYTHON_VERSION_NUMBER=2
fi
# On the Mac, we use "pyenv" to control the path which "python" points to
# DISABLED  export PYTHON_BIN_PATH="/usr/bin/python$PYTHON_VERSION_NUMBER"

# Set up some important known directories
mac_build_dir="${PWD}"
bridge_dir="${PWD}/../../.."  # Relative to ngraph-tf/test/ci/macos
bbuild_dir="${bridge_dir}/BUILD-BRIDGE-MAC"
ci_dir="${bridge_dir}/test/ci/docker"
# DISABLED  dataset_dir='/dataset'
# DISABLED  trained_dir='/trained_dataset'
venv_dir="${bridge_dir}/VENV_PYTHON${PYTHON_VERSION_NUMBER}"
ngraph_dist_dir="${bbuild_dir}/ngraph/ngraph_dist"
ngraph_wheel_dir="${bbuild_dir}/python/dist"
libngraph_so="${bbuild_dir}/src/libngraph_bridge.dylib"
libngraph_dist_dir="${bridge_dir}/libngraph_mac_dist"  # Directory to save plugin artifacts in
libngraph_tarball="${bridge_dir}/libngraph_mac_dist.tgz"  # Tarball artifact to send to Artifactory
#DISABLED  imagenet_dataset="${dataset_dir}/Imagenet_Validation"
#DISABLED  trained_resnet50_model="${trained_dir}/ngraph_tensorflow/fully_trained/resnet50"

# HOME is expected to be /home/dockuser.  See script run-as-user.sh, which
# sets this up.

# Set up a directory to build the wheel in, so we know where to grab
# the wheel from later.  If the directory already exists, remove it.
export WHEEL_BUILD_DIR="${tf_dir}/BUILD_WHEEL"

echo "In $(basename ${0}):"
echo ''
echo "  mac_build_dir=${mac_build_dir}"
echo "  bridge_dir=${bridge_dir}"
echo "  bbuild_dir=${bbuild_dir}"
echo "  tf_dir=${tf_dir}"
echo "  ci_dir=${ci_dir}"
#DISABLED  echo "  dataset_dir=${dataset_dir}"
#DISABLED  echo "  trained_dir=${trained_dir}"
echo "  venv_dir=${venv_dir}"
echo "  ngraph_dist_dir=${ngraph_dist_dir}"
echo "  ngraph_wheel_dir=${ngraph_wheel_dir}"
echo "  libngraph_so=${libngraph_so}"
echo "  libngraph_dist_dir=${libngraph_dist_dir}"
echo "  libngraph_tarball=${libngraph_tarball}"
#DISABLED  echo "  imagenet_dataset=${imagenet_dataset}"
#DISABLED  echo "  trained_resnet50_model=${trained_resnet50_model}"
echo ''
echo "  HOME=${HOME}"
echo "  PYTHON_VERSION_NUMBER=${PYTHON_VERSION_NUMBER}"
#DISABLED  echo "  PYTHON_BIN_PATH=${PYTHON_BIN_PATH}"
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

# DISABLED  if [ ! -d "${dataset_dir}" ] ; then
# DISABLED      ( >&2 echo '***** Error: *****' )
# DISABLED      ( >&2 echo "Datset directory ${dataset_dir} does not seem to be mounted inside the Docker container" )
# DISABLED      exit 1
# DISABLED  fi

# DISABELD  if [ ! -d "${imagenet_dataset}" ] ; then
# DISABLED      ( >&2 echo '***** Error: *****' )
# DISABLED      ( >&2 echo "The validation dataset for ImageNet does not seem to be found: ${imagenet_dataset}" )
# DISABLED      exit 1
# DISABLED  fi

# DISABLED  if [ ! -d "${trained_resnet50_model}" ] ; then
# DISABLED      ( >&2 echo '***** Error: *****' )
# DISABLED      ( >&2 echo "The pretrained model for resnet50 CI testing does not seem to be found: ${trained_resnet50_model}" )
# DISABLED      exit 1
# DISABLED  fi

xtime="$(date)"
echo  ' '
echo  "===== Enabling Python Version (on the Mac) at ${xtime} ====="
echo  ' '

eval "$(pyenv init -)"  # Initialize pyenv shim for this build process

if [ "${PYTHON_VERSION_NUMBER}" = '2' ] ; then
    pyenv shell 2.7.1
else
    pyenv shell 3.5.6
fi

# We have to run "set -u" here because the "pyenv" command uses unbound variables
# in its checks
set -u  # No unset variables from this point on

# DISABLED  xtime="$(date)"
# DISABLED  echo  ' '
# DISABLED  echo  "===== Starting nGraph TensorFlow Bridge Source Code Format Check at ${xtime} ====="
# DISABLED  echo  ' '
# DISABLED
# DISABLED  cd "${bridge_dir}"
# DISABLED  maint/check-code-format.sh

# Make sure the Bazel cache is in /tmp, as docker images have too little space
# in the root filesystem, where /home (and $HOME/.cache) is.  Even though we
# may not be using the Bazel cache in the builds (in docker), we do this anyway
# in case we decide to turn the Bazel cache back on.
echo "Adjusting bazel cache to be located in ${bridge_dir}/bazel-cache"
rm -fr "$HOME/.cache"
# Remove the temporary bazel-cache if it was left around in a previous build
if [ -d "${bridge_dir}/bazel-cache" ] ; then
  rm -fr "${bridge_dir}/bazel-cache"
fi
mkdir "${bridge_dir}/bazel-cache"
ln -s "${bridge_dir}/bazel-cache" "$HOME/.cache"

xtime="$(date)"
echo  ' '
echo  "===== Setting Up Virtual Environment for Tensorflow Wheel at ${xtime} ====="
echo  ' '

# Make sure the bash shell prompt variables are set, as virtualenv crashes
# if PS2 is not set.
PS1='prompt> '
PS2='prompt-more> '
#ORIG: virtualenv --system-site-packages -p "${PYTHON_BIN_PATH}" "${venv_dir}"
virtualenv --system-site-packages "${venv_dir}"
source "${venv_dir}/bin/activate"

# Only build TensorFlow if a tensorflow directory was provided
if [ -z "${tf_dir}" ] ; then

    xtime="$(date)"
    echo  ' '
    echo  "===== Installing Tensorflow Wheel at ${xtime} ====="
    echo  ' '

    set -x
    # If installing into the OS, use:
    # sudo --preserve-env --set-home pip install --ignore-installed ${PIP_INSTALL_EXTRA_ARGS:-} "${WHEEL_FILE}"
    # Here we are installing into a virtual environment, so DO NOT USE SUDO!!!
    pip install -U tensorflow==1.12.0
    set +x

else 

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

    bazel build --config=opt --verbose_failures //tensorflow/tools/pip_package:build_pip_package

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
    bazel build --config=opt //tensorflow:libtensorflow_cc.so

fi  # Else block for: if [ -z "${tf_dir} ] ; then

xtime="$(date)"
echo  ' '
echo  "===== Starting nGraph TensorFlow Bridge Build at ${xtime} ====="
echo  ' '

cd "${bridge_dir}"

mkdir "${bbuild_dir}"
cd "${bbuild_dir}"
# Only build Google tests if the TensorFlow C++ library was built
PARALLEL='-j4'
if [ -z "${tf_dir}" ] ; then  # TensorFlow installed from Internet
    cmake ..
    make ${PARALLEL}
    make install
else
    cmake -DUNIT_TEST_ENABLE=TRUE -DTF_SRC_DIR="${tf_dir}" ..
    make ${PARALLEL}
    make install
    make ${PARALLEL} gtest_ngtf
fi

xtime="$(date)"
echo  ' '
echo  "===== Creating libngraph_dist.tgz at ${xtime} ====="
echo  ' '

cd "${bridge_dir}"

if [ ! -d "${ngraph_dist_dir}" ] ; then
    ( >&2 echo '***** Error: *****' )
    ( >&2 echo "ngraph_dist directory does not exist -- this likely indicatesa build failure: ${ngraph_dist_dir}" )
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
tar czf "${libngraph_tarball}" libngraph_mac_dist

set +x

xtime="$(date)"
echo  ' '
echo  "===== Installing nGraph Wheel at ${xtime} ====="
echo  ' '

cd "${bridge_dir}"
pip install -U ngraph*.whl

# DISABLED  xtime="$(date)"
# DISABLED  echo  ' '
# DISABLED  echo  "===== Run Bridge CI Test Scripts at ${xtime} ====="
# DISABLED  echo  ' '
# DISABLED
# DISABLED  cd "${bbuild_dir}/test"
# DISABLED  export NGRAPH_IMAGENET_DATASET="${imagenet_dataset}"
# DISABLED  export NGRAPH_TRAINED_MODEL="${trained_resnet50_model}"
# DISABLED  "${bridge_dir}/test/ci/run-premerge-ci-checks.sh"

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
