#!  /bin/bash

# This script is designed to be called from within a docker container.
# It is installed into a docker image.  It will not run outside the container.

set -e  # Make sure we exit on any command that returns non-zero
set -u  # No unset variables
set -o pipefail # Make sure cmds in pipe that are non-zero also fail immediately


if [ -z "${PYTHON_VERSION_NUMBER:-}" ] ; then
    ( >&2 echo "Env. variable PYTHON_VERSION_NUMBER has not been set" )
    exit 1
fi

export PYTHON_BIN_PATH="/usr/bin/python$PYTHON_VERSION_NUMBER"

# Set up some important known directories
bridge_dir='/home/dockuser/bridge'
bbuild_dir="${bridge_dir}/BUILD-BRIDGE"
ngtf_dir='/home/dockuser/ngtf'
ci_dir="${bridge_dir}/test/ci/docker"
venv_dir="/tmp/venv_python${PYTHON_VERSION_NUMBER}"

# HOME is expected to be /home/dockuser.  See script run-as-user.sh, which
# sets this up.

# Set up a directory to build the wheel in, so we know where to grab
# the wheel from later.  If the directory already exists, remove it.
export WHEEL_BUILD_DIR="${ngtf_dir}/BUILD_WHEEL"

# Point to the ngraph dynamic libraries
export LD_LIBRARY_PATH="${bridge_dir}/ngraph_dist/lib"

echo "In $(basename ${0}):"
echo ''
echo "  bridge_dir=${bridge_dir}"
echo "  bbuild_dir=${bbuild_dir}"
echo "  ngtf_dir=${ngtf_dir}"
echo "  ci_dir=${ci_dir}"
echo "  venv_dir=${venv_dir}"
echo ''
echo "  HOME=${HOME}"
echo "  PYTHON_VERSION_NUMBER=${PYTHON_VERSION_NUMBER}"
echo "  PYTHON_BIN_PATH=${PYTHON_BIN_PATH}"
echo "  LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
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
echo  "===== Configuring Tensorflow Build at ${xtime} ====="
echo  ' '

export CC_OPT_FLAGS="-march=native"
export USE_DEFAULT_PYTHON_LIB_PATH=1
export TF_ENABLE_XLA=1

export TF_NEED_MKL=0
export TF_DOWNLOAD_MKL=0

export TF_NEED_JEMALLOC=1
export TF_NEED_GCP=0
export TF_NEED_HDFS=0
export TF_NEED_VERBS=0
export TF_NEED_OPENCL=0
export TF_NEED_CUDA=0
export TF_NEED_MPI=0

cd "${ngtf_dir}"
./configure

xtime="$(date)"
echo  ' '
echo  "===== Starting Tensorflow Binaries Build at ${xtime} ====="
echo  ' '

cd "${ngtf_dir}"

bazel build --verbose_failures ${BAZEL_BUILD_EXTRA_FLAGS:-} //tensorflow/tools/pip_package:build_pip_package | tee z-tf15-build-log.txt

xtime="$(date)"
echo  ' '
echo  "===== Starting Tensorflow Wheel Build at ${xtime} ====="
echo  ' '

bazel-bin/tensorflow/tools/pip_package/build_pip_package "${WHEEL_BUILD_DIR}"

xtime="$(date)"
echo  ' '
echo  "===== Setting Up Virtual Environment for Tensorflow Wheel at ${xtime} ====="
echo  ' '

# Make sure the bash shell prompt variables are set, as virtualenv crashes
# if PS2 is not set.
PS1='prompt> '
PS2='prompt-more> '
virtualenv -p "${PYTHON_BIN_PATH}" "${venv_dir}"
source "${venv_dir}/bin/activate"

xtime="$(date)"
echo  ' '
echo  "===== Installing Tensorflow Wheel at ${xtime} ====="
echo  ' '

set -x
echo ' '
echo 'Proxy settings in exports:'
export | grep -i proxy
echo ' '
echo 'Building the wheel:'
cd "${ngtf_dir}"
declare WHEEL_FILE="$(find "${WHEEL_BUILD_DIR}" -name '*.whl')"
# If installing into the OS, use:
# sudo --preserve-env --set-home pip install --ignore-installed ${PIP_INSTALL_EXTRA_ARGS:-} "${WHEEL_FILE}"
# Here we are installing into a virtual environment, so DO NOT USE SUDO!!!
pip install --ignore-installed ${PIP_INSTALL_EXTRA_ARGS:-} "${WHEEL_FILE}"
set +x

xtime="$(date)"
echo  ' '
echo  "===== Starting nGraph TensorFlow Bridge Build at ${xtime} ====="
echo  ' '

# Temporary kludge.  See check for ~/ngraph_dist above for more
# detailed comments.
ln -s "${bridge_dir}/ngraph_dist" "${HOME}/ngraph_dist"

cd "${bridge_dir}"

mkdir "${bbuild_dir}"
cd "${bbuild_dir}"
cmake ..
make -j16

xtime="$(date)"
echo  ' '
echo  "===== Run Bridge CI Test Scripts at ${xtime} ====="
echo  ' '

cd "${bridge_dir}/test/ci"

export USER_PLUGIN_PATH="${bbuild_dir}/src/libngraph_plugin.so"
export TF_ROOT="${ngtf_dir}"
"${bridge_dir}/test/ci/run-premerge-ci-checks.sh"

xtime="$(date)"
echo  ' '
echo  "===== Deactivating the Virtual Environment at ${xtime} ====="
echo  ' '

deactivate

xtime="$(date)"
echo ' '
echo "===== Completed Tensorflow Build and Test at ${xtime} ====="
echo ' '
