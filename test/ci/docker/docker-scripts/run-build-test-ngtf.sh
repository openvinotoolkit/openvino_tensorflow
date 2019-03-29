
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

echo ' '
echo 'python version is:'
python --version

echo ' '
echo 'python2 version is:'
python2 --version

echo ' '
echo 'python3 version is:'
python3 --version

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
echo  "===== Setting Up Virtual Environment for Code-Format Check at ${xtime} ====="
echo  ' '

# Make sure the bash shell prompt variables are set, as virtualenv crashes
# if PS2 is not set.
PS1='prompt> '
PS2='prompt-more> '
virtualenv --system-site-packages -p /usr/bin/python3 "${venv_dir}"
source "${venv_dir}/bin/activate"

# yapf and futures are needed for code-format checks (ngraph-tf PR#211)
pip install yapf==0.26.0
pip install futures

xtime="$(date)"
echo  ' '
echo  "===== Starting nGraph TensorFlow Bridge Source Code Format Check at ${xtime} ====="
echo  ' '

cd "${bridge_dir}"
maint/check-code-format.sh

xtime="$(date)"
echo  ' '
echo  "===== Deactivating Virtual Environment at ${xtime} ====="
echo  ' '

deactivate

xtime="$(date)"
echo  ' '
echo  "===== Run build_ngtf.py at ${xtime} ====="
echo  ' '

cd "${bridge_dir}"
./build_ngtf.py

xtime="$(date)"
echo  ' '
echo  "===== Run test_ngtf.py at ${xtime} ====="
echo  ' '

cd "${bridge_dir}"
./test_ngtf.py

xtime="$(date)"
echo ' '
echo "===== Completed Tensorflow Build and Test at ${xtime} ====="
echo ' '
