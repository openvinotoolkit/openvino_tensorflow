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

# Set up some important known directories
mac_build_dir="${PWD}"
bridge_dir="${PWD}/../../.."  # Relative to ngraph-tf/test/ci/macos
bbuild_dir="${bridge_dir}/build"

echo "In $(basename ${0}):"
echo ''
echo "  mac_build_dir=${mac_build_dir}"
echo "  bridge_dir=${bridge_dir}"
echo "  bbuild_dir=${bbuild_dir}"
echo ''
echo "  HOME=${HOME}"

# Do some up-front checks, to make sure necessary directories are in-place and
# build directories are not-in-place

if [ -d "${bbuild_dir}" ] ; then
    ( >&2 echo '***** Error: *****' )
    ( >&2 echo "Bridge build directory already exists -- please remove it before calling this script: ${bbuild_dir}" )
    exit 1
fi

xtime="$(date)"
echo  ' '
echo  "===== Enabling Python Version 3 (on the Mac) at ${xtime} ====="
echo  ' '

# Currently build_ngtf.py only works with Python 3
eval "$(pyenv init -)"  # Initialize pyenv shim for this build process
pyenv shell 3.5.6

# We have to run "set -u" here because the "pyenv" command uses unbound variables
# in its checks
set -u  # No unset variables from this point on

# Make sure the Bazel cache is in the workspace, so it does not collide with
# other Jenkins jobs.  Also to make sure to start with a fresh bazel cache,
# to avoid cache corruption from other builds.
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
echo  "===== Run MacOS Build Using build_ngtf.py at ${xtime} ====="
echo  ' '

cd "${bridge_dir}"
./build_ngtf.py

xtime="$(date)"
echo  ' '
echo  "===== Run MacOS Unit-Tests Using test_ngtf.py at ${xtime} ====="
echo  ' '

cd "${bridge_dir}"
./test_ngtf.py

xtime="$(date)"
echo ' '
echo "===== Completed MacOS Tensorflow Build and Test at ${xtime} ====="
echo ' '
