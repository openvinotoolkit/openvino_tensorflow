#!/bin/bash
set -e
set -u

# ******************************************************************************
# Copyright 2017-2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

declare SRC_DIRS="src examples test logging tools diagnostics python"

# NOTE: The results of `clang-format` depend _both_ of the following factors:
# - The `.clang-format` file, and
# - The particular version of the `clang-format` program being used.
#
# For this reason, this script specifies the exact version of clang-format to be used.
# Similarly for python/yapf, we shall use Python 3 and yapf 0.26.0
declare _intelnervana_clang_format_lib_SCRIPT_NAME="${BASH_SOURCE[${#BASH_SOURCE[@]} - 1]}"
declare _maint_SCRIPT_DIR="$( cd $(dirname "${_intelnervana_clang_format_lib_SCRIPT_NAME}") && pwd )"
source "${_maint_SCRIPT_DIR}/bash_lib.sh"
declare SED_FLAGS
if [[ "$(uname)" == 'Darwin' ]]; then
    SED_FLAGS='-En'
else
    SED_FLAGS='-rn'
fi

declare CLANG_FORMAT_BASENAME="clang-format-3.9"
declare REQUIRED_CLANG_FORMAT_VERSION=3.9
declare YAPF_FORMAT_BASENAME="yapf"
declare REQUIRED_YAPF_FORMAT_VERSION=0.26.0

# Check the YAPF format
declare YAPF_VERSION=`python -c "import yapf; print(yapf.__version__)"`

if [[ "${YAPF_VERSION}" != "${REQUIRED_YAPF_FORMAT_VERSION}" ]] ; then
    echo -n "Unable to match version for ${YAPF_FORMAT_BASENAME}"
    echo -n " Required: ${REQUIRED_YAPF_FORMAT_VERSION}"
    echo  " Installed: ${YAPF_VERSION}"
    exit -1
fi

declare YAPF_FORMAT_PROG="python3 -m ${YAPF_FORMAT_BASENAME}"

declare THIS_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

source "${THIS_SCRIPT_DIR}/bash_lib.sh"
source "${THIS_SCRIPT_DIR}/clang_format_lib.sh"

declare CLANG_FORMAT_PROG
if ! CLANG_FORMAT_PROG="$(which "${CLANG_FORMAT_BASENAME}")"; then
    bash_lib_die "Unable to find program ${CLANG_FORMAT_BASENAME}" >&2
fi

format_lib_verify_version "${CLANG_FORMAT_PROG}" "${REQUIRED_CLANG_FORMAT_VERSION}" "CLANG"
bash_lib_status "Verified that '${CLANG_FORMAT_PROG}' has version '${REQUIRED_CLANG_FORMAT_VERSION}'"

pushd "${THIS_SCRIPT_DIR}/.."

declare PYBIND_WRAPPER="python/pyngraph"

declare ROOT_SUBDIR
for ROOT_SUBDIR in ${SRC_DIRS}; do
    if ! [[ -d "${ROOT_SUBDIR}" ]]; then
	    bash_lib_status "In directory '$(pwd)', no subdirectory named '${ROOT_SUBDIR}' was found."
    else
        bash_lib_status "Formatting C/C++ code in: '$(pwd)/${ROOT_SUBDIR}'"

        # Note that we restrict to "-type f" to exclude symlinks. Emacs sometimes
        # creates dangling symlinks with .cpp/.hpp suffixes as a sort of locking
        # mechanism, and this confuses clang-format.
        #
        # We also skip any dir named "cpu_codegen" in case there are
        # nGraph-generated files lying around from a test run.
        find "${ROOT_SUBDIR}"                                       \
          -name cpu_codegen -prune -o                               \
          \( -type f -and \( -name '*.cc' -or -name '*.h'           \
                             -or -name '*.cpp' -or -name '*.hpp' \) \
             -print \) | xargs "${CLANG_FORMAT_PROG}" -i -style=file

        bash_lib_status "Formatting Python code in: '$(pwd)/${ROOT_SUBDIR}' ..."
        declare SRC_FILE
        # ignore the .in.py file (python/setup.in.py) which has format that crashes yapf
        for SRC_FILE in $(find "${ROOT_SUBDIR}"                                      \
                        -name *.in.py -prune -o                                   \
                        \( -type f -and \( -name '*.py' \)                        \
                            -print \) ); do
            python3 -m yapf -i -p --style google --no-local-style "${SRC_FILE}"
        done
    fi
done

popd
