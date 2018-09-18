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
# Similarly for python/yapf, we shall use Python 2 and yapf 0.24
declare _intelnervana_clang_format_lib_SCRIPT_NAME="${BASH_SOURCE[${#BASH_SOURCE[@]} - 1]}"
declare _maint_SCRIPT_DIR="$( cd $(dirname "${_intelnervana_clang_format_lib_SCRIPT_NAME}") && pwd )"
source "${_maint_SCRIPT_DIR}/bash_lib.sh"
declare SED_FLAGS
if [[ "$(uname)" == 'Darwin' ]]; then
    SED_FLAGS='-En'
else
    SED_FLAGS='-rn'
fi

# Find out python version. Use yapf only when in Python 2
if PYTHON_VERSION=$(python -c 'import sys; print(sys.version_info[:][0])')
then
    if [[ "2" != "${PYTHON_VERSION}" ]]; then
        echo "Python reports version number '${PYTHON_VERSION}' so will skip yapf formatting. Please use Python2"
    fi
else
    bash_lib_print_error "Failed invocation of Python."
    exit 1
fi


declare CLANG_FORMAT_BASENAME="clang-format-3.9"
declare REQUIRED_CLANG_FORMAT_VERSION=3.9
if [[ "2" == "${PYTHON_VERSION}" ]]; then
    declare YAPF_FORMAT_BASENAME="yapf"
    declare REQUIRED_YAPF_FORMAT_VERSION=0.24
fi

declare THIS_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

source "${THIS_SCRIPT_DIR}/bash_lib.sh"
source "${THIS_SCRIPT_DIR}/clang_format_lib.sh"

declare CLANG_FORMAT_PROG
if ! CLANG_FORMAT_PROG="$(which "${CLANG_FORMAT_BASENAME}")"; then
    bash_lib_die "Unable to find program ${CLANG_FORMAT_BASENAME}" >&2
fi

if [[ "2" == "${PYTHON_VERSION}" ]]; then
    declare YAPF_FORMAT_PROG
    if ! YAPF_FORMAT_PROG="$(which "${YAPF_FORMAT_BASENAME}")"; then
        bash_lib_die "Unable to find program ${YAPF_FORMAT_BASENAME}" >&2
    fi
fi

format_lib_verify_version "${CLANG_FORMAT_PROG}" "${REQUIRED_CLANG_FORMAT_VERSION}" "CLANG"
bash_lib_status "Verified that '${CLANG_FORMAT_PROG}' has version '${REQUIRED_CLANG_FORMAT_VERSION}'"
if [[ "2" == "${PYTHON_VERSION}" ]]; then
    format_lib_verify_version "${YAPF_FORMAT_PROG}" "${REQUIRED_YAPF_FORMAT_VERSION}" "YAPF"
    bash_lib_status "Verified that '${YAPF_FORMAT_PROG}' has version '${REQUIRED_YAPF_FORMAT_VERSION}'"
fi

pushd "${THIS_SCRIPT_DIR}/.."

declare PYBIND_WRAPPER="python/pyngraph"

declare ROOT_SUBDIR
for ROOT_SUBDIR in ${SRC_DIRS}; do
    if ! [[ -d "${ROOT_SUBDIR}" ]]; then
	    bash_lib_status "In directory '$(pwd)', no subdirectory named '${ROOT_SUBDIR}' was found."
    else
        bash_lib_status "About to format C/C++ code in directory tree '$(pwd)/${ROOT_SUBDIR}' ..."

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

        bash_lib_status "Done."

        if [[ "2" == "${PYTHON_VERSION}" ]]; then
            bash_lib_status "About to format Python code in directory tree '$(pwd)/${ROOT_SUBDIR}' ..."
            declare SRC_FILE
            # ignore the .in.py file (python/setup.in.py) which has format that crashes yapf
            for SRC_FILE in $(find "${ROOT_SUBDIR}"                                      \
                            -name *.in.py -prune -o                                   \
                            \( -type f -and \( -name '*.py' \)                        \
                                -print \) ); do
                "${YAPF_FORMAT_PROG}"  -i -p --style google --no-local-style "${SRC_FILE}"
            done
            bash_lib_status "Done."
        fi
    fi
done

popd
