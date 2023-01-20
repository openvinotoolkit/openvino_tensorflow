#!/bin/bash
set -e
set -u

# ******************************************************************************
# Copyright (C) 2023 Intel Corporation
 
# SPDX-License-Identifier: Apache-2.0
# ******************************************************************************

declare SRC_DIRS=${1:-openvino_tensorflow examples test logging tools diagnostics python}

# NOTE: The results of `clang-format` depend _both_ of the following factors:
# - The `.clang-format` file, and
# - The particular version of the `clang-format` program being used.
#
# For this reason, this script specifies the exact version of clang-format to be used.
# Similarly for python/yapf, we shall use Python 3 and yapf 0.26.0

declare CLANG_FORMAT_BASENAME="clang-format-3.9"
declare REQUIRED_CLANG_FORMAT_VERSION=3.9
declare YAPF_FORMAT_BASENAME="yapf"
declare REQUIRED_YAPF_FORMAT_VERSION=0.26.0

declare THIS_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

source "${THIS_SCRIPT_DIR}/bash_lib.sh"
source "${THIS_SCRIPT_DIR}/clang_format_lib.sh"

declare SED_FLAGS
if [[ "$(uname)" == 'Darwin' ]]; then
    SED_FLAGS='-En'
else
    SED_FLAGS='-rn'
fi

# Check the YAPF format
declare YAPF_VERSION=`python3 -c "import yapf; print(yapf.__version__)"`

if [[ "${YAPF_VERSION}" != "${REQUIRED_YAPF_FORMAT_VERSION}" ]] ; then
    echo -n "Unable to match version for ${YAPF_FORMAT_BASENAME}"
    echo -n " Required: ${REQUIRED_YAPF_FORMAT_VERSION}"
    echo  " Installed: ${YAPF_VERSION}"
    exit -1
fi

declare YAPF_FORMAT_PROG="python3 -m ${YAPF_FORMAT_BASENAME}"

declare CLANG_FORMAT_PROG
if ! CLANG_FORMAT_PROG="$(which "${CLANG_FORMAT_BASENAME}")"; then
    bash_lib_die "Unable to find program ${CLANG_FORMAT_BASENAME}" >&2
fi

format_lib_verify_version "${CLANG_FORMAT_PROG}" "${REQUIRED_CLANG_FORMAT_VERSION}" "CLANG"
bash_lib_status "Verified that '${CLANG_FORMAT_PROG}' has version '${REQUIRED_CLANG_FORMAT_VERSION}'"

pushd "${THIS_SCRIPT_DIR}/.."

declare PYBIND_WRAPPER="python3/pyngraph"

declare ROOT_SUBDIR
for ROOT_SUBDIR in ${SRC_DIRS}; do
    if ! [[ -d "${ROOT_SUBDIR}" ]]; then
	    bash_lib_status "In directory '$(pwd)', no subdirectory named '${ROOT_SUBDIR}' was found."
    else
        bash_lib_status "Formatting C/C++ code in: '$(pwd)/${ROOT_SUBDIR}'"

        # Note that we restrict to "-type f" to exclude symlinks. Emacs sometimes
        # creates dangling symlinks with .cpp/.hpp suffixes as a sort of locking
        # mechanism, and this confuses clang-format.
        find "${ROOT_SUBDIR}"                                       \
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

# Format py files at root (build_ovtf.py, build_tf.py, test_ovtf.py etc)
for SRC_FILE in $(find . -maxdepth 1  -name '*.py' -print); do
    python3 -m yapf -i -p --style google --no-local-style "${SRC_FILE}"
done

popd
