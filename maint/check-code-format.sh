#!/bin/bash
set -e
set -u

# ******************************************************************************
# Copyright (C) 2023 Intel Corporation
 
# SPDX-License-Identifier: Apache-2.0
# ******************************************************************************

declare SRC_DIRS=${1:-openvino_tensorflow examples experimental test logging tools diagnostics python}

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
declare YAPF_VERSION=`python -c "import yapf; print(yapf.__version__)"`

if [[ "${YAPF_VERSION}" != "${REQUIRED_YAPF_FORMAT_VERSION}" ]] ; then
    echo -n "Unable to match version for ${YAPF_FORMAT_BASENAME}"
    echo -n " Required: ${REQUIRED_YAPF_FORMAT_VERSION}"
    echo  " Installed: ${YAPF_VERSION}"
    exit -1
fi

declare YAPF_FORMAT_PROG="python3 -m ${YAPF_FORMAT_BASENAME}"

declare CLANG_FORMAT_PROG
if ! CLANG_FORMAT_PROG="$(which "${CLANG_FORMAT_BASENAME}")"; then
    bash_lib_die "Unable to find program  ${CLANG_FORMAT_BASENAME}" >&2
fi

format_lib_verify_version "${CLANG_FORMAT_PROG}" "${REQUIRED_CLANG_FORMAT_VERSION}" "CLANG"
bash_lib_status "Verified that '${CLANG_FORMAT_PROG}' has version '${REQUIRED_CLANG_FORMAT_VERSION}'"

declare -a FAILED_FILES_CLANG=()
declare NUM_FILES_CHECKED_CLANG=0

declare -a FAILED_FILES_YAPF=()
declare NUM_FILES_CHECKED_YAPF=0

pushd "${THIS_SCRIPT_DIR}/.."

declare ROOT_SUBDIR
for ROOT_SUBDIR in ${SRC_DIRS}; do
    if ! [[ -d "${ROOT_SUBDIR}" ]]; then
        bash_lib_status "In directory '$(pwd)', no subdirectory named '${ROOT_SUBDIR}' was found."
    else
        bash_lib_status "Checking C/C++ formatting in directory: '$(pwd)/${ROOT_SUBDIR}' "
        declare SRC_FILE
        # Note that we restrict to "-type f" to exclude symlinks. Emacs sometimes
        # creates dangling symlinks with .cpp/.hpp suffixes as a sort of locking
        # mechanism, and this confuses clang-format.
        for SRC_FILE in $(find "${ROOT_SUBDIR}"                                      \
                           \( -type f -and \( -name '*.cc' -or -name '*.h'           \
                                              -or -name '*.cpp' -or -name '*.hpp' \) \
                              -print \) ); do
            if "${CLANG_FORMAT_PROG}" -style=file -output-replacements-xml "${SRC_FILE}" | grep -c "<replacement " >/dev/null; then
                FAILED_FILES_CLANG+=( "${SRC_FILE}" )
            fi
            NUM_FILES_CHECKED_CLANG=$((NUM_FILES_CHECKED_CLANG+1))
        done

        bash_lib_status "Checking Python formatting in directory:  '$(pwd)/${ROOT_SUBDIR}'"
        declare SRC_FILE
        # ignore the .in.py file (python/setup.in.py) which has format that crashes yapf
        for SRC_FILE in $(find "${ROOT_SUBDIR}"                                      \
                        -name *.in.py -prune -o                                   \
                        \( -type f -and \( -name '*.py' \)                        \
                            -print \) ); do
            if ! python3 -m yapf  -d -p --style google --no-local-style "${SRC_FILE}" >/dev/null; then
                FAILED_FILES_YAPF+=( "${SRC_FILE}" )
            fi
            NUM_FILES_CHECKED_YAPF=$((NUM_FILES_CHECKED_YAPF+1))
        done
    fi
done

# Check format of py files at root (build_ovtf.py, build_tf.py, test_ovtf.py etc)
for SRC_FILE in $(find . -maxdepth 1  -name '*py' -print); do
    if ! python3 -m yapf -d -p --style google --no-local-style "${SRC_FILE}">/dev/null; then
        FAILED_FILES_YAPF+=( "${SRC_FILE}" )
    fi
done

popd

if [[ ${#FAILED_FILES_CLANG[@]} -eq 0 ]]; then
    bash_lib_status "All ${NUM_FILES_CHECKED_CLANG}  C/C++ files pass the code-format check."
else
    echo "${#FAILED_FILES_CLANG[@]} of ${NUM_FILES_CHECKED_CLANG} source files failed the code-format check:"
    declare FAILED_SRC_FILE
    for FAILED_SRC_FILE in ${FAILED_FILES_CLANG[@]}; do
        echo "    ${FAILED_SRC_FILE}"
    done
    exit 1
fi

if [[ ${#FAILED_FILES_YAPF[@]} -eq 0 ]]; then
    bash_lib_status "All ${NUM_FILES_CHECKED_YAPF}  Python files pass the code-format check."
else
    echo "${#FAILED_FILES_YAPF[@]} of ${NUM_FILES_CHECKED_YAPF} source files failed the code-format check:"
    declare FAILED_SRC_FILE
    for FAILED_SRC_FILE in ${FAILED_FILES_YAPF[@]}; do
        echo "    ${FAILED_SRC_FILE}"
    done
    exit 1
fi
