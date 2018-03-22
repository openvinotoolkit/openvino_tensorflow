#!/bin/bash
#*******************************************************************************
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
#*******************************************************************************

#---------------------------------------------------------------------------------------------------
# Some TF-supplied MNIST scripts attempt to download image data if it cannot be found on the local
# filesystem.  In some cases, that dowloading fails due to HTTPS / proxy / firewall issues.
#
# The URLs in this script are taken from the TensorFlow 1.3 version of
# /usr/local/lib/python2.7/dist-packages/tensorflow/contrib/learn/python/learn/datasets/mnist.py
#
# This script provides a manual means of downloading those files to the local filesystem.
#---------------------------------------------------------------------------------------------------

if [[ ${#} !=  1 ]]; then
	printf "\n" >&2
	printf "USAGE: %s <destination-dir>\n" "$(basename ${0})" >&2
	printf "\n" >&2
	exit 1
fi

declare DEST_DIR="${1}"

if [[ ! -d "${DEST_DIR}" ]]; then
	mkdir -p "${DEST_DIR}"
fi

cd "${DEST_DIR}"

declare -a FILE_BASENAMES=(
	'train-images-idx3-ubyte.gz'
	'train-labels-idx1-ubyte.gz'
	't10k-images-idx3-ubyte.gz'
	't10k-labels-idx1-ubyte.gz'
	)

declare URL_PREFIX="https://storage.googleapis.com/cvdf-datasets/mnist/"

declare FILE_BASENAME
for FILE_BASENAME in "${FILE_BASENAMES[@]}"; do
	declare WHOLE_URL="${URL_PREFIX}${FILE_BASENAME}"
	declare LOCAL_FILE="$(pwd)/${FILE_BASENAME}"

	printf "\n"
	printf "SOURCE URL: %s\n" "${WHOLE_URL}"
	printf "LOCAL FILE: %s\n" "${LOCAL_FILE}"
	printf "\n"

	wget -c "${WHOLE_URL}"

	if [[ -f "${LOCAL_FILE}" ]]; then
		printf "\n"
		printf "DOWNLOAD SUCCESSFUL\n"
		printf "\n"
	else
		printf "\n" >&2
		printf "ERROR: DOWNLOAD FAILED\n" >&2
		printf "\n" >&2
		exit 1
	fi
done

