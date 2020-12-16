#!/bin/bash
PWD=`pwd`
ARTIFACTS_DIR=$1
if [ "${ARTIFACTS_DIR}" == "" ]; then
    echo "Error: artifacts dir not specified!" && exit 1
fi

if [ ! -d "${ARTIFACTS_DIR}/examples" ]; then echo "Cannot find dir ${ARTIFACTS_DIR}/examples !"; exit 1; fi
if [ ! -d "${ARTIFACTS_DIR}/lib" ]; then echo "Cannot find dir ${ARTIFACTS_DIR}/lib !"; exit 1; fi

# Assumed pwd: /localdisk/buildkite/artifacts/$BUILDKITE_BUILD_ID/examples
# or for local run .../build_cmake/artifacts/examples

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${ARTIFACTS_DIR}/lib:${ARTIFACTS_DIR}/tensorflow"
echo "PWD=`pwd`, LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
if [ "${BUILDKITE}" == "true" ]; then
    LOCALSTORE_PREFIX=/localdisk/buildkite/artifacts
else
    LOCALSTORE_PREFIX=/tmp
fi
LOCALSTORE=${LOCALSTORE_PREFIX}/pretrained_models
if [ ! -d "${LOCALSTORE}" ]; then mkdir -p ${LOCALSTORE}; fi

function get_artifacts {
    TYPE=$1
    LINK=$2
    if [ ! -f "${TYPE}" ]; then
        if [ ! -f "${LOCALSTORE}/${TYPE}" ]; then
            # download
            echo "Downloading ${TYPE} ..."
            if [[ ${LINK} =~ tar.gz$ ]]; then
                targzfile="${LOCALSTORE}/${TYPE}.tar.gz"
                wget ${LINK} -O "${targzfile}" && tar xvf "${targzfile}" -C "${LOCALSTORE}" && rm "${targzfile}"
            else
                wget ${LINK} -O "${LOCALSTORE}/${TYPE}"
            fi
            # check if successful...
            if [ ! -f "${LOCALSTORE}/${TYPE}" ]; then echo "Failed!"; exit 1; fi
        fi
        if [ -L "${TYPE}" ]; then unlink ${TYPE}; fi
        ln -s ${LOCALSTORE}/${TYPE} ${TYPE}
    fi
}

cd ${ARTIFACTS_DIR}/examples
MODEL=inception_v3_2016_08_28_frozen.pb
get_artifacts ${MODEL} "https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz"
IMAGE=grace_hopper.jpg
get_artifacts ${IMAGE} "https://github.com/tensorflow/tensorflow/raw/master/tensorflow/examples/label_image/data/grace_hopper.jpg"
LABELS=imagenet_slim_labels.txt
get_artifacts ${LABELS} "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt"

./cpp/inference/infer_single_network --graph=${MODEL} \
    --labels=${LABELS} \
    --image=${IMAGE} \
    --input_width=299 --input_height=299 \
    --input_layer="input" --output_layer="InceptionV3/Predictions/Reshape_1" 2>&1 | tee tmp_output

echo
echo "Checking inference result..."
ret_code=1
grep -E 'military uniform \(653\): (0.8|0.9)' tmp_output && echo "TEST PASSED" && ret_code=0
rm tmp_output

exit $((ret_code))
