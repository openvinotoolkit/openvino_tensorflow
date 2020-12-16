#!/bin/bash

set -euo pipefail

echo "BUILDKITE_AGENT_META_DATA_QUEUE: ${BUILDKITE_AGENT_META_DATA_QUEUE}"
echo "BUILDKITE_AGENT_META_DATA_NAME: ${BUILDKITE_AGENT_META_DATA_NAME}"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

export TF_LOCATION=/localdisk/buildkite-agent/prebuilt_tensorflow_2_2_0
export OV_LOCATION=/localdisk/buildkite-agent/prebuilt_openvino_2021_2/artifacts/openvino
export NGRAPH_TF_BACKEND=CPU

# Always run setup for now
PIPELINE_STEPS=" ${SCRIPT_DIR}/setup.yml "
if [ "${BUILDKITE_PIPELINE_NAME}" == "cpu-grappler" ]; then
   export BUILD_OPTIONS=--use_grappler
   PIPELINE_STEPS+=" ${SCRIPT_DIR}/cpu.yml "
elif [ "${BUILDKITE_PIPELINE_NAME}" == "cpu" ]; then
   PIPELINE_STEPS+=" ${SCRIPT_DIR}/cpu.yml "
elif [ "${BUILDKITE_PIPELINE_NAME}" == "cpu-intel-tf" ]; then
   export BUILD_OPTIONS=--use_intel_tensorflow
   export TF_LOCATION=/localdisk/buildkite-agent/prebuilt_intel_tensorflow
   PIPELINE_STEPS+=" ${SCRIPT_DIR}/cpu.yml "
elif [ "${BUILDKITE_PIPELINE_NAME}" == "gpu" ]; then
   export NGRAPH_TF_BACKEND=GPU
   PIPELINE_STEPS+=" ${SCRIPT_DIR}/cpu.yml "
elif [ "${BUILDKITE_PIPELINE_NAME}" == "models-gpu" ]; then
   export NGRAPH_TF_BACKEND=GPU
   PIPELINE_STEPS+=" ${SCRIPT_DIR}/models-cpu.yml "
elif [ "${BUILDKITE_PIPELINE_NAME}" == "myriad" ]; then
   export NGRAPH_TF_BACKEND=MYRIAD
   export NGRAPH_TF_UTEST_RTOL=0.0001
   PIPELINE_STEPS+=" ${SCRIPT_DIR}/cpu.yml "
elif [ "${BUILDKITE_PIPELINE_NAME}" == "models-myriad" ]; then
   export NGRAPH_TF_BACKEND=MYRIAD
   PIPELINE_STEPS+=" ${SCRIPT_DIR}/models-cpu.yml "
else
   PIPELINE_STEPS+=" ${SCRIPT_DIR}/${BUILDKITE_PIPELINE_NAME}.yml "
fi

cat ${PIPELINE_STEPS} | buildkite-agent pipeline upload
