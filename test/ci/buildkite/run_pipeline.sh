#!/bin/bash

set -euo pipefail

echo "BUILDKITE_AGENT_META_DATA_QUEUE: ${BUILDKITE_AGENT_META_DATA_QUEUE}"
echo "BUILDKITE_AGENT_META_DATA_NAME: ${BUILDKITE_AGENT_META_DATA_NAME}"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Always run setup for now
PIPELINE_STEPS=" ${SCRIPT_DIR}/setup.yml "
if [ "${BUILDKITE_PIPELINE_NAME}" == "ngtf-cpu-ubuntu-grappler" ]; then
   export BUILD_OPTIONS=--use_grappler
   PIPELINE_STEPS+=" ${SCRIPT_DIR}/ngtf-cpu-ubuntu.yml "
elif [ "${BUILDKITE_PIPELINE_NAME}" == "ngtf-interpreter-ubuntu" ]; then
   export NGRAPH_TF_BACKEND=INTERPRETER
   PIPELINE_STEPS+=" ${SCRIPT_DIR}/ngtf-cpu-ubuntu.yml "
else
   PIPELINE_STEPS+=" ${SCRIPT_DIR}/${BUILDKITE_PIPELINE_NAME}.yml "
fi

cat ${PIPELINE_STEPS} | buildkite-agent pipeline upload