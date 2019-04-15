#!/bin/bash

set -euo pipefail

echo "BUILDKITE_PULL_REQUEST_REPO: " $BUILDKITE_PULL_REQUEST_REPO
echo "BUILDKITE_REPO: $BUILDKITE_REPO"
echo "PIPELINE OS: $PIPELINE_QUEUE"

if [[ $PIPELINE_QUEUE = 'cpu' ]]; then
   TF_PY_WHEEL=tensorflow-1.13.1-cp35-cp35m-linux_x86_64.whl
   # For the time being - hardcode the file
   # Eventually we will replace the queue and other variables during the pipeline creation
   STEPS_FILE=ngtf-cpu_ubuntu.yaml
elif [[ $PIPELINE_QUEUE = 'cpu-centos' ]]; then
   TF_PY_WHEEL=tensorflow-1.13.1-cp36-cp36m-linux_x86_64.whl
   STEPS_FILE=ngtf-cpu_centos.yaml
else
   echo "Unknown PILELINE_QUEUE: $PIPELINE_QUEUE"
   exit -1
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if [[ $BUILDKITE_PULL_REQUEST = 'false' ]]; then
   echo "Not a pull request"
   cat $SCRIPT_DIR/header.yaml  $SCRIPT_DIR/$STEPS_FILE | buildkite-agent pipeline upload
else
if [[ -n \"${BUILDKITE_PULL_REQUEST_REPO##*//}\" && \"${BUILDKITE_REPO##*//}\" != \"${BUILDKITE_PULL_REQUEST_REPO##*//}\" ]]; then
   echo "External Commit"
   export BUILDKITE_CLEAN_CHECKOUT=true
   export BUILDKITE_NO_LOCAL_HOOKS=true
   cat $SCRIPT_DIR/header.yaml  $SCRIPT_DIR/block_build.yaml $SCRIPT_DIR/$STEPS_FILE | buildkite-agent pipeline upload
else
   echo "Internal Commit"
   cat $SCRIPT_DIR/header.yaml  $SCRIPT_DIR/$STEPS_FILE | buildkite-agent pipeline upload
fi
fi

