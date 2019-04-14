#!/bin/bash

set -euo pipefail

echo "BUILDKITE_PULL_REQUEST_REPO: " $BUILDKITE_PULL_REQUEST_REPO
echo "BUILDKITE_REPO: $BUILDKITE_REPO"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if [[ $BUILDKITE_PULL_REQUEST = 'false' ]]; then
   echo "Not a pull request"
   cat $SCRIPT_DIR/header.yaml  $SCRIPT_DIR/ngtf-cpu_centos.yaml | buildkite-agent pipeline upload
else
if [[ -n \"${BUILDKITE_PULL_REQUEST_REPO##*//}\" && \"${BUILDKITE_REPO##*//}\" != \"${BUILDKITE_PULL_REQUEST_REPO##*//}\" ]]; then
   echo "External Commit"
   export BUILDKITE_CLEAN_CHECKOUT=true
   export BUILDKITE_NO_LOCAL_HOOKS=true
   cat $SCRIPT_DIR/header.yaml  $SCRIPT_DIR/block_build.yaml $SCRIPT_DIR/ngtf-cpu_centos.yaml | buildkite-agent pipeline upload
else
   echo "Internal Commit"
   cat $SCRIPT_DIR/header.yaml  $SCRIPT_DIR/ngtf-cpu_centos.yaml | buildkite-agent pipeline upload
fi
fi

