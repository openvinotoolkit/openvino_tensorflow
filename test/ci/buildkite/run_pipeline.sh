#!/bin/bash

set -euo pipefail

echo "BUILDKITE_PULL_REQUEST_REPO: " $BUILDKITE_PULL_REQUEST_REPO
echo "BUILDKITE_REPO: $BUILDKITE_REPO"

if [[ -n \"${BUILDKITE_PULL_REQUEST_REPO##*//}\" && \"${BUILDKITE_REPO##*//}\" != \"${BUILDKITE_PULL_REQUEST_REPO##*//}\" ]]; then
   echo "External Commit"
   export BUILDKITE_CLEAN_CHECKOUT=true
   export BUILDKITE_NO_LOCAL_HOOKS=true
   cat header.yaml  block_build.yaml ngtf-cpu_centos.yaml | buildkite-agent pipeline upload
else
   echo "Internal Commit"
   cat header.yaml  ngtf-cpu_centos.yaml | buildkite-agent pipeline upload
fi

exit 0
~                                                                                                                       


