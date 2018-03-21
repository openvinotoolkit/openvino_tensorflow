#!/bin/bash
set -u
set -e

echo "****************************************************************************************"
echo "Run TensorFlow bridge <----> NGraph-C++ Pre-Merge CI Tests..."
echo "****************************************************************************************"

declare THIS_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

TF_ROOT="$THIS_SCRIPT_DIR"/../../../ngraph-tensorflow
if [ ! -e $TF_ROOT ]; then
    echo "TensorFlow installation directory not found: " $TF_ROOT
    exit 1
fi

# Copy the scripts to the TensorFlow directory
cp run-unit-tests.sh $TF_ROOT
if [ ! -e $TF_ROOT/run-unit-tests.sh ]; then
    echo "Cannot copy scripts to TensorFlow installation directory: " $TF_ROOT/run-unit-tests.sh
    exit 1
fi

pushd "$TF_ROOT"

#===================================================================================================
# Run the test...
#===================================================================================================

echo "Running TensorFlow unit tests using CPU backend"
XLA_NGRAPH_BACKEND=CPU ./run-unit-tests.sh

echo "Running TensorFlow unit tests using INTERPRETER backend"
XLA_NGRAPH_BACKEND=INTERPRETER ./run-unit-tests.sh

popd

# Run the test using NGRAPH INTERPRETER backend
XLA_NGRAPH_BACKEND=INTERPRETER python mnist-compare-final-accuracy.py

# Now run the test using NGRAPH CPU backend
# We are setting the OMP_NUM_THREADS to 44 as that matches the 
# number of cores for the CI system.
OMP_NUM_THREADS=44 KMP_AFFINITY=granularity=fine,scatter XLA_NGRAPH_BACKEND=CPU \
 python mnist-compare-final-accuracy.py

