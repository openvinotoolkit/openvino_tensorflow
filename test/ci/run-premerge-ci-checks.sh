#!/bin/bash
set -u
set -e

echo "**************************************************************************"
echo "Run source code formatting check"
echo "**************************************************************************"
../../maint/check-code-format.sh

echo "**************************************************************************"
echo "Run TensorFlow bridge <----> NGraph-C++ Pre-Merge CI Tests..."
echo "**************************************************************************"

declare THIS_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Relative to ngraph-tf/build-dir/test, where this script is run
declare BUILD_DIR="$( realpath .. )"  

if [ -z ${TF_ROOT+x} ]; then
    TF_ROOT="$THIS_SCRIPT_DIR"/../../../tensorflow
fi

if [ ! -e $TF_ROOT ]; then
    echo "TensorFlow installation directory not found: " $TF_ROOT
    exit 1
fi

# Wrapper to produce JUnit XML result output
declare JUNIT="${THIS_SCRIPT_DIR}/junit-wrap.sh"

# Define default dataset and trained model directories, if not already set
if [ -z ${NGRAPH_IMAGENET_DATASET+x} ]; then
    NGRAPH_IMAGENET_DATASET='/mnt/data/Imagenet_Validation/'
fi
if [ -z ${NGRAPH_TRAINED_MODEL+x} ]; then
    NGRAPH_TRAINED_MODEL='/nfs/fm/disks/aipg_trained_dataset/ngraph_tensorflow/fully_trained/resnet50'
fi


#===============================================================================
# Run the test...
#===============================================================================
echo "--------------------------------------------------------------------------"
echo "Running TensorFlow unit tests"
echo "--------------------------------------------------------------------------"

export GTEST_OUTPUT="xml:${BUILD_DIR}/xunit_gtest.xml"
./gtest_ngtf 

pushd python
# We need to explictly run python here, since "pytest" is also a shell script,
# and that shell script starts with "#! /usr/bin/python", overriding any
# python installed in a virtual environment.
python -m pytest \
    test_abs.py \
    test_cast.py \
    test_conv2dbackpropinput.py \
    test_resize_to_dynamic_shape.py \
    test_slice.py \
    test_sigmoidgrad.py \
    test_tanhgrad.py
popd

echo "--------------------------------------------------------------------------"
echo "Running test for installation of the ngraph module"
echo "--------------------------------------------------------------------------"
export JUNIT_WRAP_FILE="${BUILD_DIR}/junit_install_test.xml"
export JUNIT_WRAP_SUITE='installer'
export JUNIT_WRAP_TEST='install_test.py'
${JUNIT} python ../../test/install_test.py

echo "--------------------------------------------------------------------------"
echo "Running a quick inference test"
echo "--------------------------------------------------------------------------"

pushd ${BUILD_DIR}
rm -rf benchmarks
git clone https://github.com/tensorflow/benchmarks.git
pushd benchmarks/scripts/tf_cnn_benchmarks/
git checkout 4c7b09ad87bbfc4b1f89650bcee40b3fc5e7dfed
echo "import ngraph" >> convnet_builder.py
export JUNIT_WRAP_FILE="${BUILD_DIR}/junit_resnet50_imagenet_inference.xml"
export JUNIT_WRAP_SUITE='inference_validation'
export JUNIT_WRAP_TEST='tf_cnn_benchmarks_resnet50'
#${JUNIT} python tf_cnn_benchmarks.py --model=resnet50 --eval --num_inter_threads=1 \
#  --batch_size=16 --num_batches=50 \
#  --train_dir "${NGRAPH_TRAINED_MODEL}" \
#  --data_format NCHW \
#  --data_name=imagenet --data_dir "${NGRAPH_IMAGENET_DATASET}" --datasets_use_prefetch=False 

# Training test
${JUNIT} OMP_NUM_THREADS=28 KMP_AFFINITY=granularity=fine,compact,1,0 \
    python tf_cnn_benchmarks.py --data_format NCHW  --num_inter_threads=1 \
        --train_dir=./modelsavepath/ --num_batches 5 --model=resnet50 --batch_size=128
# Inference test
${JUNIT} OMP_NUM_THREADS=28 KMP_AFFINITY=granularity=fine,compact,1,0 \
    python tf_cnn_benchmarks.py --data_format NCHW --num_inter_threads 1 \
        --train_dir=$(pwd)/modelsavepath --eval --model=resnet50 --batch_size=128
popd

