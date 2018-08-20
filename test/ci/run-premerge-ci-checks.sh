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

if [ -z ${TF_ROOT+x} ]; then
    TF_ROOT="$THIS_SCRIPT_DIR"/../../../tensorflow
fi

if [ ! -e $TF_ROOT ]; then
    echo "TensorFlow installation directory not found: " $TF_ROOT
    exit 1
fi

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
./gtest_ngtf --gtest_filter="-tf_exec.Op_*:tf_exec.BiasAddGrad:tf_exec.FusedBatchNormGrad_NHWC:tf_exec.Tile:assign_clusters.cone"

####### Disabled tests for now #######
#pushd python
# We need to explictly run python here, since "pytest" is also a shell script,
# and that shell script starts with "#! /usr/bin/python", overriding any
# python installed in a virtual environment.
#python -m pytest
#popd
####### Disabled tests for now #######

echo "--------------------------------------------------------------------------"
echo "Running test for installation of the ngraph module"
echo "--------------------------------------------------------------------------"
python ../../test/install_test.py

echo "--------------------------------------------------------------------------"
echo "Running a quick inference test"
echo "--------------------------------------------------------------------------"

pushd ../../examples/resnet
python tf_cnn_benchmarks.py --model=resnet50 --eval --num_inter_threads=1 \
  --batch_size=16 --num_batches=50 \
  --train_dir "${NGRAPH_TRAINED_MODEL}" \
  --data_format NCHW --select_device NGRAPH \
  --data_name=imagenet --data_dir "${NGRAPH_IMAGENET_DATASET}" --datasets_use_prefetch=False
popd

