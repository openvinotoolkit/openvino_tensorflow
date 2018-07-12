#!/bin/bash
set -u
set -e

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

#===============================================================================
# Run the test...
#===============================================================================
echo "Running TensorFlow unit tests"
./gtest_ngtf

pushd python
# We need to explictly run python here, since "pytest" is also a shell script,
# and that shell script starts with "#! /usr/bin/python", overriding any
# python installed in a virtual environment.
python -m pytest
popd

# Temporarily disabled, as per discussion in standup with Avijit, since the
# dataset is only available on NFS, which won't work in Docker containers or
# when testing in the cloud.
echo "Inference test disabled for now -- see comments in script."
# echo "Running a quick inference test"
# pushd ../../examples/resnet
# python tf_cnn_benchmarks.py --model=resnet50 --eval --num_inter_threads=1 \
#   --batch_size=128 --num_batches=50 \
#   --train_dir /nfs/fm/disks/aipg_trained_dataset/ngraph_tensorflow/fully_trained/resnet50\
#   --data_format NCHW --select_device NGRAPH \
#   --data_name=imagenet --data_dir /mnt/data/TF_ImageNet_latest/ --datasets_use_prefetch=False
# popd

