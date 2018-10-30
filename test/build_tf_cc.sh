#!/bin/bash

echo "TensorFlow Source directory 1: " $1
echo "_GLIBCXX_USE_CXX11_ABI: " $2
# If this script is not invoked with the TF source directory then return error
if [ $# -lt 2 ]
then
    echo "Usage: $0 <Tensorflow source directory> <CXX11_ABI=0/1>"
    exit -1
fi

TF_ROOT=$1
USE_CXX11_ABI=$2

if [ $2 -eq 0 ]
then
   COPT=--copt='-D_GLIBCXX_USE_CXX11_ABI=0'
else
   COPT=--copt='-D_GLIBCXX_USE_CXX11_ABI=1'
fi

if [ ! -e $TF_ROOT ]; then
    echo "TensorFlow installation directory not found: " $TF_ROOT
    exit 1
fi

# Set the python directory. Must be inside a virtual env
export PYTHON_BIN_PATH=$VIRTUAL_ENV/bin/python

# Declare the build options for TensorFlow 
export CC_OPT_FLAGS="-march=native"
export USE_DEFAULT_PYTHON_LIB_PATH=1

export TF_NEED_NGRAPH=0

export TF_NEED_IGNITE=0
export TF_NEED_ROCM=0

export TF_NEED_JEMALLOC=1
export TF_NEED_AWS=0
export TF_NEED_GCP=0
export TF_NEED_HDFS=0
export TF_NEED_KAFKA=0
export TF_ENABLE_XLA=0
export TF_NEED_GDR=0
export TF_NEED_VERBS=0
export TF_NEED_OPENCL_SYCL=0
export TF_NEED_COMPUTECPP=0
export TF_NEED_CUDA=0
export TF_NEED_OPENCL=0
export TF_NEED_MPI=0
export TF_DOWNLOAD_CLANG=0
export TF_SET_ANDROID_WORKSPACE=0

# Configure TF
pushd "${TF_ROOT}"
./configure

# Build the TensorFlow C++ Library 
bazel build --config=opt ${COPT} //tensorflow:libtensorflow_cc.so

popd

