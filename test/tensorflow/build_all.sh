#!/bin/bash
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
git checkout v1.8.0
./configure
bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package ./tensorflow_pkg/
virtualenv .venv
. .venv/bin/activate
pip install ./tensorflow_pkg/*
bazel build -c opt //tensorflow:libtensorflow_cc.so
cd ..
# assuming ngraph-tf is already checked out
cd ngraph-tf
mkdir build
cd build
cmake ../
make
cp src/libngraph_device.so /usr/lib
ldconfig
cd ../../
