#!/bin/bash
sudo apt-get update
sudo apt-get install g++5 cmake make -y
sudo apt-get install python-numpy python-dev python-pip python-wheel -y
sudo apt-get install pkg-config zip g++ zlib1g-dev unzip python virtualenv -y
sudo apt-get install wget -y
sudo apt-get install git -y
get https://github.com/bazelbuild/bazel/releases/download/0.14.0/bazel-0.14.0-installer-linux-x86_64.sh
chmod a+x bazel-0.14.0-installer-linux-x86_64.sh
./bazel-0.14.0-installer-linux-x86_64.sh --user 

