# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

################################################################################
FROM tensorflow/build:latest-python3.8 AS tensorflow_build
################################################################################

LABEL description="This is the dev image for OpenVINO™ integration with TensorFlow on Ubuntu 20.04 LTS"
LABEL vendor="Intel Corporation"

ARG TENSORFLOW_TAG="v2.7.0"
ARG OPENVINO_TAG="2021.4.2"

RUN mkdir -p /tf/tensorflow tf/cache /tf/pkg/artifacts/tensorflow/

WORKDIR /tf/tensorflow/

RUN git clone https://github.com/tensorflow/tensorflow /tf/tensorflow/; \
    git checkout ${TENSORFLOW_TAG}

# Build pip wheel
# ubuntu18.04-gcc7_manylinux2010-cuda11.2-cudnn8.1-tensorrt7.2_config_cuda//crosstool:toolchain is the release toolchain for CPU as well
# https://github.com/tensorflow/tensorflow/blob/07606166fdec2b21d645322b5465d13809bf06de/.bazelrc#L442
ARG CROSSTOOL_TOP="@ubuntu18.04-gcc7_manylinux2010-cuda11.2-cudnn8.1-tensorrt7.2_config_cuda//crosstool:toolchain"

RUN bazel --bazelrc=/usertools/cpu.bazelrc build \
    --crosstool_top=${CROSSTOOL_TOP} \
    --config=sigbuild_local_cache //tensorflow/tools/pip_package:build_pip_package

RUN ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tf/pkg/ --cpu; \
    /usertools/rename_and_verify_wheels.sh

# Build libtensorflow_cc
RUN bazel --bazelrc=/usertools/cpu.bazelrc build \
    --crosstool_top=${CROSSTOOL_TOP} \
    --config=sigbuild_local_cache //tensorflow:libtensorflow_cc.so.2 //tensorflow/core/kernels:ops_testutil

SHELL ["/bin/bash", "-c"]

# Prepare artifacts for OVTF build
RUN cp -L /tf/tensorflow/bazel-bin/tensorflow/{libtensorflow_cc.so.2,libtensorflow_framework.so.2} /tf/pkg/artifacts/tensorflow/; \
    git clone /tf/tensorflow/ /tf/pkg/tensorflow/ --branch ${TENSORFLOW_TAG}; \
    cp --parents bazel-bin/tensorflow/libtensorflow_cc.so.2 /tf/pkg/tensorflow/; \
    cp --parents bazel-bin/tensorflow/core/kernels/{libtfkernel_sobol_op.so,libops_testutil.so} /tf/pkg/tensorflow/; \
    mv /tf/pkg/*whl /tf/pkg/artifacts/tensorflow/; \
    rm /tf/pkg/audit.txt /tf/pkg/profile.json

CMD ["/bin/bash"]

################################################################################
FROM ubuntu:20.04 AS ovtf_build
################################################################################

LABEL description="This is the dev image for OpenVINO™ integration with TensorFlow on Ubuntu 20.04 LTS"
LABEL vendor="Intel Corporation"

ENV DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-c"]

# Copy tensorflow wheel and libs from the previous stage
COPY --from=tensorflow_build /tf/pkg/ /tensorflow_pkg/

ARG OVTF_BRANCH="master"

RUN apt-get update; \
    apt-get install -y --no-install-recommends \
    git wget build-essential\
    python3.8 python3.8-venv python3-pip \
    libusb-1.0-0-dev \
    gcc-7 g++-7; \
    rm -rf /var/lib/apt/lists/*

# Set defaults to gxx-7 and python3.8
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 70; \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 70; \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.8 70; \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 70;

## Installing CMake 3.18.4
RUN wget https://github.com/Kitware/CMake/releases/download/v3.18.4/cmake-3.18.4-Linux-x86_64.tar.gz && \
    tar -xzvf cmake-3.18.4-Linux-x86_64.tar.gz && \
    cp cmake-3.18.4-Linux-x86_64/bin/* /usr/local/bin/ && \
    cp -r cmake-3.18.4-Linux-x86_64/share/cmake-3.18 /usr/local/share/ && \
    rm -rf cmake-3.18.4-Linux-x86_64 cmake-3.18.4-Linux-x86_64.tar.gz

WORKDIR /opt/intel/

RUN git clone https://github.com/openvinotoolkit/openvino_tensorflow.git && \
    cd openvino_tensorflow && \
    git checkout ${OVTF_BRANCH} && \
    git submodule init && \
    git submodule update --recursive

WORKDIR /opt/intel/openvino_tensorflow/

RUN python3 -m pip install --upgrade pip pytest; \
    python3 -m pip install --no-cache-dir -r requirements.txt

RUN python3 build_ovtf.py --use_tensorflow_from_location=/tensorflow_pkg/

RUN source build_cmake/venv-tf-py3/bin/activate; \
    OPENVINO_TF_BACKEND=CPU python3 test_ovtf.py;

CMD ["/bin/bash"]

################################################################################
FROM ubuntu:20.04 AS openvino_tensorflow
################################################################################

USER root
WORKDIR /

SHELL ["/bin/bash", "-xo", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive

ARG OVTF_BRANCH="master"

RUN apt-get update; \
    apt-get install -y --no-install-recommends \
    git python3.8 python3.8-venv python3-pip \
    wget libgtk-3-0 libgl1 libsm6; \
    rm -rf /var/lib/apt/lists/*

COPY --from=ovtf_build /tensorflow_pkg/artifacts/tensorflow/libtensorflow_cc.so.2 /bazel-bin/tensorflow/
COPY --from=ovtf_build /tensorflow_pkg/artifacts/tensorflow/*whl /
COPY --from=ovtf_build /opt/intel/openvino_tensorflow/build_cmake/artifacts/*whl /

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 70; \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 70;

RUN git clone https://github.com/openvinotoolkit/openvino_tensorflow.git && \
    cd openvino_tensorflow && \
    git checkout ${OVTF_BRANCH} && \
    git submodule init && \
    git submodule update --recursive

WORKDIR /openvino_tensorflow/

RUN python3 -m pip install --no-cache-dir --upgrade pip jupyter; \
    python3 -m pip install --no-cache-dir /*whl; \
    python3 -m pip install --no-cache-dir -r examples/requirements.txt; \
    rm /*whl;

WORKDIR /openvino_tensorflow/examples/notebooks/

## Creating a shell script file which will be executed at the end to activate the environment and run jupyter notebook and
## Granting execution permission to shell script

RUN echo -e "jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root" | \
        tee /run-jupyter.sh && chmod +x /run-jupyter.sh

## .run-jupyter.sh file will be executed when the container starts

CMD ["/bin/bash", "/run-jupyter.sh"]