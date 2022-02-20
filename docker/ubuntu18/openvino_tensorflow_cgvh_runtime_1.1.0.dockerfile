# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

################################################################################
FROM tensorflow/build:latest-python3.8 AS tensorflow_build
################################################################################

LABEL description="This is the runtime image for OpenVINO™ integration with TensorFlow on Ubuntu 20.04 LTS"
LABEL vendor="Intel Corporation"

ARG TF_VERSION="v2.7.0"

SHELL ["/bin/bash", "-xo", "pipefail", "-c"]

RUN mkdir -p /tf/tensorflow /tf/pkg/artifacts/tensorflow/

WORKDIR /tf/tensorflow/

RUN git clone https://github.com/tensorflow/tensorflow /tf/tensorflow/; \
    git checkout ${TF_VERSION}

# Build pip wheel
# ubuntu18.04-gcc7_manylinux2010-cuda11.2-cudnn8.1-tensorrt7.2_config_cuda//crosstool:toolchain is the release toolchain for CPU as well
# https://github.com/tensorflow/tensorflow/blob/07606166fdec2b21d645322b5465d13809bf06de/.bazelrc#L442
ARG CROSSTOOL_TOP="@ubuntu18.04-gcc7_manylinux2010-cuda11.2-cudnn8.1-tensorrt7.2_config_cuda//crosstool:toolchain"

RUN bazel --bazelrc=/usertools/cpu.bazelrc build \
    --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=1" \
    --crosstool_top=${CROSSTOOL_TOP} \
    --config=sigbuild_local_cache //tensorflow/tools/pip_package:build_pip_package

RUN ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tf/pkg/ --cpu; \
    /usertools/rename_and_verify_wheels.sh

# Build libtensorflow_cc
RUN bazel --bazelrc=/usertools/cpu.bazelrc build \
    --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=1" \
    --crosstool_top=${CROSSTOOL_TOP} \
    --config=sigbuild_local_cache //tensorflow:libtensorflow_cc.so.2 //tensorflow/core/kernels:ops_testutil

# Prepare artifacts for OVTF build
RUN cp -L /tf/tensorflow/bazel-bin/tensorflow/{libtensorflow_cc.so.2,libtensorflow_framework.so.2} /tf/pkg/artifacts/tensorflow/; \
    git clone /tf/tensorflow/ /tf/pkg/tensorflow/ --branch ${TF_VERSION}; \
    cp --parents bazel-bin/tensorflow/libtensorflow_cc.so.2 /tf/pkg/tensorflow/; \
    cp --parents bazel-bin/tensorflow/core/kernels/{libtfkernel_sobol_op.so,libops_testutil.so} /tf/pkg/tensorflow/; \
    mv /tf/pkg/*whl /tf/pkg/artifacts/tensorflow/; \
    rm /tf/pkg/audit.txt /tf/pkg/profile.json

CMD ["/bin/bash"]

################################################################################
FROM openvino/ubuntu18_dev:2021.4.2 AS ovtf_build
################################################################################

# Stage 1 builds OpenVINO™ integration with TensorFlow from source, prepares wheel and other shared libs for use by the final image

LABEL description="This is the runtime image for OpenVINO™ integration with TensorFlow on Ubuntu 20.04 LTS"
LABEL vendor="Intel Corporation"

USER root

ENV DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-xo", "pipefail", "-c"]

ARG TF_VERSION="v2.7.0"
ARG OPENVINO_VERSION="2021.4.2"
ARG OVTF_BRANCH="master"

RUN apt-get update; \
    apt-get install -y --no-install-recommends \
    git wget build-essential \
    python3.8 python3.8-venv python3-pip; \
    rm -rf /var/lib/apt/lists/*;

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 70; \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 70;

WORKDIR /opt/intel/

RUN git clone https://github.com/openvinotoolkit/openvino_tensorflow.git && \
    cd openvino_tensorflow && \
    git checkout ${OVTF_BRANCH} && \
    git submodule init && \
    git submodule update --recursive

WORKDIR /opt/intel/openvino_tensorflow/

RUN python3 -m pip install --upgrade pip pytest; \
    python3 -m pip install --no-cache-dir -r requirements.txt

ENV TENSORFLOW_PKG_DIR /tf/tensorflow_pkg
ENV INTEL_OPENVINO_DIR /opt/intel/openvino
ENV OVTF_ARTIFACTS_DIR /opt/intel/openvino_tensorflow/build_artifacts/

COPY --from=tensorflow_build /tf/pkg/ ${TENSORFLOW_PKG_DIR}

# Build OpenVINO™ integration with TensorFlow
RUN python3 build_ovtf.py \
    --tf_version=${TF_VERSION} \
    --openvino_version=${OPENVINO_VERSION} \
    --use_tensorflow_from_location=${TENSORFLOW_PKG_DIR} \
    --use_openvino_from_location=${INTEL_OPENVINO_DIR} \
    --cxx11_abi_version=1 \
    --disable_packaging_openvino_libs \
    --resource_usage_ratio=1.0;

# Run Unit Tests
RUN source build_cmake/venv-tf-py3/bin/activate && \
    source ${INTEL_OPENVINO_DIR}/bin/setupvars.sh && \
    python3 test_ovtf.py

RUN cp -r build_cmake/artifacts/ build_artifacts && rm -rf build_cmake

CMD ["/bin/bash"]

################################################################################
FROM openvino/ubuntu18_runtime:2021.4.2_src AS ovtf_runtime
################################################################################

LABEL description="This is the runtime image for OpenVINO™ integration with TensorFlow on Ubuntu 20.04 LTS"
LABEL vendor="Intel Corporation"

USER root

ENV DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-xo", "pipefail", "-c"]

RUN apt-get update; \
    apt-get install -y --no-install-recommends \
    git wget \
    python3.8 python3.8-venv python3-pip; \
    rm -rf /var/lib/apt/lists/*;

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 70; \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 70;

ENV INTEL_OPENVINO_DIR /opt/intel/openvino

COPY --from=ovtf_build /opt/intel/openvino_tensorflow/ /opt/intel/openvino_tensorflow/

WORKDIR /opt/intel/openvino_tensorflow/

RUN python3 -m pip install --upgrade pip; \
    python3 -m pip install --no-cache-dir build_artifacts/openvino_tensorflow*whl build_artifacts/tensorflow/tensorflow*whl; \
    python3 -m pip install --no-cache-dir -r examples/requirements.txt; \
    python3 -m pip install --upgrade numpy jupyter;

WORKDIR /opt/intel/openvino_tensorflow/examples/notebooks/

## Creating a shell script file which will be executed at the end to activate the environment and run jupyter notebook and
## Granting execution permission to shell script

RUN echo -e "source \${INTEL_OPENVINO_DIR}/bin/setupvars.sh\njupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root" | \
        tee /home/openvino/run-jupyter.sh && chmod +x /home/openvino/run-jupyter.sh

USER openvino

## .run-jupyter.sh file will be executed when the container starts
CMD ["/bin/bash", "/home/openvino/run-jupyter.sh"]