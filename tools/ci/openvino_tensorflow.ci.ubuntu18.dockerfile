# Copyright (C) 2019-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Defaults to py3.8, can be changed with --build-arg PY_VERSION
ARG PY_VERSION="python3.8"

################################################################################
FROM tensorflow/build:latest-${PY_VERSION} as tensorflow_build
################################################################################

# Defaults to v2.8.0, can be changed with --build-arg TF_TAG
ARG TF_TAG="v2.8.0"

SHELL ["/bin/bash", "-xo", "pipefail", "-c"]

RUN mkdir -p /tf/tensorflow /tf/pkg/

WORKDIR /tf/tensorflow/

RUN git clone https://github.com/tensorflow/tensorflow /tf/tensorflow/; \
    git checkout ${TF_TAG}

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

RUN bazel --bazelrc=/usertools/cpu.bazelrc build \
    --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=1" \
    --crosstool_top=${CROSSTOOL_TOP} \
    --config=sigbuild_local_cache //tensorflow:libtensorflow_cc.so.2 //tensorflow/core/kernels:ops_testutil;

RUN mkdir -p /tf/pkg/artifacts/tensorflow/; \
    cp -L /tf/tensorflow/bazel-bin/tensorflow/{libtensorflow_cc.so.2,libtensorflow_framework.so.2} /tf/pkg/artifacts/tensorflow/; \
    git clone /tf/tensorflow/ /tf/pkg/tensorflow/ --branch ${TF_TAG}; \
    cp --parents bazel-bin/tensorflow/libtensorflow_cc.so.2 /tf/pkg/tensorflow/; \
    cp --parents bazel-bin/tensorflow/core/kernels/{libtfkernel_sobol_op.so,libops_testutil.so} /tf/pkg/tensorflow/; \
    mv /tf/pkg/*whl /tf/pkg/artifacts/tensorflow/; \
    rm /tf/pkg/audit.txt /tf/pkg/profile.json

CMD ["/bin/bash"]

################################################################################
FROM openvino/ubuntu18_dev:2022.1 AS ovtf_build
################################################################################

LABEL description="This is the runtime image for OpenVINO™ integration with TensorFlow on Ubuntu 20.04 LTS"
LABEL vendor="Intel Corporation"

USER root

ENV DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-xo", "pipefail", "-c"]

RUN apt-get update; \
    apt-get install -y --no-install-recommends \
    git wget build-essential clang-format-3.9 \
    python3.8 python3.8-venv python3-pip openjdk-8-jdk; \
    rm -rf /var/lib/apt/lists/*;

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 70; \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 70;

RUN pip3 install --no-cache-dir setuptools>=54.1.2 psutil>=5.8.0 wheel==0.36.2

COPY --from=tensorflow_build /tf/pkg/ /home/openvino/tensorflow_pkg/

RUN curl -LsS https://aka.ms/InstallAzureCLIDeb | bash \
  && rm -rf /var/lib/apt/lists/*

# Set up Azure services
ENV AGENT_VERSION=2.200.2

WORKDIR /home/openvino/ci_setup/

RUN AZP_AGENTPACKAGE_URL=https://vstsagentpackage.azureedge.net/agent/${AGENT_VERSION}/vsts-agent-linux-${TARGETARCH}-${AGENT_VERSION}.tar.gz; \
    curl -LsS "$AZP_AGENTPACKAGE_URL" | tar -xz

COPY ./start.sh .
RUN chmod +x start.sh

RUN chown openvino -R /home/openvino
USER openvino

ENTRYPOINT [ "./start.sh" ]
