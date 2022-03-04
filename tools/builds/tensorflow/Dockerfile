# Copyright (C) 2019-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

FROM gcr.io/tensorflow-testing/nosla-cuda10.1-cudnn7-ubuntu16.04-manylinux2010-multipython

USER root
WORKDIR /
SHELL ["/bin/bash", "-xo", "pipefail", "-c"]

ENV IS_NIGHTLY=0
ENV TF_NEED_CUDA=0
ENV TF_NEED_ROCM=0

RUN git clone https://github.com/tensorflow/tensorflow/ /tensorflow/

WORKDIR /tensorflow/

RUN git checkout v2.8.0

COPY abi_flag.patch /tensorflow/

RUN git apply abi_flag.patch

RUN cd "/usr/local/lib/bazel/bin" && curl -fLO https://releases.bazel.build/3.1.0/release/bazel-3.1.0-linux-x86_64 && chmod +x bazel-3.1.0-linux-x86_64

RUN apt update && apt install patchelf

RUN bash tensorflow/tools/ci_build/rel/ubuntu/cpu_py37_pip.sh

RUN cp /tensorflow/pip_test/whl/* /

RUN bash tensorflow/tools/ci_build/rel/ubuntu/cpu_py38_pip.sh

RUN cp /tensorflow/pip_test/whl/* /

# Install Python3.9 as it doesn't already exist
RUN /install/build_and_install_python.sh "3.9.7"
RUN /install/install_pip_packages_by_version.sh "/usr/local/bin/pip3.9"

RUN bash tensorflow/tools/ci_build/rel/ubuntu/cpu_py39_pip.sh

RUN cp /tensorflow/pip_test/whl/* /
