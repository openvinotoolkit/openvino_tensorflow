# Copyright (C) 2019-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

FROM gcr.io/tensorflow-testing/nosla-cuda10.1-cudnn7-ubuntu16.04-manylinux2010-multipython

ENV IS_NIGHTLY=0
ENV TF_NEED_CUDA=0
ENV TF_NEED_ROCM=0

RUN git clone https://github.com/tensorflow/tensorflow/ /tensorflow/

WORKDIR /tensorflow/

RUN git checkout r2.4

RUN cd "/usr/local/lib/bazel/bin" && curl -fLO https://releases.bazel.build/3.1.0/release/bazel-3.1.0-linux-x86_64 && chmod +x bazel-3.1.0-linux-x86_64

RUN apt update && apt install patchelf

RUN bash tensorflow/tools/ci_build/release/ubuntu_16/cpu_py36_full/pip.sh

RUN cp /tensorflow/pip_test/whl/* /

RUN bash tensorflow/tools/ci_build/release/ubuntu_16/cpu_py37_full/pip.sh

RUN cp /tensorflow/pip_test/whl/* /

RUN bash tensorflow/tools/ci_build/release/ubuntu_16/cpu_py38_full/pip.sh

RUN cp /tensorflow/pip_test/whl/* /
