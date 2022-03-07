# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

ARG TF_VERSION="2.8.0"

################################################################################
FROM openvino/ubuntu20_dev:2022.1.0 AS build_ovtf
################################################################################

LABEL description="This is the TF Serving runtime image for OpenVINO™ integration with TensorFlow on Ubuntu 20.04 LTS"
LABEL vendor="Intel Corporation"

USER root

ENV DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-xo", "pipefail", "-c"]

ARG TF_VERSION="v2.8.0"
ARG OPENVINO_VERSION="2022.1.0"
ARG OVTF_BRANCH="releases/2.0.0"

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

# Build OpenVINO™ integration with TensorFlow
RUN python3 build_ovtf.py \
    --tf_version=${TF_VERSION} \
    --openvino_version=${OPENVINO_VERSION} \
    --use_openvino_from_location=${INTEL_OPENVINO_DIR} \
    --cxx11_abi_version=1 \
    --disable_packaging_openvino_libs \
    --resource_usage_ratio=1.0;

# Run Unit Tests
RUN source build_cmake/venv-tf-py3/bin/activate && \
    source ${INTEL_OPENVINO_DIR}/setupvars.sh && \
    python3 test_ovtf.py

CMD ["/bin/bash"]

#######################################################################################
FROM tensorflow/serving:${TF_VERSION}-devel as build_serving
#######################################################################################

LABEL description="This is the TF Serving runtime image for OpenVINO™ integration with TensorFlow on Ubuntu 20.04 LTS"
LABEL vendor="Intel Corporation"

USER root

ENV DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-xo", "pipefail", "-c"]

COPY --from=build_ovtf /opt/intel/openvino_tensorflow/build_cmake/artifacts/lib/ /usr/local/lib/openvino_tensorflow/

# Download TF Serving sources (optionally at specific commit).
WORKDIR /tensorflow-serving

COPY serving_ovtf.patch /
RUN patch -p1 < /serving_ovtf.patch

# Build, and install TensorFlow Serving
ARG TF_SERVING_BUILD_OPTIONS="--config=release"
RUN echo "Building with build options: ${TF_SERVING_BUILD_OPTIONS}"
ARG TF_SERVING_BAZEL_OPTIONS=""
RUN echo "Building with Bazel options: ${TF_SERVING_BAZEL_OPTIONS}"

# Replace ABI flag in .bazelrc
RUN awk '{sub(/D_GLIBCXX_USE_CXX11_ABI=0/,"D_GLIBCXX_USE_CXX11_ABI=1")}1' .bazelrc > .temprc && mv .temprc .bazelrc

RUN bazel build --color=yes --curses=yes \
    ${TF_SERVING_BAZEL_OPTIONS} \
    --verbose_failures \
    --output_filter=DONT_MATCH_ANYTHING \
    ${TF_SERVING_BUILD_OPTIONS} \
    tensorflow_serving/model_servers:tensorflow_model_server && \
    cp bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server \
    /usr/local/bin/

# Clean up Bazel cache when done.
RUN bazel clean --expunge --color=yes && \
    rm -rf /root/.cache

################################################################################
FROM openvino/openvino_tensorflow_ubuntu20_runtime:2.0.0 AS runtime_serving
################################################################################

LABEL description="This is the TF Serving runtime image for OpenVINO™ integration with TensorFlow on Ubuntu 20.04 LTS"
LABEL vendor="Intel Corporation"

USER root

ENV DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-xo", "pipefail", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install TF Serving pkg
COPY --from=build_serving /usr/local/bin/tensorflow_model_server /usr/bin/tensorflow_model_server

# Expose ports
# gRPC
EXPOSE 8500

# REST
EXPOSE 8501

# Set where models should be stored in the container
ENV MODEL_BASE_PATH=/models
RUN mkdir -p ${MODEL_BASE_PATH}

# The only required piece is the model name in order to differentiate endpoints
ENV MODEL_NAME=model

# Create a script that runs the model server so we can use environment variables
# while also passing in arguments from the docker command line
RUN printf '#!/bin/bash \n\
source /opt/intel/openvino/setupvars.sh \n\
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/python3.8/dist-packages/openvino_tensorflow:/usr/local/lib/python3.8/dist-packages/tensorflow/ \n\
tensorflow_model_server --port=8500 --rest_api_port=8501 \
--model_name=${MODEL_NAME} --model_base_path=${MODEL_BASE_PATH}/${MODEL_NAME} \
"$@"' > /usr/bin/tf_serving_entrypoint.sh \
&& chmod +x /usr/bin/tf_serving_entrypoint.sh

USER openvino

CMD ["/bin/bash", "/usr/bin/tf_serving_entrypoint.sh"]
