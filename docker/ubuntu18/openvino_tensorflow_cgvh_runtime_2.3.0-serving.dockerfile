# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

ARG TF_SERVING_VERSION="2.9.2"
ARG OVTF_VERSION="2.3.0"


#######################################################################################
FROM openvino/openvino_tensorflow_ubuntu18_runtime:${OVTF_VERSION} as ovtf_runtime
#######################################################################################

#######################################################################################
FROM tensorflow/serving:${TF_SERVING_VERSION}-devel as build_serving
#######################################################################################

LABEL description="This is the TF Serving runtime image for OpenVINO™ integration with TensorFlow on Ubuntu 18.04 LTS"
LABEL vendor="Intel Corporation"

USER root

ENV DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-xo", "pipefail", "-c"]

COPY --from=ovtf_runtime /usr/local/lib/python3.8/dist-packages/openvino_tensorflow/ /usr/local/lib/openvino_tensorflow/

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

# Use C++14 to build TensorFlow Serving
RUN awk '{sub(/c\+\+17/,"c++14")}1' .bazelrc > .temprc && mv .temprc .bazelrc

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
FROM ovtf_runtime AS ovtf_serving_runtime
################################################################################

LABEL description="This is the TF Serving runtime image for OpenVINO™ integration with TensorFlow on Ubuntu 18.04 LTS"
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
