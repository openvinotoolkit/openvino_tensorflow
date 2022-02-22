# Copyright (C) 2019-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

###############################################################################
FROM docker.io/tensorflow/serving:2.7.0-devel as base_build
###############################################################################

FROM base_build as binary_build

ARG OPENVINO_TENSORFLOW_VERSION="1.1.0"
ARG OPENVINO_VERSION="2021.4.2"

# Build, and install TensorFlow Serving
ARG TF_SERVING_BUILD_OPTIONS="--config=release --copt=-mavx  --cxxopt=-D_GLIBCXX_USE_CXX11_ABI=0"
RUN echo "Building with build options: ${TF_SERVING_BUILD_OPTIONS}"
ARG TF_SERVING_BAZEL_OPTIONS=""
RUN echo "Building with Bazel options: ${TF_SERVING_BAZEL_OPTIONS}"

WORKDIR /
RUN git clone https://github.com/openvinotoolkit/openvino_tensorflow.git && \
    cd openvino_tensorflow &&\
    git checkout ${OVTF_BRANCH} && \
    git submodule init && \
    git submodule update --recursive

# Build Tensorflow
WORKDIR /openvino_tensorflow
RUN pip install psutil
RUN mkdir /tensorflow_build && export PATH_TO_TF_BUILD=/tensorflow_build && \
    python build_tf.py --output_dir=${PATH_TO_TF_BUILD} --tf_version=v2.7.0 --cxx11_abi_version=0 

# Build OpenVINO integration with Tensorflow
RUN apt-get update && apt-get install -y libusb-1.0-0-dev
RUN wget https://github.com/Kitware/CMake/releases/download/v3.18.4/cmake-3.18.4-Linux-x86_64.tar.gz && \
    tar -xzvf cmake-3.18.4-Linux-x86_64.tar.gz && \
    cp cmake-3.18.4-Linux-x86_64/bin/* /usr/local/bin/ && \
    cp -r cmake-3.18.4-Linux-x86_64/share/cmake-3.18 /usr/local/share/

RUN python build_ovtf.py --use_tensorflow_from_location=/tensorflow_build --openvino_version=${OPENVINO_VERSION}

COPY serving_ovtf.patch /
WORKDIR /tensorflow-serving
RUN patch -p1 < /serving_ovtf.patch

# Build TF Model Server enabled with OpenVINO integration with Tensorflow
RUN bazel build --color=yes --curses=yes \
    ${TF_SERVING_BAZEL_OPTIONS} \
    --verbose_failures \
    --output_filter=DONT_MATCH_ANYTHING \
    ${TF_SERVING_BUILD_OPTIONS} \
    tensorflow_serving/model_servers:tensorflow_model_server && \
    cp bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server \
    /usr/local/bin/

#FROM binary_build as clean_build
# Clean up Bazel cache when done.
#RUN bazel clean --expunge --color=yes && \
#    rm -rf /root/.cache

###############################################################################
FROM docker.io/tensorflow/serving:2.7.0
###############################################################################
ARG OPENVINO_VERSION=2021.4.2
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.7 python3.7-dev python3-pip python3-setuptools wget && \
    rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 0

COPY --from=binary_build /openvino_tensorflow/build_cmake/artifacts/lib /openvino_tensorflow
RUN  python3 -m pip install --upgrade pip; python3 -m pip install --no-cache-dir tensorflow-cpu==2.7.0
COPY --from=binary_build /usr/local/bin/tensorflow_model_server /usr/bin/tensorflow_model_server

# GPU dependencies
WORKDIR /tmp/opencl
RUN useradd -ms /bin/bash -G video,users openvino && \
    chown openvino -R /home/openvino
RUN groupmod -g 44 video

# Please use `20.35.17767` for 10th generation Intel® Core™ processor (formerly Ice Lake) or 11th generation Intel® Core™ processor (formerly Tiger Lake)
ARG INTEL_OPENCL=19.41.14441

RUN wget https://raw.githubusercontent.com/openvinotoolkit/openvino/${OPENVINO_VERSION}/scripts/install_dependencies/install_NEO_OCL_driver.sh
RUN chmod +x install_NEO_OCL_driver.sh && apt-get update && apt-get install -y curl
RUN ./install_NEO_OCL_driver.sh --no_numa -y --install_driver ${INTEL_OPENCL} && \
    rm -rf /var/lib/apt/lists/*

ARG BUILD_DEPENDENCIES="autoconf \
                        automake \
                        build-essential \
                        libtool \
                        unzip \
                        udev"
RUN apt-get update && \
    apt-get install -y --no-install-recommends ${BUILD_DEPENDENCIES} && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN curl -L https://github.com/libusb/libusb/archive/v1.0.22.zip --output v1.0.22.zip && \
    unzip v1.0.22.zip

WORKDIR /opt/libusb-1.0.22
RUN ./bootstrap.sh && \
    ./configure --disable-udev --enable-shared && \
    make -j4

WORKDIR /opt/libusb-1.0.22/libusb
RUN /bin/mkdir -p '/usr/local/lib' && \
    /bin/bash ../libtool --mode=install /usr/bin/install -c   libusb-1.0.la '/usr/local/lib' && \
    /bin/mkdir -p '/usr/local/include/libusb-1.0' && \
    /usr/bin/install -c -m 644 libusb.h '/usr/local/include/libusb-1.0' && \
    /bin/mkdir -p '/usr/local/lib/pkgconfig'

WORKDIR /opt/libusb-1.0.22/
RUN wget https://raw.githubusercontent.com/openvinotoolkit/openvino/${OPENVINO_VERSION}/inference-engine/thirdparty/movidius/mvnc/src/97-myriad-usbboot.rules
RUN /usr/bin/install -c -m 644 libusb-1.0.pc '/usr/local/lib/pkgconfig' && \
    cp 97-myriad-usbboot.rules /etc/udev/rules.d/ && \
    ldconfig