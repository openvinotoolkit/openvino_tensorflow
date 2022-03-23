# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

################################################################################
FROM openvino/ubuntu18_dev:2022.1.0 AS ovtf_build
################################################################################

# Stage 1 builds OpenVINO™ integration with TensorFlow from source, prepares wheel for use by the final image

LABEL description="This is the runtime image for OpenVINO™ integration with TensorFlow on Ubuntu 18.04 LTS"
LABEL vendor="Intel Corporation"

USER root

ENV DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-xo", "pipefail", "-c"]

ARG TF_VERSION="v2.8.0"
ARG OPENVINO_VERSION="2022.1"
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

RUN mkdir build_artifacts && \
    cp build_cmake/artifacts/*whl build_artifacts/ && \
    rm -rf build_cmake;

CMD ["/bin/bash"]

################################################################################
FROM openvino/ubuntu18_runtime:2022.1.0 AS ovtf_runtime
################################################################################

LABEL description="This is the runtime image for OpenVINO™ integration with TensorFlow on Ubuntu 18.04 LTS"
LABEL vendor="Intel Corporation"

USER root

ARG tf_package_url="https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v2.0.0/tensorflow_abi1-2.8.0-cp38-cp38-manylinux_2_12_x86_64.manylinux2010_x86_64.whl"

ENV DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-xo", "pipefail", "-c"]

RUN apt-get update; \
    dpkg --get-selections | grep -v deinstall | awk '{print $1}' > base_packages.txt; \
    apt-get install -y --no-install-recommends \
    git wget libsm6 \
    python3.8 python3.8-venv python3-pip; \
    rm -rf /var/lib/apt/lists/*;

# Download sources for GPL/LGPL packages
RUN apt-get update; \
    sed -Ei 's/# deb-src /deb-src /' /etc/apt/sources.list && \
    apt-get update && \
    dpkg --get-selections | grep -v deinstall | awk '{print $1}' > all_packages.txt && \
    grep -v -f base_packages.txt all_packages.txt | while read line; do \
    package=$(echo $line); \
    name=(${package//:/ }); \
    grep -l GPL /usr/share/doc/${name[0]}/copyright; \
    exit_status=$?; \
    if [ $exit_status -eq 0 ]; then \
    apt-get source -q --download-only $package;  \
    fi \
    done && \
    echo "Download source for $(ls | wc -l) third-party packages: $(du -sh)"; \
    rm -rf /var/lib/apt/lists/*;

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 70; \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 70;

ENV INTEL_OPENVINO_DIR /opt/intel/openvino

COPY --from=ovtf_build /opt/intel/openvino_tensorflow/ /home/openvino/openvino_tensorflow/

RUN chown openvino -R /home/openvino

WORKDIR /home/openvino/openvino_tensorflow/

RUN python3 -m pip install --upgrade pip; \
    python3 -m pip install --no-cache-dir ${tf_package_url}; \
    python3 -m pip install --no-cache-dir build_artifacts/openvino_tensorflow*whl; \
    python3 -m pip install --no-cache-dir -r examples/requirements.txt; \
    python3 -m pip install --upgrade numpy jupyter;

WORKDIR /home/openvino/openvino_tensorflow/examples/notebooks/

## Creating a shell script file which will be executed at the end to activate the environment and run jupyter notebook and
## Granting execution permission to shell script

RUN echo -e "source \${INTEL_OPENVINO_DIR}/setupvars.sh\njupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root" | \
        tee /home/openvino/run-jupyter.sh && chmod +x /home/openvino/run-jupyter.sh

USER openvino

## .run-jupyter.sh file will be executed when the container starts
CMD ["/bin/bash", "/home/openvino/run-jupyter.sh"]