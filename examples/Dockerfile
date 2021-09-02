#-------------------------------------------------------------------------------
# Docker image builder for OpenVINO itegration with Tensorflow on Ubuntu 18.04
#-------------------------------------------------------------------------------
FROM ubuntu:18.04 as tf_builder

# Install SW packages
RUN apt-get update \
    && apt-get -y upgrade \
    && apt-get -y install python3 python3-pip python3-venv git curl wget

RUN pip3 install -U pip==21.0.1 \
    && pip3 install -U tensorflow==2.4.1 \
    && pip3 install openvino_tensorflow \
    && pip3 install pillow \
    && pip3 install opencv-python-headless \
    && pip3 install virtualenv

# Download and build examples
RUN git clone https://github.com/openvinotoolkit/openvino_tensorflow.git \
    && cd openvino_tensorflow \
    && git checkout notebook_demo_branch \
    && curl -L "https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz" | tar -C ./examples/data -xz \
    && cd examples \
    && chmod +x convert_yolov3.sh \
    && ./convert_yolov3.sh \
    && chmod +x convert_yolov3_160.sh \
    && ./convert_yolov3_160.sh

