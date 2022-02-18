
# Dockerfiles for Ubuntu* 18.04 and Ubuntu* 20.04


We provide Dockerfiles for Ubuntu* 18.04 and Ubuntu* 20.04 which can be used to build runtime Docker* images for OpenVINO™ integration with TensorFlow on CPU, GPU, and VPU.
They contain all required runtime python packages, and shared libraries to support execution of a TensorFlow Python app with the OpenVINO™ backend. By default, it hosts a Jupyter server with an Image Classification and an Object Detection sample that demonstrate the performance benefits of using OpenVINO™ integration with TensorFlow on the CPU.

Build the docker image

	$ docker build -t openvino_tensorflow/ubuntu20_runtime:1.1.0 - < ubuntu20/openvino_tensorflow_cgv_runtime_1.1.0.dockerfile

Launch the Jupyter server with

	$ docker run -it -p 8888:8888 openvino_tensorflow/ubuntu20_runtime:1.1.0

To get iGPU,and MYRIAD support

	$ docker run -it -p 8888:8888 -v /dev:/dev --network=host --privileged openvino_tensorflow/ubuntu20_runtime:1.1.0

Launches a Jupyter server by default. Change runtime target to /bin/bash for container shell

	$ docker run -it openvino_tensorflow/ubuntu20_runtime:1.1.0 /bin/bash

If execution fails on iGPU for 10th and 11th Generation Intel devices, provide docker build arg INTEL_OPENCL as 20.35.17767 

	$ docker build -t openvino_tensorflow/ubuntu20_runtime:1.1.0 --build-arg INTEL_OPENCV=20.35.17767 - < ubuntu20/openvino_tensorflow_cgv_runtime_1.1.0.dockerfile

---
\* Other names and brands may be claimed as the property of others.
