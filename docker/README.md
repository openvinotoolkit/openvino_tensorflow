
<p>English | <a href="./README_cn.md">简体中文</a></p>

# **OpenVINO™ integration with TensorFlow Runtime** Dockerfiles for Ubuntu* 18.04 and Ubuntu* 20.04


We provide Dockerfiles for Ubuntu* 18.04 and Ubuntu* 20.04 which can be used to build runtime Docker* images for OpenVINO™ integration with TensorFlow on CPU, GPU, VPU, and VAD-M.
They contain all required runtime python packages, and shared libraries to support execution of a TensorFlow Python app with the OpenVINO™ backend. By default, it hosts a Jupyter server with an Image Classification and an Object Detection sample that demonstrate the performance benefits of using OpenVINO™ integration with TensorFlow.

The following ARGS are available to configure the docker build

TF_VERSION: TensorFlow version to be used. Defaults to "v2.9.1"
OPENVINO_VERSION: OpenVINO version to be used. Defaults to "2022.1.0"
OVTF_BRANCH: OpenVINO™ integration with TensorFlow branch to be used. Defaults to "releases/2.1.0"

Build the docker image

	docker build -t openvino/openvino_tensorflow_ubuntu20_runtime:2.1.0 - < ubuntu20/openvino_tensorflow_cgvh_runtime_2.1.0.dockerfile

Launch the Jupyter server with **CPU** access:

	docker run -it --rm \
		   -p 8888:8888 \
		   openvino/openvino_tensorflow_ubuntu20_runtime:2.1.0

Launch the Jupyter server with **iGPU** access:

	docker run -it --rm \
		   -p 8888:8888 \
		   --device-cgroup-rule='c 189:* rmw' \
		   --device /dev/dri:/dev/dri \
		   openvino/openvino_tensorflow_ubuntu20_runtime:2.1.0

Launch the Jupyter server with **MYRIAD** access:

	docker run -it --rm \
		   -p 8888:8888 \
		   --device-cgroup-rule='c 189:* rmw' \
		   -v /dev/bus/usb:/dev/bus/usb \
		   openvino/openvino_tensorflow_ubuntu20_runtime:2.1.0

Launch the Jupyter server with **VAD-M** access:

	docker run -itu root:root --rm \
		   -p 8888:8888 \
		   --device-cgroup-rule='c 189:* rmw' \
		   --mount type=bind,source=/var/tmp,destination=/var/tmp \
		   --device /dev/ion:/dev/ion \
		   -v /dev/bus/usb:/dev/bus/usb \
		   openvino/openvino_tensorflow_ubuntu20_runtime:2.1.0

Run image with runtime target /bin/bash for container shell with CPU, iGPU, and MYRIAD device access

	docker run -itu root:root --rm \
		   -p 8888:8888 \
		   --device-cgroup-rule='c 189:* rmw' \
		   --device /dev/dri:/dev/dri \
		   --mount type=bind,source=/var/tmp,destination=/var/tmp \
		   -v /dev/bus/usb:/dev/bus/usb \
		   openvino/openvino_tensorflow_ubuntu20_runtime:2.1.0 /bin/bash

If execution fails on iGPU for 10th and 11th Generation Intel devices, provide docker build arg INTEL_OPENCL as 20.35.17767 

	docker build -t openvino/openvino_tensorflow_ubuntu20_runtime:2.1.0 --build-arg INTEL_OPENCL=20.35.17767 - < ubuntu20/openvino_tensorflow_cgvh_runtime_2.1.0.dockerfile

# Dockerfiles for [TF-Serving](#https://github.com/tensorflow/serving) with OpenVINO<sup>TM</sup> integration with Tensorflow

The TF Serving dockerfile requires the **OpenVINO™ integration with TensorFlow Runtime** image to be built. Refer to the section above for instructions on building it.

The following ARGS are available to configure the docker build

TF_SERVING_VERSION: Tag of the TF Serving image to use to build the model serving executable. Defaults to "2.9.0"
OVTF_VERSION: Tag of the **OpenVINO™ integration with TensorFlow Runtime** image to use. Defaults to "2.1.0"

Build serving docker images:

1. This dockerfile builds, and installs tensorflow model server binary onto the **OpenVINO<sup>TM</sup> integration with Tensorflow Runtime** image.

		docker build -t openvino/openvino_tensorflow_ubuntu20_runtime:2.1.0-serving -f ubuntu20/openvino_tensorflow_cgvh_runtime_2.1.0-serving.dockerfile .

Here is an example to serve Resnet50 model using OpenVINO™ Integration with Tensorflow and a client script that performs inference on the model using the REST API.

1. Download [Resnet50 model](#https://storage.googleapis.com/tfhub-modules/google/imagenet/resnet_v2_50/classification/5.tar.gz) from TF Hub and untar its contents into the folder `resnet_v2_50_classifiation/5` 

2. Start serving container for the resnet50 model:
	
	To run on **CPU** backend:

		docker run -it --rm \
			   -p 8501:8501 \
			   -v <path to resnet_v2_50_classifiation>:/models/resnet \
			   -e MODEL_NAME=resnet \
			   openvino/openvino_tensorflow_ubuntu20_runtime:2.1.0-serving

	To run on **iGPU**:

		docker run -it --rm \
			   -p 8501:8501 \
			   --device-cgroup-rule='c 189:* rmw' \
			   --device /dev/dri:/dev/dri \
			   -v <path to resnet_v2_50_classifiation>:/models/resnet \
			   -e MODEL_NAME=resnet \
			   -e OPENVINO_TF_BACKEND=GPU \
			   openvino/openvino_tensorflow_ubuntu20_runtime:2.1.0-serving

	To run on **MYRIAD**:

		docker run -it --rm \
			   -p 8501:8501 \
			   --device-cgroup-rule='c 189:* rmw' \
			   -v /dev/bus/usb:/dev/bus/usb \
			   -v <path to resnet_v2_50_classifiation>:/models/resnet \
			   -e MODEL_NAME=resnet \
			   -e OPENVINO_TF_BACKEND=MYRIAD \
			   openvino/openvino_tensorflow_ubuntu20_runtime:2.1.0-serving
	
	To run on **VAD-M**:

		docker run -itu root:root --rm \
			   -p 8501:8501 \
			   --device-cgroup-rule='c 189:* rmw' \
			   -v /dev/bus/usb:/dev/bus/usb \
			   --mount type=bind,source=/var/tmp,destination=/var/tmp \
			   --device /dev/ion:/dev/ion \
			   -v <path to resnet_v2_50_classifiation>:/models/resnet \
			   -e MODEL_NAME=resnet \
			   openvino/openvino_tensorflow_ubuntu20_runtime:2.1.0-serving

3. Run the script to send inference request from client and get predictions from server.

		wget https://raw.githubusercontent.com/tensorflow/serving/master/tensorflow_serving/example/resnet_client.py
		python resnet_client.py

All related environmental variables that applies during the execution of **OpenVINO™ integration with TensorFlow** is applicable while running through containers also. For example, to disable **OpenVINO™ integration with TensorFlow** while starting a TensorFlow Serving container, simply provide OPENVINO_TF_DISABLE=1 as one of the environmental variables of the `docker run` command. See [USAGE.md](../docs/USAGE.md) for more such environmental variables.

		
		docker run -it --rm \
			   -p 8501:8501 \
			   -v <path to resnet_v2_50_classifiation>:/models/resnet \
			   -e MODEL_NAME=resnet \
			   -e OPENVINO_TF_DISABLE=1 \
			   openvino/openvino_tensorflow_ubuntu20_runtime:2.1.0-serving

# Prebuilt Images

### [Ubuntu 18 runtime image on Docker* Hub](https://hub.docker.com/r/openvino/openvino_tensorflow_ubuntu18_runtime)
### [Ubuntu 20 runtime image on Docker* Hub](https://hub.docker.com/r/openvino/openvino_tensorflow_ubuntu20_runtime)
### [Azure* Marketplace](https://azuremarketplace.microsoft.com/en-us/marketplace/apps/intel_corporation.openvinotensorflow)

---
\* Other names and brands may be claimed as the property of others.
