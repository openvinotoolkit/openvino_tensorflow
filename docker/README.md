
# Dockerfiles for Ubuntu* 18.04 and Ubuntu* 20.04


We provide Dockerfiles for Ubuntu* 18.04 and Ubuntu* 20.04 which can be used to build runtime Docker* images for OpenVINO™ integration with TensorFlow on CPU, GPU, VPU, and VAD-M.
They contain all required runtime python packages, and shared libraries to support execution of a TensorFlow Python app with the OpenVINO™ backend. By default, it hosts a Jupyter server with an Image Classification and an Object Detection sample that demonstrate the performance benefits of using OpenVINO™ integration with TensorFlow.

Build the docker image

	docker build -t openvino/openvino_tensorflow_ubuntu20_runtime:2.0.0 - < ubuntu20/openvino_tensorflow_cgvh_runtime_2.0.0.dockerfile

Launch the Jupyter server with **CPU** access:

	docker run -it --rm \
		   -p 8888:8888 \
		   openvino/openvino_tensorflow_ubuntu20_runtime:2.0.0

Launch the Jupyter server with **iGPU** access:

	docker run -it --rm \
		   -p 8888:8888 \
		   --device-cgroup-rule='c 189:* rmw' \
		   --device /dev/dri:/dev/dri \
		   openvino/openvino_tensorflow_ubuntu20_runtime:2.0.0

Launch the Jupyter server with **MYRIAD** access:

	docker run -it --rm \
		   -p 8888:8888 \
		   --device-cgroup-rule='c 189:* rmw' \
		   -v /dev/bus/usb:/dev/bus/usb \
		   openvino/openvino_tensorflow_ubuntu20_runtime:2.0.0

Launch the Jupyter server with **VAD-M** access:

	docker run -itu root:root --rm \
		   -p 8888:8888 \
		   --device-cgroup-rule='c 189:* rmw' \
		   --mount type=bind,source=/var/tmp,destination=/var/tmp \
		   --device /dev/ion:/dev/ion \
		   -v /dev/bus/usb:/dev/bus/usb \
		   openvino/openvino_tensorflow_ubuntu20_runtime:2.0.0

Run image with runtime target /bin/bash for container shell with CPU, iGPU, and MYRIAD device access

	docker run -itu root:root --rm \
		   -p 8888:8888 \
		   --device-cgroup-rule='c 189:* rmw' \
		   --device /dev/dri:/dev/dri \
		   --mount type=bind,source=/var/tmp,destination=/var/tmp \
		   -v /dev/bus/usb:/dev/bus/usb \
		   openvino/openvino_tensorflow_ubuntu20_runtime:2.0.0 /bin/bash

If execution fails on iGPU for 10th and 11th Generation Intel devices, provide docker build arg INTEL_OPENCL as 20.35.17767 

	docker build -t openvino/openvino_tensorflow_ubuntu20_runtime:2.0.0 --build-arg INTEL_OPENCL=20.35.17767 - < ubuntu20/openvino_tensorflow_cgvh_runtime_2.0.0.dockerfile

# Dockerfiles for [TF-Serving](#https://github.com/tensorflow/serving) with OpenVINO<sup>TM</sup> integration with Tensorflow

Build serving docker images:

1. Build the runtime docker image. This dockerfile builds, and installs tensorflow model server binary onto the OpenVINO<sup>TM</sup> integration with Tensorflow runtime image.

		docker build -t openvino/openvino_tensorflow_ubuntu20_runtime:2.0.0-serving -f ubuntu20/openvino_tensorflow_cgvh_runtime_2.0.0-serving.dockerfile .

Here is an example to serve Resnet50 model using OpenVINO™ Integration with Tensorflow and a client script that performs inference on the model using the REST API.

1. Download [Resnet50 model](#https://storage.googleapis.com/tfhub-modules/google/imagenet/resnet_v2_50/classification/5.tar.gz) from TF Hub and untar its contents into the folder `resnet_v2_50_classifiation/5` 

2. Start serving container for the resnet50 model:
	
	To run on **CPU** backend:

		docker run -it --rm \
			   -p 8501:8501 \
			   -v <path to resnet_v2_50_classifiation>:/models/resnet \
			   -e MODEL_NAME=resnet \
			   openvino/openvino_tensorflow_ubuntu20_runtime:2.0.0-serving

	To run on **iGPU**:

		docker run -it --rm \
			   -p 8501:8501 \
			   --device-cgroup-rule='c 189:* rmw' \
			   --device /dev/dri:/dev/dri \
			   -v <path to resnet_v2_50_classifiation>:/models/resnet \
			   -e MODEL_NAME=resnet \
			   -e OPENVINO_TF_BACKEND=GPU \
			   openvino/openvino_tensorflow_ubuntu20_runtime:2.0.0-serving

	To run on **MYRIAD**:

		docker run -it --rm \
			   -p 8501:8501 \
			   --device-cgroup-rule='c 189:* rmw' \
			   -v /dev/bus/usb:/dev/bus/usb \
			   -v <path to resnet_v2_50_classifiation>:/models/resnet \
			   -e MODEL_NAME=resnet \
			   -e OPENVINO_TF_BACKEND=MYRIAD \
			   openvino/openvino_tensorflow_ubuntu20_runtime:2.0.0-serving
	
	To run on **VAD-M**:

		docker run -itu root:root --rm \
			   -p 8501:8501 \
			   --device-cgroup-rule='c 189:* rmw' \
			   -v /dev/bus/usb:/dev/bus/usb \
			   --mount type=bind,source=/var/tmp,destination=/var/tmp \
			   --device /dev/ion:/dev/ion \
			   -v <path to resnet_v2_50_classifiation>:/models/resnet \
			   -e MODEL_NAME=resnet \
			   openvino/openvino_tensorflow_ubuntu20_runtime:2.0.0-serving

3. Run the script to send inference request from client and get predictions from server.

		wget https://raw.githubusercontent.com/tensorflow/serving/master/tensorflow_serving/example/resnet_client.py
		python resnet_client.py

# Prebuilt Images

### [Ubuntu 18 runtime image on Docker* Hub](https://hub.docker.com/r/openvino/openvino_tensorflow_ubuntu18_runtime)
### [Ubuntu 20 runtime image on Docker* Hub](https://hub.docker.com/r/openvino/openvino_tensorflow_ubuntu20_runtime)
### [Azure* Marketplace](https://azuremarketplace.microsoft.com/en-us/marketplace/apps/intel_corporation.openvinotensorflow)

---
\* Other names and brands may be claimed as the property of others.
