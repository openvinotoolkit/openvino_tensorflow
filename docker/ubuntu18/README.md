# Dockerfile for enabling [TF-Serving](#https://github.com/tensorflow/serving) with OpenVINO<sup>TM</sup> integration with Tensorflow

Build serving docker images:

1. Build the runtime docker image. This dockerfile uses the OpenVINO Integration with Tensorflow Runtime image as base, builds, and installs tensorflow model server binary.

		$ docker build -t openvino_tensorflow/tensorflow-serving -f openvino_tensorflow_cgvh_runtime_2.0.0-serving.dockerfile .

Here is an example to serve Resnet50 model using OpenVINO Integration with Tensorflow with the client script included in TF-Serving repository.

1. Download [Resnet50 model](#https://tfhub.dev/google/imagenet/resnet_v2_50/classification/5) from TF Hub and save it in saved model format. 

2. Start serving container Resnet50 model:
	
	To run on default CPU backend:

		$ docker run -t --rm -p 8501:8501 -v <path to Resnet50 model>:/models/resnet -e MODEL_NAME=resnet openvino_tensorflow/tensorflow-serving &

	To run on iGPU:

		$ docker run -t --rm -p 8501:8501 --device /dev/dri:/dev/dri -v <path to Resnet50 model>:/models/resnet MODEL_NAME=resnet  -e OPENVINO_TF_BACKEND=GPU openvino_tensorflow/tensorflow-serving &

	To run on Myriad:

		$ docker run -t --rm -p 8501:8501 --device-cgroup-rule='c 189:* rmw' -v /dev/bus/usb:/dev/bus/usb -v <path to Resnet50 model>:/models/resnet MODEL_NAME=resnet  -e OPENVINO_TF_BACKEND=MYRIAD openvino_tensorflow/tensorflow-serving &

3. Run the script to send inference request from client and get predictions from server.

		$ wget https://raw.githubusercontent.com/tensorflow/serving/master/tensorflow_serving/example/resnet_client.py
		$ python resnet_client.py