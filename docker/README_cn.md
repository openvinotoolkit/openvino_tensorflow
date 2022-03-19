
[English](./README.md) | 简体中文
#Ubuntu* 18.04和Ubuntu* 20.04 Docker文件


我们提供Ubuntu* 18.04和Ubuntu* 20.04 Dockerfiles， 可用来构建用于CPU、GPU、VPU和VAD-M上**OpenVINO™ integration with TensorFlow**的运行时Docker*图像。
它们包含所有python运行时所需安装包及共享库，以支持使用OpenVINO™后端执行TensorFlow Python应用程序。默认条件下，它可托管一个Jupyter服务器，该服务器附带Image Classification及演示在CPU上使用OpenVINO™ integration with TensorFlow的性能优势的Object Detection示例。

构建docker镜像

	$ docker build -t openvino/openvino_tensorflow_ubuntu20_runtime:2.0.0 - < ubuntu20/openvino_tensorflow_cgvh_runtime_2.0.0.dockerfile
启动Jupyter服务器

	$ docker run -it -p 8888:8888 openvino/openvino_tensorflow_ubuntu20_runtime:2.0.0

获取iGPU及MYRIAD支持

	$ docker run -it -p 8888:8888 -v /dev:/dev --network=host --device /dev/dri:/dev/dri --device-cgroup-rule='c 189:* rmw' -v /dev/bus/usb:/dev/bus/usb openvino/openvino_tensorflow_ubuntu20_runtime:2.0.0

默认启动Jupyter服务器。将运行时目标更改到容器shell的/bin/bash。

	$ docker run -it openvino/openvino_tensorflow_ubuntu20_runtime:2.0.0 /bin/bash

如果在英特尔第10和11代设备iGPU上执行失败， 请设定docker构建参数INTEL_OPENCL为20.35.17767 

	$ docker build -t openvino/openvino_tensorflow_ubuntu20_runtime:2.0.0 --build-arg INTEL_OPENCL=20.35.17767 - < ubuntu20/openvino_tensorflow_cgvh_runtime_2.0.0.dockerfile

# Dockerfiles for [TF-Serving](#https://github.com/tensorflow/serving) with OpenVINO<sup>TM</sup> integration with Tensorflow

构建服务docker镜像：
1. 构建运行时docker镜像。该docker文件可构建OpenVINO<sup>TM</sup> integration with Tensorflow运行时镜像并在上面安装tensorflow模型服务器二进制文件。

		$ docker build -t openvino/openvino_tensorflow_ubuntu20_runtime:2.0.0-serving -f ubuntu20/openvino_tensorflow_cgvh_runtime_2.0.0-serving.dockerfile .

此处为使用带有TF-Serving储存库内所含客户端脚本的OpenVINO Integration with Tensorflow服务Resnet50模型的示例。

1. 从TF社区下载[Resnet50 model](#https://tfhub.dev/google/imagenet/resnet_v2_50/classification/5)并将其以已保存模型形式保存。 

2. 启动Resnet50模型的服务容器：
	
	默认运行CPU后端：

		$ docker run -t --rm -p 8501:8501 -v <path to Resnet50 model>:/models/resnet -e MODEL_NAME=resnet openvino/openvino_tensorflow_ubuntu20_runtime:2.0.0-serving &

	在iGPU上运行：

		$ docker run -t --rm -p 8501:8501 --device /dev/dri:/dev/dri -v <path to Resnet50 model>:/models/resnet MODEL_NAME=resnet  -e OPENVINO_TF_BACKEND=GPU openvino/openvino_tensorflow_ubuntu20_runtime:2.0.0-serving &

	在Myriad上运行：

		$ docker run -t --rm -p 8501:8501 --device-cgroup-rule='c 189:* rmw' -v /dev/bus/usb:/dev/bus/usb -v <path to Resnet50 model>:/models/resnet MODEL_NAME=resnet  -e OPENVINO_TF_BACKEND=MYRIAD openvino/openvino_tensorflow_ubuntu20_runtime:2.0.0-serving &

3. 运行脚本从客户端发送推理请求并从服务器获取预测。

		$ wget https://raw.githubusercontent.com/tensorflow/serving/master/tensorflow_serving/example/resnet_client.py
		$ python resnet_client.py

# 预构建镜像

- [Docker* Hub](https://hub.docker.com/u/openvino/)
- [Azure* Marketplace](https://azuremarketplace.microsoft.com/en-us/marketplace/apps/intel_corporation.openvino)

---
\* 其他名称和品牌可能已被声明为他人资产。
