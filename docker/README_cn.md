
[English](./README.md) | 简体中文
# **OpenVINO™ integration with TensorFlow Runtime** Docker文件 Ubuntu* 18.04和Ubuntu* 20.04 


我们提供Ubuntu* 18.04和Ubuntu* 20.04 Dockerfiles， 可用来构建用于CPU、GPU、VPU和VAD-M上**OpenVINO™ integration with TensorFlow**的运行时Docker*图像。
它们包含所有运行时python所需安装包及共享库，以支持使用OpenVINO™后端执行TensorFlow Python应用程序。默认条件下，它可托管一个Jupyter服务器，该服务器附带Image Classification及演示在CPU上使用OpenVINO™ integration with TensorFlow的性能优势的Object Detection示例。

以下 ARGS 可用于配置 docker build

TF_VERSION：要使用的 TensorFlow 版本。默认为“v2.9.2”  
OPENVINO_VERSION：要使用的 OpenVINO 版本。默认为“2022.2.0”  
OVTF_BRANCH：要使用的 OpenVINO™ integration with TensorFlow 分支。默认为“releases/2.2.0”  

构建docker镜像

	docker build -t openvino/openvino_tensorflow_ubuntu20_runtime:2.2.0 - < ubuntu20/openvino_tensorflow_cgvh_runtime_2.2.0.dockerfile
启动可访问**CPU**的Jupyter服务器：

	docker run -it --rm \
		   -p 8888:8888 \
		   openvino/openvino_tensorflow_ubuntu20_runtime:2.2.0

启动可访问**iGPU**的Jupyter服务器： 

	docker run -it --rm \
		   -p 8888:8888 \
		   --device-cgroup-rule='c 189:* rmw' \
		   --device /dev/dri:/dev/dri \
		   openvino/openvino_tensorflow_ubuntu20_runtime:2.2.0


启动可访问**MYRIAD**的Jupyter服务器： 

	docker run -it --rm \
		   -p 8888:8888 \
		   --device-cgroup-rule='c 189:* rmw' \
		   -v /dev/bus/usb:/dev/bus/usb \
		   openvino/openvino_tensorflow_ubuntu20_runtime:2.2.0

启动可访问**VAD-M**的Jupyter服务器：

	docker run -itu root:root --rm \
		   -p 8888:8888 \
		   --device-cgroup-rule='c 189:* rmw' \
		   --mount type=bind,source=/var/tmp,destination=/var/tmp \
		   --device /dev/ion:/dev/ion \
		   -v /dev/bus/usb:/dev/bus/usb \
		   openvino/openvino_tensorflow_ubuntu20_runtime:2.2.0

启动可以访问“所有“计算单元的容器，并通过/bin/bash 提供容器shell访问：

	docker run -itu root:root --rm \
		   -p 8888:8888 \
		   --device-cgroup-rule='c 189:* rmw' \
		   --device /dev/dri:/dev/dri \
		   --mount type=bind,source=/var/tmp,destination=/var/tmp \
		   -v /dev/bus/usb:/dev/bus/usb \
		   openvino/openvino_tensorflow_ubuntu20_runtime:2.2.0 /bin/bash

如果在英特尔第10和11代设备iGPU上执行失败， 请设定docker构建参数INTEL_OPENCL为20.35.17767 

	docker build -t openvino/openvino_tensorflow_ubuntu20_runtime:2.2.0 --build-arg INTEL_OPENCL=20.35.17767 - < ubuntu20/openvino_tensorflow_cgvh_runtime_2.2.0.dockerfile

# Dockerfiles for [TF-Serving](#https://github.com/tensorflow/serving) with OpenVINO<sup>TM</sup> integration with Tensorflow

The TF Serving dockerfile requires the **OpenVINO™ integration with TensorFlow Runtime** image to be built. Refer to the section above for instructions on building it.

以下 ARGS 可用于配置 docker build

TF_SERVING_VERSION: 用于构建模型服务可执行文件的 TF Serving 映像的标记。默认为“v2.9.2”  
OVTF_VERSION: 要使用的 **OpenVINO™ integration with TensorFlow Runtime** 集成图像的标签。认为"2.2.0"    

构建服务docker镜像：
1. 该docker文件可构建OpenVINO<sup>TM</sup> integration with Tensorflow运行时镜像并在上面安装tensorflow模型服务器二进制文件。

		docker build -t openvino/openvino_tensorflow_ubuntu20_runtime:2.2.0-serving -f ubuntu20/openvino_tensorflow_cgvh_runtime_2.2.0-serving.dockerfile .

此处为Resnet50模型使用OpenVINO Integration with Tensorflow实例，提供了REST API相关客户端脚本。

1. 从TF社区下载[Resnet50 model](#https://storage.googleapis.com/tfhub-modules/google/imagenet/resnet_v2_50/classification/5.tar.gz)并将其目录解压至`resnet_v2_50_classifiation/5`文件夹。 

2. 启动resnet50模型的服务容器：
	
	在**CPU**后端上运行：

		docker run -it --rm \
			   -p 8501:8501 \
			   -v <path to resnet_v2_50_classifiation>:/models/resnet \
			   -e MODEL_NAME=resnet \
			   openvino/openvino_tensorflow_ubuntu20_runtime:2.2.0-serving

	在**iGPU**上运行：

		docker run -it --rm \
			   -p 8501:8501 \
			   --device-cgroup-rule='c 189:* rmw' \
			   --device /dev/dri:/dev/dri \
			   -v <path to resnet_v2_50_classifiation>:/models/resnet \
			   -e MODEL_NAME=resnet \
			   -e OPENVINO_TF_BACKEND=GPU \
			   openvino/openvino_tensorflow_ubuntu20_runtime:2.2.0-serving

	在**MYRIAD**上运行：

		docker run -it --rm \
			   -p 8501:8501 \
			   --device-cgroup-rule='c 189:* rmw' \
			   -v /dev/bus/usb:/dev/bus/usb \
			   -v <path to resnet_v2_50_classifiation>:/models/resnet \
			   -e MODEL_NAME=resnet \
			   -e OPENVINO_TF_BACKEND=MYRIAD \
			   openvino/openvino_tensorflow_ubuntu20_runtime:2.2.0-serving
	
	在**MYRIAD**上运行：

		docker run -itu root:root --rm \
			   -p 8501:8501 \
			   --device-cgroup-rule='c 189:* rmw' \
			   -v /dev/bus/usb:/dev/bus/usb \
			   --mount type=bind,source=/var/tmp,destination=/var/tmp \
			   --device /dev/ion:/dev/ion \
			   -v <path to resnet_v2_50_classifiation>:/models/resnet \
			   -e OPENVINO_TF_BACKEND=VAD-M \
			   -e MODEL_NAME=resnet \
			   openvino/openvino_tensorflow_ubuntu20_runtime:2.2.0-serving

3. 运行脚本从客户端发送推理请求并从服务器获取预测。
		wget https://raw.githubusercontent.com/tensorflow/serving/master/tensorflow_serving/example/resnet_client.py
		python resnet_client.py

在执行 **OpenVINO™ integration with TensorFlow** 集成时应用的所有相关环境变量在通过容器运行时也适用。例如，要在启动 TensorFlow Serving 容器时禁用 **OpenVINO™ integration with TensorFlow 的集成**，只需提供 OPENVINO_TF_DISABLE=1 作为 `docker run` 命令的环境变量之一。有关更多此类环境变量，请参见 [USAGE.md](../docs/USAGE_cn.md)。


		docker run -it --rm \
			   -p 8501:8501 \
			   -v <path to resnet_v2_50_classifiation>:/models/resnet \
			   -e MODEL_NAME=resnet \
			   -e OPENVINO_TF_DISABLE=1 \
			   openvino/openvino_tensorflow_ubuntu20_runtime:2.2.0-serving

# 预构建镜像

- [Ubuntu18 runtime image on Docker* Hub](https://hub.docker.com/r/openvino/openvino_tensorflow_ubuntu18_runtime)
- [Ubuntu20 runtime image on Docker* Hub](https://hub.docker.com/r/openvino/openvino_tensorflow_ubuntu20_runtime)
- [Azure* Marketplace](https://azuremarketplace.microsoft.com/en-us/marketplace/apps/intel_corporation.openvinotensorflow)

---
\* 其他名称和品牌可能已被声明为他人资产。
