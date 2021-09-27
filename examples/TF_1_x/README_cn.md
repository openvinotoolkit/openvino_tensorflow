[English](./README.md) | 简体中文

# Intel<sup>®</sup> OpenVINO<sup>TM</sup> integration with TensorFlow Python Object Detection 示例

此示例演示了如何使用 **Intel<sup>®</sup> </sup> OpenVINO<sup>TM</sup> integration with Tensorflow** 来识别和检测图像和视频中的对象。

## 此示例中展示的内容

 使用从 Darknet 转换而来的 YOLOv3 模型进行对象检测，以检测给定图像、视频、目录或相机输入中的对象。

## 示例的设置

在继续运行示例之前，您必须将 `openvino_tensorflow` 存储库克隆到本地计算机。 为此请运行以下命令：

```bash
$ git clone https://github.com/openvinotoolkit/openvino_tensorflow.git
$ cd openvino_tensorflow
$ git submodule init
$ git submodule update --recursive
```

## 用于对象检测的 Python 实现

对于此示例，我们假设您已经：

* 在您的系统上安装了 TensorFlow
* 在您的系统上安装了  **Intel<sup>®</sup> </sup> OpenVINO<sup>TM</sup> integration with Tensorflow**

参考 [**this page**](https://github.com/openvinotoolkit/openvino_tensorflow#installation) 使用 pip 快速安装。

安装先决条件
```bash
$ cd <path-to-openvino_tensorflow-repository>/examples
$ pip3 install -r requirements.txt
```

本演示中使用的 TensorFlow Yolo v3 模型由于其大小而未打包在该仓库中。 因此，请按照以下说明将模型从 DarkNet 转换为 TensorFlow，并将标签和权重下载到“openvino_tensorflow 的克隆存储库”中的“<path-to-openvino_tensorflow-repository>/examples/data”目录：

请注意：以下说明不应在活动的虚拟环境中执行。 convert_yolov3.sh 脚本激活用于转换的python 虚拟环境。

```bash
$ cd <path-to-openvino_tensorflow-repository>/examples/TF_1_x
$ chmod +x convert_yolov3.sh
$ ./convert_yolov3.sh
```

完成后，`<path-to-openvino_tensorflow-repository>/examples/data` 文件夹将包含运行对象检测示例所需的以下文件：

* coco.names
* yolo_v3_darknet_1.pb

使用以下说明运行对象检测示例：

```bash
$ cd <path-to-openvino_tensorflow-repository>
$ python3 examples/TF_1_x/object_detection_sample.py
```

**注意**：使用 ```--no_show``` 标志禁用应用程序显示窗口。 默认情况下，显示窗口是启用的。

使用此仓库附带的默认示例图像，并且输出类似于下面的内容并将结果写入 detections.jpg：



<p align="left">
  <img src="../data/detections.jpg" width="200" height="200"
</p>

在本例中，我们使用 Admiral Grace Hopper 的默认图像。 如您所见，网络检测并正确绘制了人物周围的边界框。

接下来，通过将路径传递到新图像，在您自己的图像上进行尝试。 您可以提供绝对或相对路径，例如

```bash
$ python3 examples/TF_1_x/object_detection_sample.py --image=<absolute-or-relative-path-to-your-image>
```

如果您将新图像（例如 my_image.png）添加到 openvino_tensorflow 仓库中现有的 `<path-to-openvino_tensorflow-repository>/examples/data` 目录，它将如下所示：

```bash
$ python3 examples/TF_1_x/object_detection_sample.py --image=examples/data/my_image.png
```

要查看各种后端（英特尔<sup>®</sup> 硬件）的更多选项，请调用：
```bash
$ python3 examples/TF_1_x/object_detection_sample.py --help
```

接下来，通过将路径传递到您的输入视频，在您自己的视频文件上进行尝试。 您可以提供绝对或相对路径，例如

```bash
$ python3 examples/TF_1_x/object_detection_sample.py --input=<absolute-or-relative-path-to-your-video-file>
```
如果您将新视频（例如，examples/data/people-detection.mp4）添加到 openvino_tensorflow 仓库中现有的 `<path-to-openvino_tensorflow-repository>/examples/data` 目录，它将如下所示：

```bash
$ python3 examples/TF_1_x/object_detection_sample.py --input=examples/data/people-detection.mp4
```

使用相机作为输入使用```--input=0```。 这里的“0”指的是 /dev/video0 中的摄像头。 如果相机连接到不同的端口，请适当更改。

**注意：** 输入为图像或图像目录的结果将写入输出图像。 对于视频或相机输入，请使用应用程序显示窗口查看结果。

要尝试使用 yolo_v3_160 模型进行更快的推理，请按照以下步骤操作

请注意：以下说明不应在活动的虚拟环境中执行。 convert_yolov3_160.sh 脚本激活 python 虚拟环境进行转换。

```bash
$ cd <path-to-openvino_tensorflow-repository>/examples/TF_1_x
$ chmod +x convert_yolov3_160.sh
$ ./convert_yolov3_160.sh
```

Run the object detection example using the instructions below:

```bash
$ cd <path-to-openvino_tensorflow-repository>
$ python3 examples/TF_1_x/object_detection_sample.py --input_height 160 --input_width 160 --graph "examples/data/yolo_v3_darknet_160.pb" --input_layer "inputs" --output_layer "output_boxes" --labels "examples/data/coco.names"
```
