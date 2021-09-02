# 英特尔<sup>®</sup> OpenVINO<sup>TM</sup> 与 TensorFlow - C++ 和 Python 集成示例

这些示例演示了如何使用**英特尔<sup>®</sup> </sup>OpenVINO<sup>TM</sup> integration with Tensorflow** 识别和检测在图像和视频中的对象。

## 示例中的演示

* 分类演示使用谷歌 Inception v3 模型对指定的图像或视频进行分类。
* 对象检测演示使用 YOLOv3 模型检测指定图像或视频中的对象。

## 示例的设置

继续运行示例之前，必须将 `openvino_tensorflow` 仓库克隆到本地设备。此步骤请运行以下命令：

```bash
$ git clone https://github.com/openvinotoolkit/openvino_tensorflow.git
$ cd openvino_tensorflow
$ git submodule init
$ git submodule update --recursive
```

## 面向分类的 Python 实现

在此示例中，我们假设您已经：

* 为系统安装了 TensorFlow 
* 为系统安装了**英特尔<sup>®</sup> </sup> OpenVINO<sup>TM</sup> integration with Tensorflow**

请参阅[**此页**](https://github.com/openvinotoolkit/openvino_tensorflow#installation)了解如何快速通过 pip 安装。

此演示中所使用的 TensorFlow 模型由于大小问题，不包含在该仓库中。因此，请将模型下载至 `cloned repo of openvino_tensorflow` 中的 `data` 目录，并解压文件：

```bash
$ curl -L "https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz" |
  tar -C <path-to-openvino_tensorflow-repository>/examples/data -xz
```

解压后，数据文件夹将包含两个新文件：

* imagenet\_slim\_labels.txt
* inception\_v3\_2016\_08\_28\_frozen.pb

打开 `imagenet_slim_labels.txt`，读取数据目录中关于分类的标签。在.txt 文件中，有 1,000 个 Imagenet 竞赛中使用的类别。

现在，您可以使用图像输入运行分类示例，如下所示：

```bash
$ cd <path-to-openvino_tensorflow-repository>
$ python3 examples/classification_sample.py
```

`classification_sample.py` 对此仓库自带的默认示例图像进行推理，其输出类似于：

```
military uniform (653): 0.834306
mortarboard (668): 0.0218693
academic gown (401): 0.010358
pickelhaube (716): 0.00800814
bulletproof vest (466): 0.00535091
```

在此示例中，我们使用 Admiral Grace Hopper 的默认图像。如您所见，网络准确发现她穿着军装，得到了 0.8 的高分。

接下来，传递 --image=argument（其中 argument 是新图像的路径），试着推理自己的图像。您可以在参数中提供绝对路径或相对路径。

```bash
$ python3 examples/classification_sample.py --image=<absolute-or-relative-path-to-your-image>
```

如果您将新图像（如 my\_image.png）添加到 openvino\_tensorflow 仓库的现有数据目录，将会是这样：

```bash
$ python3 examples/classification_sample.py --image=examples/data/my_image.png
```

如要查看各种后端（英特尔<sup>®</sup> 硬件)的更多选项，请调用：

```bash
$ python3 examples/classification_sample.py --help
```

如要使用视频输入运行分类示例，请按照以下说明操作：

```bash
$ pip3 install opencv-python
$ cd <path-to-openvino_tensorflow-repository>
$ python3 examples/classification_sample_video.py
```

接下来，传递 --input=argument（其中 argument 是输入视频的路径），试着推理自己的视频文件。您可以在参数中提供绝对路径或相对路径。

```bash
$ python3 examples/classification_sample_video.py --input=<absolute-or-relative-path-to-your-video-file>
```

如果您将新视频（如 examples/data/people-detection.mp4）添加到 openvino\_tensorflow 仓库的现有数据目录，将会是这样：

```bash
$ python3 examples/classification_sample_video.py --input=examples/data/people-detection.mp4
```

## 面向对象检测的 Python 实现

在此示例中，我们假设您已经：

* 为系统安装了 TensorFlow
* 为系统安装了**英特尔<sup>®</sup> </sup> OpenVINO<sup>TM</sup> integration with Tensorflow**

请参阅[**此页**](https://github.com/openvinotoolkit/openvino_tensorflow#installation)了解如何快速通过 pip 安装。

此演示中所使用的 TensorFlow Yolo v3 模型 由于大小问题，不包含在仓库中。因此，请按照以下说明将模型从 DarkNet 转换为 TensorFlow，并将标签和权重下载到 `cloned repo of openvino_tensorflow` 中的 `data` 目录：

请注意：不能在虚拟环境中执行以下指令。convert\_yolov3.sh 脚本会创建python 虚拟环境用于模型转换。

```bash
$ cd <path-to-openvino_tensorflow-repository>/examples
$ chmod +x convert_yolov3.sh
$ ./convert_yolov3.sh
```

完成后，数据文件夹将包含运行对象检测示例所需的以下文件：

* coco.names
* yolo\_v3\_darknet.pb

使用以下指令运行对象检测示例：

```bash
$ cd <path-to-openvino_tensorflow-repository>
$ python3 examples/object_detection_sample.py
```

它将使用此仓库自带的默认示例图像，其输出类似于：

<p align="left">
  <img src="../examples/data/detections.jpg" width="200" height="200"
</p>

在此示例中，我们使用 Admiral Grace Hopper 的默认图像。如您所见，网络检测到并准确绘制了人物边界框。

接下来，传递 --image=argument（其中 argument 是新图像的路径），试着推理自己的图像。您可以在参数中提供绝对路径或相对路径。

```bash
$ python3 examples/object_detection_sample.py --image=<absolute-or-relative-path-to-your-image>
```

如果您将新图像（如 my\_image.png）添加到 openvino\_tensorflow 存储库的现有数据目录，将会是这样：

```bash
$ python3 examples/object_detection_sample.py --image=examples/data/my_image.png
```

如要查看各种后端（英特尔<sup>®</sup> 硬件)的更多选项，请调用：

```bash
$ python3 examples/object_detection_sample.py --help
```

如要使用视频输入运行对象检测示例，请按照以下说明操作：

```bash
$ pip3 install opencv-python
$ cd <path-to-openvino_tensorflow-repository>
$ python3 examples/object_detection_sample_video.py
```

接下来，传递 --input=argument（其中 argument 是输入视频的路径），试着推理自己的视频文件。您可以在参数中提供绝对路径或相对路径。

```bash
$ python3 examples/object_detection_sample_video.py --input=<absolute-or-relative-path-to-your-video-file>
```

如果您将新视频（如 examples/data/people-detection.mp4）添加到 openvino\_tensorflow 存储库的现有数据目录，将会是这样：

```bash
$ python3 examples/object_detection_sample_video.py --input=examples/data/people-detection.mp4
```

如要使用 yolo\_v3\_160 模型加快推理，请按以下步骤操作：

请注意：不能在活跃的虚拟环境中执行以下指令。convert\_yolov3\_160.sh 脚本激活用于转换的 python 虚拟环境。

```bash
$ cd <path-to-openvino_tensorflow-repository>/examples
$ chmod +x convert_yolov3_160.sh
$ ./convert_yolov3_160.sh
```

使用以下指令运行对象检测示例：

```bash
$ cd <path-to-openvino_tensorflow-repository>
$ python3 examples/object_detection_sample_vid.py --input_height 160 --input_width 160 --graph "examples/data/yolo_v3_160.pb" --input_layer "inputs" --output_layer "output_boxes" --labels "examples/data/coco.names"
```

## 面向分类的 C++ 实现

运行 C++ 示例时，我们需要通过源代码构建 TensorFlow 框架，因为示例依赖 TensorFlow 库。

通过源代码构建之前，必须确保您已安装以下关联组件：

* Python 3.6、3.7 或 3.8
* GCC 7.5 (Ubuntu 18.04)
* Cmake 3.14 或更高版本
* [Bazelisk v1.7.5](https://github.com/bazelbuild/bazelisk/tree/v1.7.5)
* Virtualenv 16.0.0 或更高版本
* Patchelf 0.9

此演示中所使用的 TensorFlow 模型由于大小问题，不包含在仓库中。因此，请将模型下载至 `cloned repo of openvino_tensorflow` 中的 `data` 目录，并解压文件：

```bash
$ curl -L "https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz" |
  tar -C <path-to-openvino_tensorflow-repository>/examples/data -xz
```

运行以下命令，构建 openvino\_tensorflow 及示例：

```bash
$ cd <path-to-openvino_tensorflow-repository>
$ python3 build_tf.py --output_dir <path-to-tensorflow-dir>
$ python3 build_ovtf.py --use_tensorflow_from_location <path-to-tensorflow-dir>
```

关于详细的构建说明，请参阅 [**BUILD.md**](https://github.com/openvinotoolkit/openvino_tensorflow/blob/master/docs/BUILD.md#build-from-source)。

现在，classification\_sample 的二进制可执行文件已构建完成。更新 LD\_LIBRARY\_PATH 并运行示例：

```bash
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<path-to-openvino_tensorflow-repository>/build_cmake/artifacts/lib:<path-to-openvino_tensorflow-repository>/build_cmake/artifacts/tensorflow
$ ./build_cmake/examples/classification_sample/infer_image
```

它将使用此存储库自带的默认示例图像，其输出类似于：

```
military uniform (653): 0.834306
mortarboard (668): 0.0218693
academic gown (401): 0.010358
pickelhaube (716): 0.00800814
bulletproof vest (466): 0.00535091
```

在此示例中，我们使用 Admiral Grace Hopper 的默认图像。如您所见，网络准确发现她穿着军装，得到了 0.8 的高分。

接下来，传递 --image=argument（其中 argument 是新图像的路径），试着推理自己的图像。您可以在参数中提供绝对路径或相对路径。

```bash
$ ./build_cmake/examples/classification_sample/infer_image --image=<absolute-or-relative-path-to-your-image>
```

如果您将新图像（如 my\_image.png）添加到 openvino\_tensorflow 仓库的现有数据目录，将会是这样：

```bash
$ ./build_cmake/examples/classification_sample/infer_image --image=examples/data/my_image.png
```

如要查看各种后端（英特尔<sup>®</sup> 硬件)的更多选项，请调用：

```bash
$ ./build_cmake/examples/classification_sample/infer_image --help
```