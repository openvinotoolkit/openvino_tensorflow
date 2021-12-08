[English](./README.md) | 简体中文

# 英特尔<sup>®</sup> OpenVINO<sup>TM</sup> integration with TensorFlow - C++ 和 Python 集成示例

这些示例演示了如何使用 **Intel<sup>®</sup> </sup>OpenVINO<sup>TM</sup> integration with Tensorflow** 识别和检测在图像和视频中的对象。

## 示例快速链接

* [分类实例的Python实现](#python-implementation-for-classification)
* [对象检测实例的Python实现](#python-implementation-for-object-detection)
* [分类实例的C++实现](#c-implementation-for-classification)
    - [Linux and macOS](#linux-and-macos)
    - [Windows](#windows)

## 示例中的演示

* 分类演示使用谷歌 Inception v3 模型对指定的图像或视频进行分类。
* 对象检测演示使用从 Darknet 转换而来的 YOLOv4 模型检测指定图像、视频、图像目录或相机输入中的对象。

## 示例的设置

继续运行示例之前，必须将 `openvino_tensorflow` 仓库克隆到本地设备。此步骤请运行以下命令：

```bash
$ git clone https://github.com/openvinotoolkit/openvino_tensorflow.git
$ cd openvino_tensorflow
$ git submodule init
$ git submodule update --recursive
```

<br/> 

## 分类实例的 Python 实现

在此示例中，我们假设您已经：
* 为系统安装了 TensorFlow 
* 为系统安装了**Intel<sup>®</sup> </sup> OpenVINO<sup>TM</sup> integration with Tensorflow**

**注意：**有关使用TF1 Inception v3模型的python分类示例，请参阅[**此页**](TF_1_x/README.md)。

请参阅[**this page**](https://github.com/openvinotoolkit/openvino_tensorflow#installation) 使用pip快速安装。

此演示使用TF Hub 图像分类模型 [**InceptionV3**](https://tfhub.dev/google/imagenet/inception_v3/classification/4)与可能用于分类的[**ImageNet标签**](https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt).在标签文件中，您将找到Imagenet比赛中曾使用的1,000 个类别。

安装先决条件
```bash
$ cd <path-to-openvino_tensorflow-repository>/examples
$ pip3 install -r requirements.txt
```
现在，您可以使用如下说明运行分类示例：


```bash
$ cd <path-to-openvino_tensorflow-repository>
$ python3 examples/classification_sample.py
```

`classification_sample.py` 在[**image**]("https://www.tensorflow.org/images/grace_hopper.jpg")上进行推理，其输出类似于：

```
military uniform 0.79601693
mortarboard 0.02091024
academic gown 0.014557127
suit 0.009166162
comic book 0.007978318
```
**注意**：使用 ```--no_show``` 标志禁用应用程序显示窗口。 默认情况下，显示窗口是启用的。

在此示例中，我们使用 Admiral Grace Hopper 的默认图像。如您所见，网络准确发现她穿着军装，得到了 0.8 的高分。

下一步，通过将路径传递给新输入来尝试一下。 您可以提供图像或视频或图像目录的绝对或相对路径。
例如：
```bash
$ python3 examples/classification_sample.py --input=<absolute-or-relative-path-to-your-input>
```
如果您将新图像或视频（如 my\_image.png）添加到 openvino\_tensorflow 仓库现有的 `<path-to-openvino_tensorflow-repository>/examples/data`  目录，将会是这样：

```bash
$ python3 examples/classification_sample.py --input=examples/data/my_image.png

或

$ python3 examples/classification_sample.py --input=examples/data/people-detection.mp4
```

使用相机作为输入使用```--input=0```。 这里的“0”指的是 /dev/video0 中的摄像头。 如果相机连接到不同的端口，请适当更改。 

如要查看各种后端（Intel<sup>®</sup> 硬件)的更多选项，请调用：
```bash
$ python3 examples/classification_sample.py --help
```

<br/> 

## 对象检测的Python实现

在此示例中，我们假设您已经：

* 为系统安装了 TensorFlow 


* 为系统安装了**Intel<sup>®</sup> </sup> OpenVINO<sup>TM</sup> integration with Tensorflow**


**注意：** 参考[**此页**](TF_1_x/README_cn.md)转换yolov4 darknet模型及其python对象检测示例。


使用pip快速安装，请参考[**此页**](https://github.com/openvinotoolkit/openvino_tensorflow#installation)。

安装先决条件
```bash
$ cd <path-to-openvino_tensorflow-repository>/examples
$ pip3 install -r requirements.txt
```

本演示中使用的 TensorFlow Yolo v4 darknet模型由于其大小而未打包在存储库中。 因此请按照以下说明将 DarkNet 模型转换为 TensorFlow模型，并将标签和权重下载到“openvino_tensorflow 的克隆存储库”中的“<path-to-openvino_tensorflow-repository>/examples/data”目录：


```bash
$ cd <path-to-openvino_tensorflow-repository>/examples
$ chmod +x convert_yolov4.sh
$ ./convert_yolov4.sh
```

完成后，`<path-to-openvino_tensorflow-repository>/examples/data`文件夹将包含运行对象检测示例所需的以下文件：

* coco.names
* yolo_v4

使用以下说明运行对象检测示例：

```bash
$ cd <path-to-openvino_tensorflow-repository>
$ python3 examples/object_detection_sample.py
```
**注意**：
*使用```--no_show``` 标志禁用应用程序显示窗口。 默认情况下，显示窗口是启用的。、
*使用```--rename```根据图像内容重命名输入图像或图像目录，例如***noexiftags-1person-1uniform-intelOpenVINO.jpg***

使用此仓库附带的默认示例图像，并且
输出类似于下面的内容并将结果写入 detections.jpg：

<p align="left">
  <img src="../examples/data/detections.jpg" width="200" height="200"
</p>

在此示例中，我们使用 Admiral Grace Hopper 的默认图像。如您所见，网络检测到并准确绘制了人物边界框。

接下来，通过将路径传递给您的新输入，在您自己的图像上进行尝试。 您可以提供绝对或相对路径，例如：

```bash
$ python3 examples/object_detection_sample.py --input=<absolute-or-relative-path-to-your-image>
```

如果您将新图像（如 my\_image.png）添加到 openvino\_tensorflow 存储库的现有 `<path-to-openvino_tensorflow-repository>/examples/data` 目录，将会是这样：

```bash
$ cd <path-to-openvino_tensorflow-repository>
$ python3 examples/object_detection_sample.py --input=examples/data/my_image.png
```

如要查看各种后端（Intel<sup>®</sup> 硬件)的更多选项，请调用：
```bash
$ python3 examples/object_detection_sample.py --help
```


接下来，通过将路径传递到您的输入视频，在您自己的视频文件上进行尝试。 您可以提供绝对或相对路径，例如

```bash
$ python3 examples/object_detection_sample.py --input=<absolute-or-relative-path-to-your-video-file>
```
如果您将新视频（例如，examples/data/people-detection.mp4）添加到 openvino_tensorflow 存储库中现有的 `<path-to-openvino_tensorflow-repository>/examples/data` 目录，它将如下所示：

```bash
$ python3 examples/object_detection_sample.py --input=examples/data/people-detection.mp4
```

使用相机作为输入使用```--input=0```。 这里的“0”指的是 /dev/video0 中的摄像头。 如果相机连接到不同的端口，请适当更改。

**注意：** 输入为图像或图像目录的结果将写入输出图像。 对于视频或相机输入，请使用应用程序显示窗口查看结果。

<br/> 

## 分类实例的C++实现

运行 C++ 示例时，我们需要通过源代码构建 TensorFlow 框架，因为示例依赖 TensorFlow 库。

在从源代码开始构建之前，您必须确保已安装 [**prerequisites**](../docs/BUILD.md#1-prerequisites)。

本演示中使用的 TensorFlow 模型由于其大小而未打包在存储库中。 因此将模型下载到“openvino_tensorflow 的克隆存储库”中的“<path-to-openvino_tensorflow-repository>/examples/data”目录并提取文件：
### Linux and macOS
- 下载模型
```bash
$ curl -L "https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz" |
  tar -C <path-to-openvino_tensorflow-repository>/examples/data -xz
```

-运行以下命令，构建 openvino\_tensorflow 及示例：

```bash
$ cd <path-to-openvino_tensorflow-repository>
$ python3 build_tf.py --output_dir <path-to-tensorflow-dir>
$ python3 build_ovtf.py --use_tensorflow_from_location <path-to-tensorflow-dir>
```
关于详细的构建说明，请参阅 [**BUILD.md**](../docs/BUILD.md#build-from-source)。

现在，classification\_sample 的二进制可执行文件已构建完成。更新 LD\_LIBRARY\_PATH 并运行示例：

```bash
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<path-to-openvino_tensorflow-repository>/build_cmake/artifacts/lib:<path-to-openvino_tensorflow-repository>/build_cmake/artifacts/tensorflow
$ ./build_cmake/examples/classification_sample/infer_image
```

### Windows
- 下载以下模型 "https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz"
解压并复制到 <path-to-openvino_tensorflow-repository>\examples\data 

- [从源代码构建TensorFlow](../docs/BUILD.md#TFWindows)  
- 运行以下指令构建openvino_tensorflow及示例
```bash
cd <path-to-openvino_tensorflow-repository>
python build_ovtf.py --use_openvino_from_location="C:\Program Files (x86)\Intel\openvino_2021.4.752" --use_tensorflow_from_location="\path\to\directory\containing\tensorflow\"
```

- 现在，classification_sample的二进制可执行文件已构建完成。更新PATH并运行示例：

```bash
set PATH=%PATH%;<path-to-openvino_tensorflow-repository>\build_cmake\artifacts\lib;<path-to-openvino_tensorflow-repository>\build_cmake\artifacts\tensorflow
build_cmake\examples\classification_sample\Release\infer_image.exe
```

使用此仓库附带的默认示例图像，并且
输出类似于：

```
military uniform (653): 0.834306
mortarboard (668): 0.0218693
academic gown (401): 0.010358
pickelhaube (716): 0.00800814
bulletproof vest (466): 0.00535091
```

在此示例中，我们使用 Admiral Grace Hopper 的默认图像。如您
所见，网络准确发现她穿着军装，得到了
 0.8 的高分。

接下来，通过将路径传递到新图像，在您自己的图像上进行尝试。 您可以提供绝对或相对路径，例如

```bash
$ ./build_cmake/examples/classification_sample/infer_image --image=<absolute-or-relative-path-to-your-image>
```

如果您将新图像（如： my\_image.png）添加到 openvino\_tensorflow 仓库的现有`<path-to-openvino_tensorflow-repository>/examples/data` 目录，将会是这样：

```bash
$ ./build_cmake/examples/classification_sample/infer_image --image=examples/data/my_image.png
```

如要查看各种后端（Intel<sup>®</sup> 硬件)的更多选项，请调用：

```bash
$ ./build_cmake/examples/classification_sample/infer_image --help
```