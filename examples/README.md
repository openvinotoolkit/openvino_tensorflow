<p>English | <a href="./README_cn.md">简体中文</a></p>

# Intel<sup>®</sup> OpenVINO<sup>TM</sup> integration with TensorFlow - C++ and Python Examples

These examples demonstrate how to use **Intel<sup>®</sup> </sup> OpenVINO<sup>TM</sup> integration with Tensorflow** to recognize and detect objects in images and videos.

## Quick Links for examples

  - [Python implementation for classification](#python-implementation-for-classification)
  - [Python implementation for object detection](#python-implementation-for-object-detection)
  - [C++ implementation for classification](#c-implementation-for-classification)
    - [Linux and macOS](#linux-and-macos)
    - [Windows](#windows)

## Demos showcased in the examples

* Classification demo uses Google's Inception v3 model to classify a given image, video, directory of images or camera input.
* Object detection demo uses YOLOv4 model converted from Darknet to detect objects in a given image, video, directory of images or camera input.

## Setup for the examples

Before you proceed to run the examples, you will have to clone the `openvino_tensorflow` repository to your local machine. For this, run the following commands:

```bash
$ git clone https://github.com/openvinotoolkit/openvino_tensorflow.git
$ cd openvino_tensorflow
$ git submodule init
$ git submodule update --recursive
```

<br/> 

## Python implementation for classification

For this example, we assume that you've already:

* Installed TensorFlow on your system
* Installed **Intel<sup>®</sup> </sup> OpenVINO<sup>TM</sup> integration with Tensorflow** on your system

**Note:** Refer to [**this page**](TF_1_x/README.md) for python classification sample using TF1 Inception v3 model.

Refer to [**this page**](https://github.com/openvinotoolkit/openvino_tensorflow#installation) for quick installation using pip.

This demo uses TF Hub image classification model [**InceptionV3**](https://tfhub.dev/google/imagenet/inception_v3/classification/4) and [**ImageNet labels**](https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt) for the possible classifications. In labels file, you'll find 1,000 categories that were used in the Imagenet competition.

Install the pre-requisites
```bash
$ cd <path-to-openvino_tensorflow-repository>/examples
$ pip3 install -r requirements.txt
```
Now, you can run classification sample using the instructions below:


```bash
$ cd <path-to-openvino_tensorflow-repository>
$ python3 examples/classification_sample.py
```

`classification_sample.py` does inference on the [**image**]("https://www.tensorflow.org/images/grace_hopper.jpg") and should output something similar to:

```
military uniform 0.79601693
mortarboard 0.02091024
academic gown 0.014557127
suit 0.009166162
comic book 0.007978318
```
**Note**: use ```--no_show``` flag to disable the application display window. By default the display window is enabled.

In this case, we are using the image of Admiral Grace Hopper. As you can see, the network correctly spots that she's wearing a military uniform, with a high score of 0.79.

Next, try it out by passing the path to your new input. You can provide either absolute or relative path to an image or video or directory of images.
e.g.
```bash
$ python3 examples/classification_sample.py --input=<absolute-or-relative-path-to-your-input>
```
If you add a new image or video (e.g, my_image.png or people-detection.mp4) to the existing `<path-to-openvino_tensorflow-repository>/examples/data` directory in the openvino_tensorflow repository, it will look like this:

```bash
$ python3 examples/classification_sample.py --input=examples/data/my_image.png

or

$ python3 examples/classification_sample.py --input=examples/data/people-detection.mp4
```

For using camera as input use ```--input=0```. Here '0' refers to the camera present at /dev/video0. If the camera is connected to a different port, change it appropriately.

To see more options for various backends (Intel<sup>®</sup> hardware), invoke:
```bash
$ python3 examples/classification_sample.py --help
```

<br/> 

## Python implementation for object detection

For this example, we assume that you've already:

* Installed TensorFlow on your system.
* Installed **Intel<sup>®</sup> </sup> OpenVINO<sup>TM</sup> integration with Tensorflow** on your system


**Note:** Refer to [**this page**](TF_1_x/README.md) for the conversion of yolov4 darknet model and it's python object detection sample.


Refer to [**this page**](https://github.com/openvinotoolkit/openvino_tensorflow#installation) for quick installation using pip.

Install the pre-requisites
```bash
$ cd <path-to-openvino_tensorflow-repository>/examples
$ pip3 install -r requirements.txt
```

The TensorFlow Yolo v4 darknet model used in this demo is not packaged in the repository because of its size. So, follow the instructions below to convert the model from DarkNet to TensorFlow and download the labels and weights to the `<path-to-openvino_tensorflow-repository>/examples/data` directory in your `cloned repo of openvino_tensorflow`:

**For Linux and macOS**

```bash
$ cd <path-to-openvino_tensorflow-repository>/examples
$ chmod +x convert_yolov4.sh
$ ./convert_yolov4.sh
```

**For Windows**

```bash
cd <path-to-openvino_tensorflow-repository>\examples
convert_yolov4.bat
```

Once completed, the `<path-to-openvino_tensorflow-repository>/examples/data` folder will contain following files needed to run the object detection example:

* coco.names
* yolo_v4

Run the object detection example using the instructions below:

```bash
$ cd <path-to-openvino_tensorflow-repository>
$ python3 examples/object_detection_sample.py
```
**Note**: 
* use ```--no_show``` flag to disable the application display window. By default the display window is enabled.
* use ```--rename``` to rename the input image or the directory of the images based on the content of image, for example ***noexiftags-1person-1uniform-intelOpenVINO.jpg***

This uses the default example image that comes with this repository, and should
output something similar as below and the result is written to detections.jpg:

<p align="left">
  <img src="../examples/data/detections.jpg" width="200" height="200"
</p>

In this case, we're using the default image of Admiral Grace Hopper. As you can see, the network detects and draws the bounding box around the person correctly.

Next, try it out on your own image by passing the path to your new input. You can provide either absolute or relative path e.g.

```bash
$ python3 examples/object_detection_sample.py --input=<absolute-or-relative-path-to-your-image>
```

If you add the new image (e.g, my_image.png) to the existing `<path-to-openvino_tensorflow-repository>/examples/data` directory in the openvino_tensorflow repository, it will look like this:

```bash
$ cd <path-to-openvino_tensorflow-repository>
$ python3 examples/object_detection_sample.py --input=examples/data/my_image.png
```

To see more options for various backends (Intel<sup>®</sup> hardware), invoke:
```bash
$ python3 examples/object_detection_sample.py --help
```


Next, try it out on your own video file by passing the path to your input video. You can provide either absolute or relative path e.g.

```bash
$ python3 examples/object_detection_sample.py --input=<absolute-or-relative-path-to-your-video-file>
```
If you add the new video (e.g, examples/data/people-detection.mp4) to the existing `<path-to-openvino_tensorflow-repository>/examples/data` directory in the openvino_tensorflow repository, it will look like this:

```bash
$ python3 examples/object_detection_sample.py --input=examples/data/people-detection.mp4
```

For using camera as input use ```--input=0```. Here '0' refers to the camera present at /dev/video0. If the camera is connected to a different port, change it appropriately.

**Note:** The results with input as an image or a directory of images, are written to output images. For video or camera input, use the application display window for the results.

<br/> 

## C++ implementation for classification

For running C++ examples, we need to build TensorFlow framework from source since examples have a dependency on the TensorFlow libraries.

Before you start building from source, you have to make sure that the [**prerequisites**](../docs/BUILD.md#1-prerequisites) are installed.

The TensorFlow model used in this demo is not packaged in the repo because of its size. So, download the model to the `<path-to-openvino_tensorflow-repository>/examples/data` directory in your `cloned repo of openvino_tensorflow` and extract the file:

### Linux and macOS
- Download model
```bash
$ curl -L "https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz" |
  tar -C <path-to-openvino_tensorflow-repository>/examples/data -xz
```

- Run the following commands to build openvino_tensorflow with samples:

```bash
$ cd <path-to-openvino_tensorflow-repository>
$ python3 build_tf.py --output_dir <path-to-tensorflow-dir>
$ python3 build_ovtf.py --use_tensorflow_from_location <path-to-tensorflow-dir>
```
For detailed build instructions read [**BUILD.md**](../docs/BUILD.md#build-from-source).

- Now, a binary executable for classification_sample is built. Update the LD_LIBRARY_PATH and run the sample:

```bash
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<path-to-openvino_tensorflow-repository>/build_cmake/artifacts/lib:<path-to-openvino_tensorflow-repository>/build_cmake/artifacts/tensorflow
$ ./build_cmake/examples/classification_sample/infer_image
```

### Windows
- Download following model "https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz"
uncompress it and  copy it to <path-to-openvino_tensorflow-repository>\examples\data 

- [Build TensorFlow from source](../docs/BUILD.md#TFWindows)  
- Run the following commands to build openvino_tensorflow with samples
```bash
cd <path-to-openvino_tensorflow-repository>
python build_ovtf.py --use_openvino_from_location="C:\Program Files (x86)\Intel\openvino_2021.4.752" --use_tensorflow_from_location="\path\to\directory\containing\tensorflow\"
```

- Now, a binary executable for classification_sample is built. Update the PATH and run the sample:

```bash
set PATH=%PATH%;<path-to-openvino_tensorflow-repository>\build_cmake\artifacts\lib;<path-to-openvino_tensorflow-repository>\build_cmake\artifacts\tensorflow
build_cmake\examples\classification_sample\Release\infer_image.exe
```

This uses the default example image that comes with this repository, and should
output something similar to:

```
military uniform (653): 0.834306
mortarboard (668): 0.0218693
academic gown (401): 0.010358
pickelhaube (716): 0.00800814
bulletproof vest (466): 0.00535091
```

In this case, we're using the default image of Admiral Grace Hopper. As you can
see the network correctly spots she's wearing a military uniform, with a high
score of 0.8.

Next, try it out on your own image by passing the path to your new input. You can provide either absolute or relative path e.g.


```bash
$ ./build_cmake/examples/classification_sample/infer_image --image=<absolute-or-relative-path-to-your-image>
```

If you add the new image (e.g, my_image.png) to the existing `<path-to-openvino_tensorflow-repository>/examples/data` directory in the openvino_tensorflow repository, it will look like this:

```bash
$ ./build_cmake/examples/classification_sample/infer_image --image=examples/data/my_image.png
```

To see more options for various backends (Intel<sup>®</sup> hardware), invoke:
```bash
$ ./build_cmake/examples/classification_sample/infer_image --help
```

<br/>

**Note**: In the above samples a warm-up run is executed first and then inference time is measured on the subsequent runs. The execution time of first run is in general higher compared to the next runs as it includes many one-time graph transformations and optimizations steps.