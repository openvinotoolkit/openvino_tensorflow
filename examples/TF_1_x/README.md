# Intel<sup>®</sup> OpenVINO<sup>TM</sup> integration with TensorFlow - Python Examples

These examples demonstrate how to use **Intel<sup>®</sup> </sup> OpenVINO<sup>TM</sup> integration with Tensorflow** to recognize and detect objects in images and videos.

## Quick Links for examples

* [Python Implementation for classification](#python-implementation-for-classification)
* [Python implementation for object detection](#python-implementation-for-object-detection)

## Demo showcased in this example
 Classification demo uses Google's Inception v3 model to classify a given image, video, directory of images or camera input.
 Object detection using YOLOv3 model converted from Darknet to detect objects in a given image, video, directory or camera input.

## Setup for the examples

Before you proceed to run the examples, you will have to clone the `openvino_tensorflow` repository to your local machine. For this, run the following commands:

```bash
$ git clone https://github.com/openvinotoolkit/openvino_tensorflow.git
$ cd openvino_tensorflow
$ git submodule init
$ git submodule update --recursive
```
## Python implementation for classification

For this example, we assume that you've already:

* Installed TensorFlow on your system
* Installed **Intel<sup>®</sup> </sup> OpenVINO<sup>TM</sup> integration with Tensorflow** on your system

Refer to [**this page**](https://github.com/openvinotoolkit/openvino_tensorflow#installation) for quick installation using pip.

The TensorFlow model used in this demo is not packaged in the repo because of its size. So, download the model to the `<path-to-openvino_tensorflow-repository>/examples/data` directory in your `cloned repo of openvino_tensorflow` and extract the file:

```bash
$ curl -L "https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz" |
  tar -C <path-to-openvino_tensorflow-repository>/examples/data -xz
```

Once extracted, the `<path-to-openvino_tensorflow-repository>/examples/data` folder will contain two new files:

* imagenet_slim_labels.txt
* inception_v3_2016_08_28_frozen.pb

Open `imagenet_slim_labels.txt` to read the labels in the `<path-to-openvino_tensorflow-repository>/examples/data` directory for the possible classifications. In the .txt file, you'll find 1,000 categories that were used in the Imagenet competition.

Install the pre-requisites
```bash
$ cd <path-to-openvino_tensorflow-repository>/examples
$ pip3 install -r requirements.txt
```
Now, you can run classification sample using the instructions below:


```bash
$ cd <path-to-openvino_tensorflow-repository>
$ python3 examples/TF_1_x/classification_sample.py
```

`classification_sample.py` does inference on the default example image that comes with this repository and should output something similar to:

```
military uniform (653): 0.834306
mortarboard (668): 0.0218693
academic gown (401): 0.010358
pickelhaube (716): 0.00800814
bulletproof vest (466): 0.00535091
```
**Note**: use ```--no_show``` flag to disable the application display window. By default the display window is enabled.

In this case, we are using the default image of Admiral Grace Hopper. As you can see, the network correctly spots that she's wearing a military uniform, with a high score of 0.8.

Next, try it out by passing the path to your new input. You can provide either absolute or relative path to an image or video or directory of images.
e.g.
```bash
$ python3 examples/TF_1_x/classification_sample.py --input=<absolute-or-relative-path-to-your-input>
```
If you add the new image (e.g, my_image.png) to the existing `<path-to-openvino_tensorflow-repository>/examples/data` directory in the openvino_tensorflow repository, it will look like this:

```bash
$ python3 examples/TF_1_x/classification_sample.py --input=examples/data/my_image.png
```


To see more options for various backends (Intel<sup>®</sup> hardware), invoke:
```bash
$ python3 examples/TF_1_x/classification_sample.py --help
```

Next, try it out on your own video file by passing the path to your input video. You can provide either absolute or relative path e.g.

```bash
$ python3 examples/TF_1_x/classification_sample.py --input=<absolute-or-relative-path-to-your-video-file>
```
If you add the new video (e.g, examples/data/people-detection.mp4) to the existing `<path-to-openvino_tensorflow-repository>/examples/data` directory in the openvino_tensorflow repository, it will look like this:

```bash
$ python3 examples/TF_1_x/classification_sample.py --input=examples/data/people-detection.mp4
```
For using camera as input use ```--input=0```. Here '0' refers to the camera present at /dev/video0. If the camera is connected to a different port, change it appropriately.

## Python implementation for object detection

For this example, we assume that you've already:

* Installed TensorFlow on your system
* Installed **Intel<sup>®</sup> </sup> OpenVINO<sup>TM</sup> integration with Tensorflow** on your system

Refer to [**this page**](https://github.com/openvinotoolkit/openvino_tensorflow#installation) for quick installation using pip.

Install the pre-requisites
```bash
$ cd <path-to-openvino_tensorflow-repository>/examples
$ pip3 install -r requirements.txt
```

The TensorFlow Yolo v3 model used in this demo is not packaged in the repository because of its size. So, follow the instructions below to convert the model from DarkNet to TensorFlow and download the labels and weights to the `<path-to-openvino_tensorflow-repository>/examples/data` directory in your `cloned repo of openvino_tensorflow`:

Please note: The instructions below should not be executed in an active virtual environment. The convert_yolov3.sh script activates a python virtual environment for conversion.

```bash
$ cd <path-to-openvino_tensorflow-repository>/examples/TF_1_x
$ chmod +x convert_yolov3.sh
$ ./convert_yolov3.sh
```

Once completed, the `<path-to-openvino_tensorflow-repository>/examples/data` folder will contain following files needed to run the object detection example:

* coco.names
* yolo_v3_darknet_1.pb

Run the object detection example using the instructions below:

```bash
$ cd <path-to-openvino_tensorflow-repository>
$ python3 examples/TF_1_x/object_detection_sample.py
```

**Note**: use ```--no_show``` flag to disable the application display window. By default the display window is enabled.

This uses the default example image that comes with this repository, and should
output something similar as below and the result is written to detections.jpg:


<p align="left">
  <img src="../data/detections.jpg" width="200" height="200"
</p>

In this case, we're using the default image of Admiral Grace Hopper. As you can see, the network detects and draws the bounding box around the person correctly.

Next, try it out on your own image by passing the path to your new image. You can provide either absolute or relative path e.g.

```bash
$ python3 examples/TF_1_x/object_detection_sample.py --image=<absolute-or-relative-path-to-your-image>
```

If you add the new image (e.g, my_image.png) to the existing `<path-to-openvino_tensorflow-repository>/examples/data` directory in the openvino_tensorflow repository, it will look like this:

```bash
$ python3 examples/TF_1_x/object_detection_sample.py --image=examples/data/my_image.png
```

To see more options for various backends (Intel<sup>®</sup> hardware), invoke:
```bash
$ python3 examples/TF_1_x/object_detection_sample.py --help
```

Next, try it out on your own video file by passing the path to your input video. You can provide either absolute or relative path  e.g.

```bash
$ python3 examples/TF_1_x/object_detection_sample.py --input=<absolute-or-relative-path-to-your-video-file>
```
If you add the new video (e.g, examples/data/people-detection.mp4) to the existing `<path-to-openvino_tensorflow-repository>/examples/data` directory in the openvino_tensorflow repository, it will look like this:

```bash
$ python3 examples/TF_1_x/object_detection_sample.py --input=examples/data/people-detection.mp4
```

For using camera as input use ```--input=0```. Here '0' refers to the camera present at /dev/video0. If the camera is connected to a different port, change it appropriately.

**Note:** The results with input as an image or a directory of images, are written to output images. For video or camera input use the application display window for the results.

To try on the yolo_v3_160 model for faster inference follow the below steps

Please note: The instructions below should not be executed in an active virtual environment. The convert_yolov3_160.sh script activates a python virtual environment for conversion.

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
