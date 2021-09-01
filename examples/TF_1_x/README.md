# Intel<sup>®</sup> OpenVINO<sup>TM</sup> integration with TensorFlow Python Object Detection Example

These examples demonstrate how to use **Intel<sup>®</sup> </sup> OpenVINO<sup>TM</sup> integration with Tensorflow** to recognize and detect objects in images and videos.

## Demos showcased in the examples

 Object detection demo uses YOLOv3 darknet model to detect objects in a given image, video, directory and camera input.

## Setup for the examples

Before you proceed to run the examples, you will have to clone the `openvino_tensorflow` repository to your local machine. For this, run the following commands:

```bash
$ git clone https://github.com/openvinotoolkit/openvino_tensorflow.git
$ cd openvino_tensorflow
$ git submodule init
$ git submodule update --recursive
```

## Python implementation for object detection

For this example, we assume that you've already:

* Installed TensorFlow on your system
* Installed **Intel<sup>®</sup> </sup> OpenVINO<sup>TM</sup> integration with Tensorflow** on your system

Refer to [**this page**](https://github.com/openvinotoolkit/openvino_tensorflow#installation) for a quick install with pip.

Install the pre-requisites
```bash
$ cd <path-to-openvino_tensorflow-repository>/examples
$ pip3 install -r requirements.txt
```

The TensorFlow Yolo v3 model used in this demo is not packaged in the repository because of its size. So, follow the instructions below to convert the model from DarkNet to TensorFlow and download the labels and weights to the `data` directory in your `cloned repo of openvino_tensorflow`:

Please note: The instructions below should not be executed in an active virtual environment. The convert_yolov3.sh script activates a python virtual environment for conversion.

```bash
$ cd <path-to-openvino_tensorflow-repository>/examples/TF_1_x
$ chmod +x convert_yolov3.sh
$ ./convert_yolov3.sh
```

Once completed, the data folder will contain following files needed to run the object detection example:

* coco.names
* yolo_v3_darknet_1.pb

Run the object detection example using the instructions below:

```bash
$ cd <path-to-openvino_tensorflow-repository>
$ python3 examples/TF_1_x/object_detection_sample.py
```

This uses the default example image that comes with this repository, and should
output something similar as below:

<p align="left">
  <img src="../data/detections.jpg" width="200" height="200"
</p>

In this case, we're using the default image of Admiral Grace Hopper. As you can see, the network detects and draws the bounding box around the person correctly.

Next, try it out on your own image by passing the --image=argument, where argument is the path to your new image. You can provide either absolute or relative path in the argument e.g.

```bash
$ python3 examples/TF_1_x/object_detection_sample.py --image=<absolute-or-relative-path-to-your-image>
```

If you add the new image (e.g, my_image.png) to the existing data directory in the openvino_tensorflow repository, it will look like this:

```bash
$ python3 examples/TF_1_x/object_detection_sample.py --image=examples/data/my_image.png
```

To see more options for various backends (Intel<sup>®</sup> hardware), invoke:
```bash
$ python3 examples/TF_1_x/object_detection_sample.py --help
```

Next, try it out on your own video file by passing the --input=argument, where argument is the path to your input video. You can provide either absolute or relative path in the argument  e.g.

```bash
$ python3 examples/TF_1_x/object_detection_sample.py --input=<absolute-or-relative-path-to-your-video-file>
```
If you add the new video (e.g, examples/data/people-detection.mp4) to the existing data directory in the openvino_tensorflow repository, it will look like this:

```bash
$ python3 examples/TF_1_x/object_detection_sample.py --input=examples/data/people-detection.mp4
```

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
$ python3 examples/TF_1_x/object_detection_sample.py --input_height 160 --input_width 160 --graph "examples/TF_1_x/yolo_v3_160.pb" --input_layer "inputs" --output_layer "output_boxes" --labels "examples/data/coco.names"
```
