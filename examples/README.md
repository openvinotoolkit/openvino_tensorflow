# Intel<sup>®</sup> OpenVINO<sup>TM</sup> integration with TensorFlow - C++ and Python Demos

These examples demonstrate how to use **Intel<sup>®</sup> </sup> OpenVINO<sup>TM</sup> integration with Tensorflow** to recognize and detect objects in images.

## AI models leveraged in the demos

* Classification demo uses Google's Inception v3 model to classify a given image.
* Object detection demo uses YOLOv3 model to detect objects in a given image.

## Setup for the examples

Before you procede to running the examples, you will have to clone the `openvino_tensorflow repository` to your local machine. For this, run the following commands:  

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

Refer to [**this page**](https://github.com/openvinotoolkit/openvino_tensorflow) for a quick install with pip. 

TensorFlow's [`GraphDef`](https://stackoverflow.com/questions/47059848/difference-between-tensorflows-graph-and-graphdef) which contains the model definition and weights is not packaged in the repo because of its size. So, download the model to the `data` directory in your `cloned repo of openvino_tensorflow` and extract the file:

```bash
$ curl -L "https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz" |
  tar -C <path-to-your-cloned-openvino_tensorflow-repository>/examples/data -xz
```

Once extracted, the data folder will have two new files:

* imagenet_slim_labels.txt
* inception_v3_2016_08_28_frozen.pb

Open `imagenet_slim_labels.txt` to read the labels in the data directory for the possible classifications. In the .txt file, you'll find 1,000 categories that were used in the Imagenet competition. 

Now, you can run classification sample using the below instructions:


```bash
$ cd <path-to-your-cloned-openvino_tensorflow-repository>
$ python3 examples/classification_sample.py
```

`classification_sample.py` does inference on the default example image that comes with this repository and should output something similar to:

```
military uniform (653): 0.834306
mortarboard (668): 0.0218693
academic gown (401): 0.010358
pickelhaube (716): 0.00800814
bulletproof vest (466): 0.00535091
```

In this case, we're using the default image of Admiral Grace Hopper. As you can see, the network correctly spots that she's wearing a military uniform, with a high score of 0.8.

Next, try it out on your own image by passing the --image= argument to a directory where your new image resides. Python accepts both absolute and relative paths and it is up to you which one you give in the argument  e.g.

```bash
$ python3 examples/classification_sample.py --image=<absolute-or-relative-path-to-your-image>/my_image.png
```
If you add the new image to the existing data directory in the openvino_tensorflow repository, it will look like this:

```bash
$ python3 examples/classification_sample.py --image=example/data/my_image.png
```

To see more options for various backends (Intel<sup>®</sup> hardware), invoke:
```bash
$ python3 examples/classification_sample.py --help
```
## Python implementation for object detection

For this example, we assume that you've already:  

* Installed TensorFlow on your system
* Installed **Intel<sup>®</sup> </sup> OpenVINO<sup>TM</sup> integration with Tensorflow** on your system

Refer to [**this page**](https://github.com/openvinotoolkit/openvino_tensorflow) for a quick install with pip. 


The TensorFlow `GraphDef` that contains the Yolo V3 model definition and weights is not packaged in the repository because of its size. So, follow the instructions below to convert the model from DarkNet to TensorFlow and download the labels and weights to the `data` directory in your `cloned repo of openvino_tensorflow`:

```bash
$ cd <path-to-your-cloned-openvino_tensorflow-repository>/examples/data
$ git clone https://github.com/mystic123/tensorflow-yolo-v3.git
$ cd tensorflow-yolo-v3
$ git checkout ed60b90
$ wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
$ wget https://pjreddie.com/media/files/yolov3.weights
$ python3 convert_weights_pb.py --class_names coco.names --data_format NHWC --weights_file yolov3.weights
```

Once completed, the data folder will contain following files needed to run the object detection example:

* coco.names
* frozen_darknet_yolov3_model.pb

Run the object detection example using the instructions:

```bash
$ cd <path-to-your-cloned-openvino_tensorflow-repository>
$ python3 examples/object_detection_sample.py
```

This uses the default example image that comes with this repository, and should
output something similar as below:

<p align="left">
  <img src="../examples/data/detections.jpg" width="200" height="200" 
</p>

In this case, we're using the default image of Admiral Grace Hopper. As you can see, the network detects and draws the bounding box around the person correctly.

Next, try it out on your own image by passing the --image= argument, to a directory where your new image resides. Python accepts both absolute and relative paths and it is up to you which one you give in the argument e.g.

```bash
$ python3 examples/object_detection_sample.py --image=<absolute-or-relative-path-to-your-image>/my_image.png
```

If you add the new image to the existing data directory in the openvino_tensorflow repository, it will look like this:

```bash
$ python3 examples/object_detection_sample.py --image=example/data/my_image.png
```

To see more options for various backends (Intel<sup>®</sup> hardware), invoke:
```bash
$ python3 examples/object_detection_sample.py --help
```


## C++ Implementation for classification 

For running C++ examples, we need to build a TensorFlow framework from source since examples have a dependency on the TensorFlow libraries. 

Before you start building from source, you have to make sure that you installed the following dependencies:

* Python 3.6, 3.7, or 3.8
* GCC 7.5 (Ubuntu 18.04)
* Cmake 3.14 or higher 
* Bazelisk v1.7.5 
* Virtualenv 16.0.0 or higher
* Patchelf 0.9

Run the following commands to build openvino_tensorflow with samples:

```bash
$ cd <path-to-your-cloned-openvino_tensorflow-repository>
$ python3 build_tf.py --output_dir <path-to-tensorflow-dir>
$ python3 build_ovtf.py --use_tensorflow_from_location <path-to-tensorflow-dir>
```
For detailed build instructions read [**this page**](https://github.com/openvinotoolkit/openvino_tensorflow/blob/master/docs/BUILD.md).

Now, a binary executable for classification_sample should be built. Update the LD_LIBRARY_PATH and run the sample:

```bash
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<path-to-your-cloned-openvino_tensorflow-repository>/build_cmake/artifacts/lib:<path-to-your-cloned-openvino_tensorflow-repository>/build_cmake/artifacts/tensorflow
$ ./build_cmake/examples/classification_sample/infer_image
```

This uses the default example image that comes with this repository, and should
output something similar as below:

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

Next, try it out on your own images by supplying the --image= argument, e.g.

```bash
$ ./build_cmake/examples/classification_sample/infer_image --image=<absolute-or-relative-path-to-your-image>/my_image.png
```

If you add the new image to the existing data directory in the openvino_tensorflow repository, it will look like this:

```bash
$ ./build_cmake/examples/classification_sample/infer_image --image=example/data/my_image.png
```

To see more options for various backends (Intel<sup>®</sup> hardware), invoke:
```bash
$ ./build_cmake/examples/classification_sample/infer_image --help
```
