# Intel<sup>(R)</sup> Openvino<sup>TM</sup> integration with TensorFlow C++ and Python Demos

These examples show how to use Intel<sup>(R)</sup> Openvino<sup>TM</sup> integration with Tensorflow to recognize objects in images in C++ & Python and object detection in Python.

## Description

The classification demo uses Google Inception V3 model to classify image that is passed in on the command line and object detection demo uses Yolo V3 model to detect objects in an image.

## Python implementation for classification 

This example assumes that you have already:  

* Installed TensorFlow on your system 
* Installed OpenVINO integration with Tensorflow on your system
* Please refer to [**this**](https://github.com/openvinotoolkit/openvino_tensorflow#use-pre-built-packages) for more details about pre-built packages.

The TensorFlow `GraphDef` that contains the model definition and weights is not packaged in the repo because of its size. Instead, you must first download the file to the `data` directory in the source tree:

```bash
$ curl -L "https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz" |
  tar -C <path-to-openvino_tensorflow>/examples/data -xz
```

Once extracted, the data folder will have two new files:

* imagenet_slim_labels.txt
* inception_v3_2016_08_28_frozen.pb

See the labels file in the data directory for the possible
classifications, which are the 1,000 categories used in the Imagenet
competition.
classification_sample.py is a python implementation that provides code corresponding to the C++ code here and could be easier to add visualization or debug code.

```bash
$ cd <path-to-openvino_tensorflow>
$ python examples/classification_sample.py
```

This uses the default example image that is shipped with this repo, and should
output something similar as below:

```
military uniform (653): 0.834306
mortarboard (668): 0.0218693
academic gown (401): 0.010358
pickelhaube (716): 0.00800814
bulletproof vest (466): 0.00535091
```

In this case, we're using the default image of Admiral Grace Hopper, and you can
see the network correctly spots she's wearing a military uniform, with a high
score of 0.8.

Next, try it out on your own images by supplying the --image= argument, e.g.

```bash
$ python examples/classification_sample.py --image=my_image.png
```
## Python implementation for object detection

This example assumes that you have already:  

* Installed TensorFlow on your system 
* Installed OpenVINO integration with Tensorflow on your system
* Please refer to [**this**](https://github.com/openvinotoolkit/openvino_tensorflow#use-pre-built-packages) for more details about pre-built packages.


The TensorFlow `GraphDef` that contains the Yolo V3 model definition and weights is not packaged in the repo because of its size. Instead, you must first follow below instructions to convert the model from DarkNet to TensorFlow and download the labels and weights in the `data` directory in the source tree:

```bash
$ cd <path-to-openvino_tensorflow>/examples/data
$ git clone https://github.com/mystic123/tensorflow-yolo-v3.git
$ cd tensorflow-yolo-v3
$ git checkout ed60b90
$ wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
$ wget https://pjreddie.com/media/files/yolov3.weights
$ python3 convert_weights_pb.py --class_names coco.names --data_format NHWC --weights_file yolov3.weights
```

Once completed, the data folder will have following files needed to run the object detection sample:

* coco.names
* frozen_darknet_yolov3_model.pb

Run the object detection sample:

```bash
$ cd <path-to-openvino_tensorflow>
$ python examples/object_detection_sample.py
```

This uses the default example image that is shipped with this repo, and should
output something similar as below:
```
person with bounding box
label:person
score:99% 
``` 
In this case, we're using the default image of Admiral Grace Hopper, and you can see the network detects and draws the bounding box around the person correctly.

Next, try it out on your own images by supplying the --image= argument, e.g.

```bash
$ python examples/object_detection_sample.py  --image=my_image.png
```
## C++ Implementation to build/install/run for classification

Assuming main tensorflow framework is already built using build_tf.py , run this command to build openvino_tensorflow with samples:

```bash
$ cd <path-to-openvino_tensorflow>
$ python3 build_ovtf.py --use_tensorflow_from_location <path-to-dir-with-tensorflow-artifacts>
```
For detailed build instructions please read [**this**](https://github.com/openvinotoolkit/openvino_tensorflow#build-from-source).

That should build a binary executable for classification_sample. Update the LD_LIBRARY_PATH and run the sample:

```bash
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<path-to-openvino_tensorflow>/build_cmake/artifacts/lib:<path-to-openvino_tensorflow>/build_cmake/artifacts/tensorflow
$ ./build_cmake/examples/classification_sample/infer_image
```

This uses the default example image that is shipped with this repo, and should
output something similar as below:

```
military uniform (653): 0.834306
mortarboard (668): 0.0218693
academic gown (401): 0.010358
pickelhaube (716): 0.00800814
bulletproof vest (466): 0.00535091
```

In this case, we're using the default image of Admiral Grace Hopper, and you can
see the network correctly spots she's wearing a military uniform, with a high
score of 0.8.

Next, try it out on your own images by supplying the --image= argument, e.g.

```bash
$ ./build_cmake/examples/classification_sample/infer_image --image=my_image.png
```


