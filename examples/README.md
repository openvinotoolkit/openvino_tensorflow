# Intel<sup>(R)</sup> Openvino<sup>TM</sup> TensorFlow Add-on C++ and Python Classification Demo

This example shows how to use Intel<sup>(R)</sup> Openvino<sup>TM</sup> Tensorflow Add-on to recognize objects in images in C++ and Python.

## Description

This demo uses Google Inception V3 model to classify image that is passed in on the command line.

## To build/install/run

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

Assuming main tensorflow framework is already built using build_tf.py , run this command to build openvino_tensorflow with samples:

```bash
$ cd <path-to-openvino_tensorflow>
$ python3 build_ovtf.py --use_tensorflow_from_location <path-to-dir-with-tensorflow-artifacts>
```

That should build a binary executable that you can then run like this:

```bash
$ ./build_cmake/examples/classification_sample/infer_image
```

This uses the default example image that ships with this repo, and should
output something similar to this:

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

## Python implementation

classification_sample.py is a python implementation that provides code corresponding to the C++ code here and could be easier to add visualization or debug code.

```bash
$ cd <path-to-openvino_tensorflow>
$ source build_cmake/venv-tf-py3/bin/activate
$ python examples/classification_sample.py
```

And get result similar to this:
```
military uniform (653): 0.834306
mortarboard (668): 0.0218693
academic gown (401): 0.010358
pickelhaube (716): 0.00800814
bulletproof vest (466): 0.00535091
```

