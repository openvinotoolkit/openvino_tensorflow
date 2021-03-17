# Intel<sup>(R)</sup> Openvino<sup>TM</sup> Add-on for TensorFlow C++ and Python Classification Demo

These examples show how to use Intel<sup>(R)</sup> Openvino<sup>TM</sup> Add-on for Tensorflow to recognize objects in images in C++ and Python.

## Description

These demos use Google Inception V3 model to classify image that is passed in on the command line. 

## Prerequisites

This examples assume that you have already  

* Installed TensorFlow on your system 
* Installed OpenVINO Add-on for TensorFlow on your system
* You've verified that both work with the following command and verified the output 

Verify that `openvino-tensorflow-addon` installed correctly:

    python -c "import tensorflow as tf; print('TensorFlow version: ',tf.__version__);\
                import openvino_tensorflow; print(openvino_tensorflow.__version__)"

This will produce something like this:

        TensorFlow version:  2.2.2
        OpenVINO Add-on for TensorFlow version: b'0.5.0'
        OpenVINO version used for this build: b'2021.2'
        TensorFlow version used for this build: v2.2.2
        CXX11_ABI flag used for this build: 1
        OpenVINO Add-on built with Grappler: False



## Download TensorFlow model 

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

## To build/install/run C++ implementation 

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



