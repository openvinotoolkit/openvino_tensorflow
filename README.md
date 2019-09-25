
<p align="center">
  <img src="images/ngraph-logo.png">
</p>

# Intel(R) nGraph(TM) Compiler and Runtime for TensorFlow*

This repository contains the code needed to enable Intel(R) nGraph(TM) Compiler and 
runtime engine for TensorFlow. Use it to speed up your TensorFlow training and 
inference workloads. The nGraph Library and runtime suite can also be used to 
customize and deploy Deep Learning inference models that will "just work" with 
a variety of nGraph-enabled backends: CPU, GPU, and custom silicon like the 
[Intel(R) Nervana(TM) NNP](https://itpeernetwork.intel.com/inteldcisummit-artificial-intelligence/).

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/tensorflow/ngraph-bridge/blob/master/LICENSE)
[![Build Status](https://badge.buildkite.com/180bbf814f1a884219849b4838cbda5fa1e03715e494185be3.svg?branch=master)](https://buildkite.com/ngraph/ngtf-cpu-ubuntu)
[![Build Status](https://badge.buildkite.com/ae8d39ef4a18eb238b58ab0637fb97e85b86e85822a08b96d1.svg?branch=master)](https://buildkite.com/ngraph/ngtf-cpu-centos)
[![Build Status](https://badge.buildkite.com/0aeaff43e378d387a160d30083f203f7147f010e3fb15b01d1.svg?branch=master)](https://buildkite.com/ngraph/ngtf-cpu-ubuntu-binary-tf)

## Installation

### Software requirements

|Using pre-built packages| Building from source|
| -----------------------|-------------------|
|Python 3| Python 3|
|TensorFlow v1.14|GCC 4.8 (Ubuntu), Clang/LLVM (macOS)|
|        |`cmake` 3.4 or higher|
|        |Bazel 0.25.2|
|        |`virtualenv` 16.0.0|
|        ||


### Use pre-built packages

 nGraph bridge enables you to use the nGraph Library with TensorFlow.
 Complete the following steps to install a pre-built nGraph bridge for
 TensorFlow.

1. Install TensorFlow:

        pip install -U tensorflow==1.14.0

2. Install `ngraph-tensorflow-bridge`:

        pip install -U ngraph-tensorflow-bridge
   
### Build nGraph from source

To use the latest version of nGraph Library, complete the following steps to
build nGraph bridge from source. 

#### Note to macOS users

The build and installation instructions are identical for Ubuntu 16.04 and
macOS. However, the Python setup may vary across different versions of Mac OS.
TensorFlow build instructions recommend using Homebrew but developers often use
Pyenv. Some users prefer Anaconda/Miniconda. Before building nGraph, ensure that
you can successfully build TensorFlow on macOS with a suitable Python
environment.

The requirements for building nGraph bridge are identical to the requirements for building TensorFlow from source. For more information, review the [TensorFlow configuration] details. 

##### Prepare your build environment

Install the following requirements before building
 `nGraph-bridge`. 
 
TensorFlow uses a build system called "bazel". For the current
 version of `bazel`, use [bazel version].

Install `bazel`:

        wget https://github.com/bazelbuild/bazel/releases/download/0.25.2/bazel-0.25.2-installer-linux-x86_64.sh      
        bash bazel-0.25.2-installer-linux-x86_64.sh --user

Add and source the `bin` path to your `~/.bashrc` file to call
bazel:

        export PATH=$PATH:~/bin
        source ~/.bashrc   

Install `cmake`, `virtualenv`, and `gcc 4.8`.

##### Build an nGraph bridge

Once TensorFlow's dependencies are installed, clone the `ngraph-bridge` repo:

        git clone https://github.com/tensorflow/ngraph-bridge.git
        cd ngraph-bridge
        git checkout v0.19.0-rc1

Run the following Python script to build TensorFlow, nGraph, and the bridge. Use Python 3.5:

        python3 build_ngtf.py --use_prebuilt_tensorflow

When the build finishes, a new `virtualenv` directory is created in `build_cmake/venv-tf-py3`. Build artifacts (i.e., the `ngraph_tensorflow_bridge-<VERSION>-py2.py3-none-manylinux1_x86_64.whl`) are created in the `build_cmake/artifacts` directory. 

Add the following flags to build PlaidML and Intel GPU backends (optional):

        --build_plaidml_backend
        --build_intelgpu_backend

For more build options:
        
        python3 build_ngtf.py --help

Test the installation:
      
        python3 test_ngtf.py

This command runs all C++ and Python unit tests from the `ngraph-bridge` source tree. It also runs various TensorFlow Python tests using nGraph.

To use the `ngraph-tensorflow-bridge`, activate the following `virtualenv` to start using nGraph with TensorFlow. 

        source build_cmake/venv-tf-py3/bin/activate
 
Alternatively, you can also install the TensorFlow and nGraph bridge outside of a `virtualenv`. The Python `whl` files are located in the `build_cmake/artifacts/` and `build_cmake/artifacts/tensorflow` directories, respectively.

Select the help option of `build_ngtf.py` script to learn more about various build options and how to build other backends. 

Verify that `ngraph-bridge` installed correctly:

    python -c "import tensorflow as tf; print('TensorFlow version: ',tf.__version__);\
                import ngraph_bridge; print(ngraph_bridge.__version__)"

This will produce something like this:

    TensorFlow version:  <1.14.0>
    nGraph bridge version: <b'0.14.0'>
    nGraph version used for this build: b'0.18.0+c5d52f1'
    TensorFlow version used for this build: <v1.14.0-...>
    CXX11_ABI flag used for this build: 0
    nGraph bridge built with Grappler: False
    nGraph bridge built with Variables and Optimizers Enablement: False


Note: The version of the ngraph-tensorflow-bridge is not going to be exactly 
the same as when you build from source. This is due to delay in the source 
release and publishing the corresponding Python wheel. 

## Classify an image

Once you have installed nGraph bridge, you can use TensorFlow to train a neural network or run inference using a trained model. 

Use TensorFlow with nGraph to classify an image using a [frozen model]. 

Download the Inception v3 trained model and labels file:

    wget https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz

Extract the frozen model and labels file from the tarball:

    tar xvf inception_v3_2016_08_28_frozen.pb.tar.gz
        
Download the image file: 

    wget https://github.com/tensorflow/tensorflow/raw/master/tensorflow/examples/label_image/data/grace_hopper.jpg

Download the TensorFlow script:

   wget https://github.com/tensorflow/tensorflow/raw/master/tensorflow/examples/label_image/label_image.py
       
Modify the downloaded TensorFlow script to run TensorFlow with nGraph optimizations:

    import ngraph_bridge
    ...
    config = tf.ConfigProto()
    config_ngraph_enabled = ngraph_bridge.update_config(config)
    sess = tf.Session(config=config_ngraph_enabled) 

Run the classification:

    python label_image.py --graph inception_v3_2016_08_28_frozen.pb \
        --image grace_hopper.jpg --input_layer=input \
        --output_layer=InceptionV3/Predictions/Reshape_1 \
        --input_height=299 --input_width=299 \
        --labels imagenet_slim_labels.txt 

This will print the following results:

    military uniform 0.8343056
    mortarboard 0.021869544
    academic gown 0.010358088
    pickelhaube 0.008008157
    bulletproof vest 0.005350913

The above instructions are derived from the [TensorFlow C++ and Python Image Recognition Demo]. 

All of the above commands are available in the [nGraph TensorFlow examples] directory. To classify your own images, modify the `infer_image.py` file in this directory.

### Add runtime options for a CPU backend

Adding runtime options for a CPU backend applies to training and inference.

By default nGraph runs with a CPU backend. To get the best performance of the CPU backend, add the following option:

    OMP_NUM_THREADS=<num_cores> KMP_AFFINITY=granularity=fine,compact,1,0 \ 
    python label_image.py --graph inception_v3_2016_08_28_frozen.pb 
        --image grace_hopper.jpg --input_layer=input \
        --output_layer=InceptionV3/Predictions/Reshape_1 \
        --input_height=299 --input_width=299 \
        --labels imagenet_slim_labels.txt 

Where `<num_cores>` equals the number of cores in your processor. 

#### Measure the time
nGraph is a Just In Time (JIT) compiler meaning that the TensorFlow computation graph is compiled to nGraph during the first instance of the execution. From the second time onwards, the execution speeds up significantly. 

Add the following Python code to measure the computation time:

```python
# Warmup
sess.run(output_operation.outputs[0], {
        input_operation.outputs[0]: t})
# Run
import time
start = time.time()
results = sess.run(output_operation.outputs[0], {
        input_operation.outputs[0]: t
        })      
elapsed = time.time() - start
print('Time elapsed: %f seconds' % elapsed)
```
Observe that the output time runs faster than TensorFlow native (i.e., without nGraph).

#### Add additional backends

You can substitute the default CPU backend with a different backend such as `PLAIDML` or `INTELGPU`. Use the following API:

    ngraph_bridge.set_backend('PLAIDML')

To determine what backends are available on your system, use the following API:

    ngraph_bridge.list_backends()

More detailed examples on how to use ngraph_bridge are located in the [examples] directory.

## Debugging 

During the build, often there are missing configuration steps for building TensorFlow. If you run into build issues, first ensure that you can build TensorFlow. For debugging run time issues, see the instructions provided in the [diagnostics] directory.

## Support

Please submit your questions, feature requests and bug reports via [GitHub issues].

## How to Contribute

We welcome community contributions to nGraph. If you have an idea for how to 
improve it:

* Share your proposal via [GitHub issues].
* Ensure you can build the product and run all the examples with your patch.
* In the case of a larger feature, create a test.
* Submit a [pull request].
* We will review your contribution and, if any additional fixes or
  modifications are necessary, may provide feedback to guide you. When
  accepted, your pull request will be merged to the repository.


## About Intel(R) nGraph(TM)

See the full documentation here:  <https://ngraph.nervanasys.com/docs/latest>
 
[linux-based install instructions on the TensorFlow website]:https://www.tensorflow.org/install/install_linux
[tensorflow]:https://github.com/tensorflow/tensorflow.git
[open-source C++ library, compiler and runtime]: https://ngraph.nervanasys.com/docs/latest/
[Github issues]: https://github.com/tensorflow/ngraph-bridge/issues
[pull request]: https://github.com/tensorflow/ngraph-bridge/pulls
[DSO]:http://csweb.cs.wfu.edu/~torgerse/Kokua/More_SGI/007-2360-010/sgi_html/ch03.html
[bazel version]: https://github.com/bazelbuild/bazel/releases/tag/0.25.2
[TensorFlow configuration]: https://www.tensorflow.org/install/source
[diagnostics]:diagnostics/README.md
[examples]:examples/README.md
[ops]:https://ngraph.nervanasys.com/docs/latest/ops/index.html
[nGraph]:https://github.com/NervanaSystems/ngraph 
[ngraph-bridge]:https://github.com/tensorflow/ngraph-bridge.git
[frozen model]: https://www.tensorflow.org/guide/extend/model_files#freezing
[TensorFlow C++ and Python Image Recognition Demo]: https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/label_image
[nGraph TensorFlow examples]: https://github.com/tensorflow/ngraph-bridge/tree/master/examples
