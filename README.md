
<p align="center">
  <img src="images/ngraph-logo.png">
</p>

# Intel® nGraph™ Compiler and Runtime for TensorFlow*

This repository contains the code needed to enable Intel(R) nGraph(TM) Compiler and 
runtime engine for TensorFlow. Use it to speed up your TensorFlow training and 
inference workloads. The nGraph Library and runtime suite can also be used to 
customize and deploy Deep Learning inference models that will "just work" with 
a variety of nGraph-enabled backends: CPU, and custom silicon like the 
[Intel(R) Nervana(TM) NNP](https://itpeernetwork.intel.com/inteldcisummit-artificial-intelligence/).

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/tensorflow/ngraph-bridge/blob/master/LICENSE)
[![Build Status](https://badge.buildkite.com/180bbf814f1a884219849b4838cbda5fa1e03715e494185be3.svg?branch=master)](https://buildkite.com/ngraph/cpu)
[![Build Status](https://badge.buildkite.com/ae8d39ef4a18eb238b58ab0637fb97e85b86e85822a08b96d1.svg?branch=master)](https://buildkite.com/ngraph/models-cpu)
[![Build Status](https://badge.buildkite.com/0aeaff43e378d387a160d30083f203f7147f010e3fb15b01d1.svg?branch=master)](https://buildkite.com/ngraph/cpu-intel-tf)


#### *** This repository is currently undergoing heavy refactoring for optimization of inference use-cases. If you are looking for the latest stable baseline, please use the following tag: [v0.22.0-rc4](https://github.com/tensorflow/ngraph-bridge/tree/v0.22.0-rc4) ***

## Installation

### Requirements

|Using pre-built packages| Building from source|
| -----------------------|-------------------|
|Python 3| Python 3|
|TensorFlow v2.2.0|GCC 7.5 (Ubuntu), Clang/LLVM (macOS)|
|        |`cmake` 3.4 or higher|
|        |Bazelisk|
|        |`virtualenv` 16.0.0+|
|        |`patchelf`|

### Use pre-built packages

 nGraph bridge enables you to use the nGraph Library with TensorFlow.
 Complete the following steps to install a pre-built nGraph bridge for
 TensorFlow.

1. Ensure the following pip version is being used:

        pip install --upgrade pip==19.3.1

2. Install TensorFlow:

        pip install -U tensorflow==1.14.0

3. Install `ngraph-tensorflow-bridge`:

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

The requirements for building nGraph bridge are identical to the requirements for 
building TensorFlow from source. For more information, review the [TensorFlow configuration] 
details. 

##### Prepare your build environment

Install the following requirements before building the `ngraph-bridge`. 

TensorFlow uses a build system called "bazel". For the current version of `bazel`, 
use [bazel version].

Install `bazelisk`:

        wget https://github.com/bazelbuild/bazelisk/releases/download/v1.5.0/bazelisk-linux-amd64
        mv bazelisk-linux-amd64 ~/bin/bazel
        chmod +x ~/bin/bazel

Add and source the `bin` path to your `~/.bashrc` file to call
bazel:

        export PATH=$PATH:~/bin
        source ~/.bashrc   

Install `cmake`, `virtualenv`, and `gcc`.

##### Build an nGraph bridge

Once TensorFlow's dependencies are installed, clone the `ngraph-bridge` repo:

        git clone https://github.com/tensorflow/ngraph-bridge.git
        cd ngraph-bridge

Run the following Python script to build TensorFlow, nGraph, and the bridge. Use Python 3:

        python3 build_ngtf.py --use_prebuilt_tensorflow

When the build finishes, a new `virtualenv` directory is created in `build_cmake/venv-tf-py3`. Build artifacts (i.e., the `ngraph_tensorflow_bridge-<VERSION>-py2.py3-none-manylinux1_x86_64.whl`) are created in the `build_cmake/artifacts` directory. 

For more build options:
        
        python3 build_ngtf.py --help

To use the `ngraph-tensorflow-bridge`, activate the following `virtualenv` to start using nGraph with TensorFlow. 

        source build_cmake/venv-tf-py3/bin/activate
 
Alternatively, you can also install the TensorFlow and nGraph bridge outside of a `virtualenv`. The Python `whl` files are located in the `build_cmake/artifacts/` and `build_cmake/artifacts/tensorflow` directories, respectively.

Select the help option of `build_ngtf.py` script to learn more about various build options. 

Verify that `ngraph-bridge` installed correctly:

    python -c "import tensorflow as tf; print('TensorFlow version: ',tf.__version__);\
                import ngraph_bridge; print(ngraph_bridge.__version__)"

This will produce something like this:

        TensorFlow version:  2.2.0
        nGraph bridge version: b'0.22.0-rc3'
        nGraph version used for this build: b'0.28.0-rc.1+d2cd873'
        TensorFlow version used for this build: v2.2.0-0-2b96f3662b
        CXX11_ABI flag used for this build: 1
        nGraph bridge built with Grappler: False


Note: The version of the ngraph-tensorflow-bridge is not going to be exactly 
the same as when you build from source. This is due to delay in the source 
release and publishing the corresponding Python wheel.

Test the installation:

        python3 test_ngtf.py

This command runs all C++ and Python unit tests from the `ngraph-bridge` source tree. It also runs various TensorFlow Python tests using nGraph.

### Build and run nGraph in Docker

A shell script and dockerfiles are provided in the [`tools`](/tools) directory for easy setup in a Docker container. 
See [this README](/tools) if you want to use Docker.

## Classify an image

Once you have installed nGraph bridge, you can use TensorFlow to train a neural network or run inference using a trained model.
The only change required to a script is adding

    import ngraph_bridge

Use `infer_image.py` in the [examples] directory to classify an image.

Note: The script downloads the inceptionV3 model and sample image.

    python examples/infer_image.py

This will print the following results:

    military uniform 0.8343056
    mortarboard 0.021869544
    academic gown 0.010358088
    pickelhaube 0.008008157
    bulletproof vest 0.005350913

To classify your own images, modify the `infer_image.py` file.

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

You can substitute the default CPU backend with a different backend. 
Use the following API:

    ngraph_bridge.set_backend('backend_name')

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


## About Intel® nGraph™

See the [full documentation] here.

[TensorFlow]:https://github.com/tensorflow/tensorflow.git
[Github issues]: https://github.com/tensorflow/ngraph-bridge/issues
[pull request]: https://github.com/tensorflow/ngraph-bridge/pulls
[bazel version]: https://github.com/bazelbuild/bazel/releases/tag/0.25.2
[TensorFlow configuration]: https://www.tensorflow.org/install/source
[diagnostics]:diagnostics/README.md
[examples]:examples/README.md
[nGraph]:https://docs.openvinotoolkit.org/latest/openvino_docs_nGraph_DG_Introduction.html
[full documentation]:https://docs.openvinotoolkit.org/latest/openvino_docs_nGraph_DG_Introduction.html
[frozen model]: https://www.tensorflow.org/guide/extend/model_files#freezing
[TensorFlow C++ and Python Image Recognition Demo]: https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/label_image
