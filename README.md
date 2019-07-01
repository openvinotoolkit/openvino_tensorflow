
<p align="center">
  <img src="images/ngraph-logo.png">
</p>

# Intel(R) nGraph(TM) Compiler and runtime for TensorFlow*

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

*   [Build using Linux](#linux-instructions)
*   [Build using OS X](#using-os-x)
*   [Debugging](#debugging)
*   [Support](#support)
*   [How to Contribute](#how-to-contribute)


## Linux instructions


### Option 1: Use a pre-built nGraph-TensorFlow bridge 

1. You can install TensorFlow and nGraph using `virtualenv` or in the system Python location. 

2. Install TensorFlow v1.14.0:

        pip install -U tensorflow==1.14.0

3. Install nGraph-TensorFlow bridge:

        pip install -U ngraph-tensorflow-bridge
   
### Option 2: Build nGraph bridge with binary TensorFlow installation

#### Prepare the build environment

The installation prerequisites are the same as described in the TensorFlow 
[prepare environment] for linux.

1. TensorFlow uses a build system called "bazel". The version of the `bazel` is determined by the TensorFlow team. For the current version, use [bazel version].

        wget https://github.com/bazelbuild/bazel/releases/download/0.25.2/bazel-0.25.2-installer-linux-x86_64.sh      
        bash bazel-0.25.2-installer-linux-x86_64.sh --user

2. Add and source the ``bin`` path to your ``~/.bashrc`` file in order to be 
   able to call bazel from the user's installation we set up:

        export PATH=$PATH:~/bin
        source ~/.bashrc   

3. Additionally, you need to install `cmake` version 3.4 or higher

4. You need to have `virtualenv` version **16.0.0** (or lower) installed on your system to be able build `ngraph-bridge` bridge. The virtualenv is configured and used by the build script but not required for running `ngraph-bridge`. 

2. Please ensure that you have gcc 4.8 version installed on your system. The nGraph bridge links with the TensorFlow libraries that are build with gcc 4.8 version of the toolchain. 

#### Build 

1. Once TensorFlow's dependencies are installed, clone `ngraph-bridge` repo:

        git clone https://github.com/tensorflow/ngraph-bridge.git
        cd ngraph-bridge
        git checkout v0.16.0-rc1

   
2. Run the following Python script to build TensorFlow, nGraph and the bridge. Please use Python 3.5:

        python3 build_ngtf.py --use_prebuilt_tensorflow

Once the build finishes, a new virtualenv directory is created in the `build_cmake/venv-tf-py3`. The build artifacts i.e., the `ngraph_tensorflow_bridge-<VERSION>-py2.py3-none-manylinux1_x86_64.whl` is created in the `build_cmake/artifacts` directory. 

3. Test the installation by running the following command:
      
        python3 test_ngtf.py

This command will run all the C++ and python unit tests from the ngraph-bridge source tree. Additionally this will also run various TensorFlow python tests using nGraph.

4. To use the ngraph-tensorflow bridge, activate this virtual environment to start using nGraph with TensorFlow. 

        source build_cmake/venv-tf-py3/bin/activate
 
Alternatively, you can also install the TensorFlow and nGraph bridge outside of virtualenv. The Python `whl` files are located in the `build_cmake/artifacts/` and `build_cmake/artifats/tensorflow` directories, respectively. 

Select the help option of `build_ngtf.py` script to learn more about various build options and how to build other backends. 

## How to use nGraph with TensorFlow

1. Test the installation by running the following command:

        python -c "import tensorflow as tf; print('TensorFlow version: ',tf.__version__);import ngraph_bridge; print(ngraph_bridge.__version__)"

   This will produce something like this:

        TensorFlow version:  1.14.0
        C Compiler version used in building TensorFlow:  7.3.0
        nGraph bridge version: b'0.14.0'
        nGraph version used for this build: <version-number>
        TensorFlow version used for this build: <version-number>
        CXX11_ABI flag used for this build: <1 or 0>
        nGraph bridge built with Grappler: False
        nGraph bridge built with Variables and Optimizers Enablement: False

    Note: The version of the ngraph-tensorflow-bridge is not going to be exactly the same as when you build from source. This is due to delay in the source release and publishing the corresponding Python wheel. 

2. You can try out the TensorFlow models by adding the following lines to your existing TensorFlow model scripts and running them the usual way:

        import ngraph_bridge
        ...
        config = tf.ConfigProto() # or your existing config
        config_ngraph_enabled = ngraph_bridge.update_config(config)
        sess = tf.Session(config=config_ngraph_enabled) # use the updated config in session creation

Detailed examples on how to use ngraph_bridge are located in the [examples] directory.

## Using OS X 

The build and installation instructions are identical for Ubuntu 16.04 and OS X. However, please
note that the Python setup is not always the same across various Mac OS versions. TensorFlow build
instructions recommend using Homebrew and often people use Pyenv. There is also Anaconda/Miniconda 
which some users prefer. Ensure that you can build TenorFlow successfully on OS X with a suitable 
Python environment prior to building nGraph.  

## Debugging

The pre-requisite for building nGraph using `Option 3` is to be able to build TensorFlow from source. Often there are missing configuration steps for building TensorFlow. If you run into build issues, first ensure that you can build TensorFlow. For debugging run time issues, see the instructions provided in the [diagnostics] directory.

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

See the full documentation here:  <http://ngraph.nervanasys.com/docs/latest>


[linux-based install instructions on the TensorFlow website]:https://www.tensorflow.org/install/install_linux
[tensorflow]:https://github.com/tensorflow/tensorflow.git
[open-source C++ library, compiler and runtime]: http://ngraph.nervanasys.com/docs/latest/
[DSO]:http://csweb.cs.wfu.edu/~torgerse/Kokua/More_SGI/007-2360-010/sgi_html/ch03.html
[Github issues]: https://github.com/tensorflow/ngraph-bridge/issues
[pull request]: https://github.com/tensorflow/ngraph-bridge/pulls
[bazel version]: https://github.com/bazelbuild/bazel/releases/tag/0.25.2
[prepare environment]: https://www.tensorflow.org/install/install_sources#prepare_environment_for_linux
[diagnostics]:diagnostics/README.md
[examples]:examples/README.md
[ops]:http://ngraph.nervanasys.com/docs/latest/ops/index.html
[nGraph]:https://github.com/NervanaSystems/ngraph 
[ngraph-bridge]:https://github.com/tensorflow/ngraph-bridge.git 
 
