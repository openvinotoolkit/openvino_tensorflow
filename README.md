
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/NervanaSystems/ngraph-tf/blob/master/LICENSE) 
[![Build Status](https://badge.buildkite.com/f20db2a4be0e82e493faa08de85953d45b313b3be12abf4acf.svg?branch=master)](https://buildkite.com/ngraph/ngtf-master-cpu)


# Intel(R) nGraph(TM) Compiler and runtime for TensorFlow*

This repository contains the code needed to enable Intel(R) nGraph(TM) Compiler and 
runtime engine for TensorFlow. Use it to speed up your TensorFlow training and 
inference workloads. The nGraph Library and runtime suite can also be used to 
customize and deploy Deep Learning inference models that will "just work" with 
a variety of nGraph-enabled backends: CPU, GPU, and custom silicon like the 
[Intel(R) Nervana(TM) NNP](https://itpeernetwork.intel.com/inteldcisummit-artificial-intelligence/).

*   [Build using Linux](#linux-instructions)
*   [Build using OS X](#using-os-x)
*   [Debugging](#debugging)
*   [Support](#support)
*   [How to Contribute](#how-to-contribute)


## Linux instructions


### Option 1: Use a pre-built nGraph-TensorFlow bridge 

1. You need to instantiate a specific kind of `virtualenv`  to 
   be able to proceed with the `ngraph-tf` bridge installation. For 
   systems with Python 3.5 or Python 2.7, these commands are

        virtualenv --system-site-packages -p python3 your_virtualenv 
        virtualenv --system-site-packages -p /usr/bin/python2 your_virtualenv  
        source your_virtualenv/bin/activate # bash, sh, ksh, or zsh
    
2. Install TensorFlow v1.13.1:

        pip install -U tensorflow

2. Install nGraph-TensorFlow bridge:

        pip install -U ngraph-tensorflow-bridge
   
4. Test the installation by running the following command:

        python -c "import tensorflow as tf; print('TensorFlow version: ',tf.__version__);import ngraph_bridge; print(ngraph_bridge.__version__)"

This will produce something like this:

        TensorFlow version:  1.13.1
        nGraph bridge version: b'0.12.0-rc6'
        nGraph version used for this build: b'0.18.0-rc.2+c5d52f1'
        TensorFlow version used for this build: v1.13.1-0-g6612da8951
        CXX11_ABI flag used for this build: 1

Next you can try out the TensorFlow models by adding one line to your existing 
TensorFlow model scripts and running them the usual way:

        import ngraph_bridge

Examples on how to use ngraph_bridge is here : https://github.com/NervanaSystems/ngraph-tf/tree/master/examples

Note: The version of the ngraph-tensorflow-bridge is not going to be exactly the same as when you build from source. This is due to delay in the source release and publishing the corresponding Python wheel. 

### Option 2: Build nGraph bridge from source

To use the latest version, or to run unit tests, or if you are planning to contribute, install the nGraph 
bridge using the TensorFlow source tree as follows: 

#### Prepare the build environment

The installation prerequisites are the same as described in the TensorFlow 
[prepare environment] for linux.

1. TensorFlow uses a build system called "bazel". These instructions were tested with [bazel version 0.21.0]. 

        wget https://github.com/bazelbuild/bazel/releases/download/0.21.0/bazel-0.21.0-installer-linux-x86_64.sh      
        chmod +x bazel-0.21.0-installer-linux-x86_64.sh
        ./bazel-0.21.0-installer-linux-x86_64.sh --user

2. Add and source the ``bin`` path to your ``~/.bashrc`` file in order to be 
   able to call bazel from the user's installation we set up:

        export PATH=$PATH:~/bin
        source ~/.bashrc   

3. Additionally, you need to install `cmake` version 3.1 or higher and gcc 4.8 or higher. 


#### Build 

1. Once TensorFlow's dependencies are installed, clone `ngraph-tf` repo:

        git clone https://github.com/NervanaSystems/ngraph-tf.git
        cd ngraph-tf
        git checkout v0.12.0-rc6

   
2. Next run the following Python script to build TensorFlow, nGraph and the bridge. Please use Python 3.5:

        python3 build_ngtf.py

:warning: Note that if you are running TensorFlow on a Skylake family processor then you need to pass the target_arch parameter, as below

        python3 build_ngtf.py --target_arch=broadwell

This is due to an issue in TensorFlow tracked here:
        https://github.com/tensorflow/tensorflow/issues/17273

Once the build finishes, a new virtualenv directory is created in the `build_cmake/venv-tf-py3`. The build artifacts i.e., the `ngraph_tensorflow_bridge-<VERSION>-py2.py3-none-manylinux1_x86_64.whl` is created in the `build_cmake/artifacts` directory. 

3. Test the installation by running the following command:
      
        python3 test_ngtf.py

This command will run all the C++ and python unit tests from the ngraph-tf source tree. Additionally this will also run various TensorFlow python tests using nGraph.

4. To use the ngraph-tensorflow bridge, activate this virtual environment to start using nGraph with TensorFlow. 

        source build_cmake/venv-tf-py3/bin/activate

Once the build and installation steps are complete, you can start using TensorFlow 
with nGraph backends. 

Please add the following line to enable nGraph: `import ngraph_bridge`

## Using OS X 

The build and installation instructions are identical for Ubuntu 16.04 and OS X. However, please
note that the Python setup is not always the same across various Mac OS versions. TensorFlow build
instructions recommend using Homebrew and often people use Pyenv. There is also Anaconda/Miniconda 
which some users prefer. Ensure that you can build TenorFlow successfully on OS X with a suitable 
Python environment prior to building nGraph.  

## Debugging

See the instructions provided in the [diagnostics] directory.

https://github.com/NervanaSystems/ngraph-tf/blob/master/diagnostics/README.md


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
[Github issues]: https://github.com/NervanaSystems/ngraph-tf/issues
[pull request]: https://github.com/NervanaSystems/ngraph-tf/pulls
[bazel version 0.21.0]: https://github.com/bazelbuild/bazel/releases/tag/0.21.0
[prepare environment]: https://www.tensorflow.org/install/install_sources#prepare_environment_for_linux
[diagnostics]:diagnostics/README.md
[ops]:http://ngraph.nervanasys.com/docs/latest/ops/index.html
[nGraph]:https://github.com/NervanaSystems/ngraph 
[ngraph-tf bridge]:https://github.com/NervanaSystems/ngraph-tf 
 
