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
    
2. Install TensorFlow v1.12.0:

        pip install -U tensorflow

2. Install nGraph-TensorFlow bridge:

        pip install -U ngraph-tensorflow-bridge
   
4. Test the installation by running the following command:

        python -c "import tensorflow as tf; print('TensorFlow version: r',tf.__version__);import ngraph_bridge; print(ngraph_bridge.__version__)"

This will produce something like this:

        TensorFlow version: r 1.12.0
        TensorFlow version installed: 1.12.0 (v1.12.0-0-ga6d8ffae09)
        nGraph bridge built with: 1.12.0 (v1.12.0-0-ga6d8ffae09)
        b'0.10.0'

Next you can try out the TensorFlow models by adding one line to your existing 
TensorFlow model scripts and running them the usual way:

        import ngraph_bridge

Note: The version of the ngraph-tensorflow-bridge is not going to be exactly the same as when you build from source. This is due to delay in the source release and publishing the corresponding Python wheel. 

### Option 2: Build nGraph bridge from source using TensorFlow source

To use the latest version, or to run unit tests, or if you are planning to contribute, install the nGraph 
bridge using the TensorFlow source tree as follows: 

#### Prepare the build environment

The installation prerequisites are the same as described in the TensorFlow 
[prepare environment] for linux.

1. TensorFlow uses a build system called "bazel". These instructions were tested with [bazel version 0.16.0]. 

        wget https://github.com/bazelbuild/bazel/releases/download/0.16.0/bazel-0.16.0-installer-linux-x86_64.sh      
        chmod +x bazel-0.16.0-installer-linux-x86_64.sh
        ./bazel-0.16.0-installer-linux-x86_64.sh --user

2. Add and source the ``bin`` path to your ``~/.bashrc`` file in order to be 
   able to call bazel from the user's installation we set up:

        export PATH=$PATH:~/bin
        source ~/.bashrc   

3. Additionally, you need to install `cmake` version 3.1 or higher and gcc 4.8 or higher. 


#### Build 

1. Once TensorFlow's dependencies are installed, clone `ngraph-tf` repo:

        git clone https://github.com/NervanaSystems/ngraph-tf.git
        cd ngraph-tf
        git checkout v0.10.0

   
2. Next run the following Python script to build TensorFlow, nGraph and the bridge. Please use Python 3.5:

        python3 build_ngtf.py

:warning: Note that if you are running TensorFlow on a Skylake family processor then you need to pass the target_arch parameter, as below

        python3 build_ngtf.py --target_arch=broadwell

This is due to an issue in TensorFlow tracked here:
        https://github.com/tensorflow/tensorflow/issues/17273

Once the build finishes, a new virtualenv directory is created in the `build/venv-tf-py3`. The build artifacts i.e., the `ngraph_tensorflow_bridge-<VERSION>-py2.py3-none-manylinux1_x86_64.whl` is created in the `build/artifacts` directory. 

3. Test the installation by running the following command:
      
        python3 test_ngtf.py

This command will run all the C++ and python unit tests from the ngraph-tf source tree. Additionally this will also run various TensorFlow python tests using nGraph.

4. To use the ngraph-tensorflow bridge, activate this virtual environment to start using nGraph with TensorFlow. 

        source build/venv-tf-py3/bin/activate

Once the build and installation steps are complete, you can start using TensorFlow 
with nGraph backends. 

Please add the following line to enable nGraph: `import ngraph_bridge`

## Option 3: Using the upstreamed version

nGraph is updated in the TensorFlow source tree using pull requests periodically. 

In order to build that version of nGraph, follow the steps below, which involves building TensorFlow from source with certain settings.

1. Install bazel using the same instructions outlined in option 2, step 1 above.

2. Create a virtual environment using the instructions outlined in option 1, step 1 above.

3. Get tensorflow v1.12.0

        git clone https://github.com/tensorflow/tensorflow.git
        cd tensorflow
        git checkout v1.12.0

Note: To get the latest version of nGraph, use the tip of `master` branch of TensorFlow. The exact version of `bazel` changes for a specific version of TensorFlow. Please consult the build instructions from TensorFlow web site for specific bazel requirements.

4. Now run `./configure` and choose `no` for the following when prompted to build TensorFlow.

    XLA support:

        Do you wish to build TensorFlow with XLA JIT support? [Y/n]: n
        No XLA JIT support will be enabled for TensorFlow.

    CUDA support:
    
        Do you wish to build TensorFlow with CUDA support? [y/N]: N
        No CUDA support will be enabled for TensorFlow.
    
    :warning: Note that if you are running TensorFlow on a Skylake family processor then select
    `-march=broadwell` when prompted to specify the optimization flags:
    
        Please specify optimization flags to use during compilation 
        when bazel option "--config=opt" is specified 
        [Default is -march=native]: -march=broadwell
    
    This is due to an issue in TensorFlow tracked here: 
    https://github.com/tensorflow/tensorflow/issues/17273

5. Prepare the pip package

        bazel build --config=opt --config=ngraph //tensorflow/tools/pip_package:build_pip_package 
        bazel-bin/tensorflow/tools/pip_package/build_pip_package ./

Note: The specific questions for the `configure` step and the build command mentioned above changes for different versions of TensorFlow. 

6. Once the pip package is built, install using

        pip install -U ./tensorflow-1.*whl

For this final option, there is **no need to separately build `ngraph-tf` or to 
use `pip` to install the nGraph module**. With this configuration, your TensorFlow model scripts will work without any changes, ie, you do not need to add `import ngraph_bridge`, like option 1 and 2. 

Note: The version that is available in the upstreamed version of TensorFlow usually
lags the features and bug fixes available in the `master` branch of this repository.

You can run a few of your own DL models to validate the end-to-end 
functionality. Also, you can use the `ngraph-tf/examples` directory and try to 
run the following model: 

        cd examples
        python3 keras_sample.py 

## Using OS X 

The build and installation instructions are idential for Ubuntu 16.04 and OS X. However, please
note that the Python setup is not always the same across various Mac OS versions. TensorFlow build
instructions recommend using homebrew and often people use pyenv. There is also Anaconda/Miniconda 
which some users prefer. The basic criteria for building nGraph and the bridge on a Mac OS is 
to ensure that TensorFlow is successfully built. 

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
[bazel version 0.16.0]: https://github.com/bazelbuild/bazel/releases/tag/0.16.0
[prepare environment]: https://www.tensorflow.org/install/install_sources#prepare_environment_for_linux
[diagnostics]:diagnostics/README.md
[ops]:http://ngraph.nervanasys.com/docs/latest/ops/index.html
[nGraph]:https://github.com/NervanaSystems/ngraph 
[ngraph-tf bridge]:https://github.com/NervanaSystems/ngraph-tf 
 
