# Bridge TensorFlow*/XLA to run on Intel® nGraph™ backends

This directory contains the bridge code needed to build a TensorFlow*/XLA 
plugin that can be used with Intel® nGraph™. nGraph is an [open-source C++ 
library, compiler and runtime] that provides developers with the means to 
train and run DNN models efficiently on custom backends: GPU, CPU, and custom 
silicon.

To enable the bridge from TensorFlow to Intel nGraph backends, two repos are 
needed: `ngraph-tensorflow` and this repository, `ngraph-tensorflow-bridge`. After 
everything is built, we end up with a [DSO] that acts like a plugin to Intel 
nGraph backends.

**Note**
Currently we are using ngraph-tensorflow but eventually this will no longer 
be needed. You will be able to use official tensorflow distribution.

## Prepare the build environment

The installation prerequisites are the same as TensorFlow as described in the 
TensorFlow [prepare environment] for linux.

1. We use the standard build process which is a system called "bazel". These 
   instructions were tested with [bazel version 0.11.0]. 

   ```
   $ wget https://github.com/bazelbuild/bazel/releases/download/0.11.0/bazel-0.11.0-installer-linux-x86_64.sh      
   $ chmod +x bazel-0.11.0-installer-linux-x86_64.sh
   $ ./bazel-0.11.0-installer-linux-x86_64.sh --user
   ```

2. Add and source the ``bin`` path to your ``~/.bashrc`` file in order to be 
   able to call bazel from the user's installation we set up:

   ```  
   export PATH=$PATH:~/bin
   $ source ~/.bashrc   
   ```

3. Ensure that all the TensorFlow dependencies are installed, as per the
   TensorFlow [prepare environment] for linux.

   **Note** You do not need CUDA in order to use the ngraph-tensorflow bridge.


## How to enable the bridge

1. Once TensorFlow's dependencies are installed, clone the source of the 
   [tensorflow] repo to your machine; this is the required for running the unit
   test. If you want to run nGraph without the unit tests then you can 
   install the binary TensorFlow wheel and skip this section.

   ```
   git clone https://github.com/tensorflow/tensorflow.git
   cd tensorflow
   ```
2. When setting up and activating the virtual environment with TensorFlow 
   frameworks, you must use a specific kind of venv designed for 
   ``system-site-packages``.  A regular venv tends to not detect nGraph installs:

   ```
   virtualenv --system-site-packages <your_virtual_env_dir> # for Python 2.7
   ```
   For Python 3.n version:
   ```
   virtualenv --system-site-packages -p python3 <your_virtual_env_dir> # for Python 3.n
   ```
   Activate virtual environment:
   ```
   source <your_virtual_env_dir>/bin/activate # bash, sh, ksh, or zsh
   ```

3. Now run `./configure` and choose all the defaults when prompted to build TensorFlow.

4. Prepare the pip package and the TensorFlow C++ library:

    ```
    bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package
    bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
    ```

5. Install the pip package, replacing the `tensorflow-1.*` with your 
   version of TensorFlow:

    ```
    pip install -U /tmp/tensorflow_pkg/tensorflow-1.*whl
    ```  

6. If you want to run the unit tests, build the TensorFlow C++ library: 

    ```
    bazel build --config=opt //tensorflow:libtensorflow_cc.so
    ```

7. Now clone the `ngraph-tf` repo one level above -- in the 
  *parent* directory of the `tensorflow` repo cloned in step 1:

    ```
    cd ..
    git clone https://github.com/NervanaSystems/ngraph-tf.git
    cd ngraph-tf
    ```

8. Next, build and install `ngraph-tensorflow-bridge`:

    ```
    mkdir build
    cd build
    cmake ..
    make
    ```

This final step automatically downloads the necessary version of `ngraph` and 
the dependencies. The resulting plugin [DSO] is named `libngraph_device.so`.

Once the build and installation steps are complete, you can start experimenting 
with nGraph backends. 

## How to run unit tests
In order to run the unit tests, you need a copy of the [tensorflow] source tree.
We assume that you have done that using the instructions above. 

Now you need to build the tensorflow C++ library using the following instructions:
1. Go to the ngraph-tf/build/test directory
    ```
    cd build/test
    ./gtest_ngtf
    ```

### Run MNIST Softmax with the activated bridge 
TODO

## Support

Please submit your questions, feature requests and bug reports via [GitHub issues].

### Troubleshooting


## How to Contribute

We welcome community contributions to nGraph. If you have an idea of how
to improve the library:

* Share your proposal via [GitHub issues].
* Ensure you can build the product and run all the examples with your patch.
* In the case of a larger feature, create a test.
* Submit a [pull request].
* We will review your contribution and, if any additional fixes or
  modifications are necessary, may provide feedback to guide you. When
  accepted, your pull request will be merged to the repository.

## About Intel® nGraph™

See the full documentation here:  http://ngraph.nervanasys.com/docs/latest

## Future plans

[tensorflow]:https://github.com/tensorflow/tensorflow.git
[building a modified version of TensorFlow]:http://ngraph.nervanasys.com/docs/latest/framework-integration-guides.html#tensorflow 
[official distribution of TensorFlow]:https://github.com/tensorflow/tensorflow.git
[upstreaming discussions here]: https://groups.google.com/d/topic/xla-dev/LZdKcq7goko/discussion
[open-source C++ library, compiler and runtime]: http://ngraph.nervanasys.com/docs/latest/
[DSO]:http://csweb.cs.wfu.edu/~torgerse/Kokua/More_SGI/007-2360-010/sgi_html/ch03.html
[Github issues]: https://github.com/NervanaSystems/ngraph/issues
[pull request]: https://github.com/NervanaSystems/ngraph/pulls
[how to import]: http://ngraph.nervanasys.com/docs/latest/howto/import.html
[ngraph-ecosystem]: doc/sphinx/source/graphics/ngraph-ecosystem.png "nGraph Ecosystem"
[bazel version 0.11.0]: https://github.com/bazelbuild/bazel/releases/tag/0.11.0
[installation guide]: https://www.tensorflow.org/install/install_linux
[prepare environment]: https://www.tensorflow.org/install/install_sources#prepare_environment_for_linux
[installing with Virtualenv]: https://www.tensorflow.org/install/install_linux#installing_with_virtualenv
