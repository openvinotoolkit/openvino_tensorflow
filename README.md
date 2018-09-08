# Bridge TensorFlow* to run on Intel® nGraph™ backends

This directory contains the code needed to build a TensorFlow 
plugin that can be used with Intel® nGraph™. nGraph is an [open-source C++ library, compiler and runtime] that provides developers with the means to 
train and run DNN models efficiently on custom backends: GPU, CPU, and custom 
silicon.

## Linux instructions

There are a few ways you can run nGraph with TensorFlow. They are described in the 
following section.

## Build nGraph bridge from source using existing TensorFlow installation

1. Install TensorFlow using the instructions from the TensorFlow web site
   https://www.tensorflow.org/install/install_linux

2. Now clone the `ngraph-tf` repo and go to the `ngraph-tf` directory

    ```
    git clone https://github.com/NervanaSystems/ngraph-tf.git
    cd ngraph-tf
    ```

3. Next, build and install nGraph bridge. 
   :warning: You must be inside the `virtualenv` wher TensorFlow installed 
   during the ngraph-tf build.

    ```
    mkdir build
    cd build
    cmake ..
    make -j <your_processor_cores>
    make install 
    pip install -U python/dist/<ngraph-0.5.0-py2.py3-none-linux_x86_64.whl>
    ```


## Build nGraph bridge from source using TensorFlow source

If you want to run unit tests or planning to contribute, then you need to install 
the nGraph bridge using the TensorFlow source tree. 

### Prepare the build environment

The installation prerequisites are the same as TensorFlow as described in the 
TensorFlow [prepare environment] for linux.

1. We use the standard build process which is a system called "bazel". These 
   instructions were tested with [bazel version 0.16.0]. 

   ```
   $ wget https://github.com/bazelbuild/bazel/releases/download/0.16.0/bazel-0.16.0-installer-linux-x86_64.sh      
   $ chmod +x bazel-0.16.0-installer-linux-x86_64.sh
   $ ./bazel-0.16.0-installer-linux-x86_64.sh --user
   ```

2. Add and source the ``bin`` path to your ``~/.bashrc`` file in order to be 
   able to call bazel from the user's installation we set up:

   ```  
   export PATH=$PATH:~/bin
   $ source ~/.bashrc   
   ```

3. Ensure that all the TensorFlow dependencies are installed, as per the
   TensorFlow [prepare environment] for linux.

   **Note:** You do not need CUDA in order to use the ngraph-tensorflow bridge.

4. Additional dependencies.
   - Install ```apt-get install libicu-dev``` to avoid the following (potential) error:
     ```unicode/ucnv.h: No such file or directory```.


### Installation

1. Once TensorFlow's dependencies are installed, clone the source of the 
   [tensorflow] repo to your machine. 

   :warning: You need the following version of TensorFlow: `v1.10.0`

   ```
   $ git clone https://github.com/tensorflow/tensorflow.git
   $ cd tensorflow
   $ git checkout v1.10.0
   $ git status
   HEAD detached at v1.10.0
   ```
2. When setting up and activating the virtual environment with TensorFlow 
   frameworks, you must use a specific kind of venv designed for 
   ``system-site-packages``.  A regular venv tends to not detect nGraph installs:

   ```
   virtualenv --system-site-packages -p /usr/bin/python2 <your_virtual_env_dir> # for Python 2.7
   ```
   For Python 3.n version:
   ```
   virtualenv --system-site-packages -p python3 <your_virtual_env_dir> # for Python 3.n
   ```
   Activate virtual environment:
   ```
   source <your_virtual_env_dir>/bin/activate # bash, sh, ksh, or zsh
   ```

3. Now run `./configure` and choose `no` for all the questions when prompted to build TensorFlow.

    Note that if you are running TensorFlow on a Skylake falily processor then select
    `-march=broadwell` when prompted to specify the optimization flags:
    ```
    Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native]: -march=broadwell
    ```
    This is due to an issue in TensorFlow which is being actively worked on: 
    https://github.com/tensorflow/tensorflow/issues/17273

4. Prepare the pip package and the TensorFlow C++ library:

    ```
    bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package
    bazel-bin/tensorflow/tools/pip_package/build_pip_package ./
    ```

   **Note:** You may run into the following error:
    ```
    AttributeError: 'int' object attribute '__doc__' is read-only
    Target //tensorflow/tools/pip_package:build_pip_package failed to build
    Use --verbose_failures to see the command lines of failed build steps.
    ```
    in which case you need to install enum34:
    ```
    pip install enum34
    ```

    You may also need to install a Python package named ```mock``` to prevent an import 
    error during the build process.

5. Install the pip package, replacing the `tensorflow-1.*` with your 
   version of TensorFlow:

    ```
    pip install -U ./tensorflow-1.*whl
    ```  

6. Now clone the `ngraph-tf` repo one level above -- in the 
  *parent* directory of the `tensorflow` repo cloned in step 1:

    ```
    cd ..
    git clone https://github.com/NervanaSystems/ngraph-tf.git
    cd ngraph-tf
    ```

7. Next, build and install nGraph bridge. 
   :warning: You must be inside the `virtualenv` during the ngraph-tf build.

    ```
    mkdir build
    cd build
    cmake -DUNIT_TEST_ENABLE=TRUE -DTF_SRC_DIR=<location of the TensorFlow source directory> ..
    make -j <your_processor_cores>
    make install 
    pip install -U python/dist/<ngraph-0.5.0-py2.py3-none-linux_x86_64.whl>
    ```

This final step automatically downloads the necessary version of `ngraph` and 
the dependencies. The resulting plugin [DSO] is named `libngraph_bridge.so`.

Once the build and installation steps are complete, you can start using TensorFlow 
with nGraph backends. 

Note: The actual filename for the pip package may be different as it's version 
dependent. Please check the `build/python/dist` directory for the actual pip wheel.

### Running tests

To run the C++ unit tests, please do the following:

1. Go to the build directory and run the following command:
    ```
    cd test
    ./gtest_ngtf
    ```
Next is to run a few DL models to validate the end-to-end functionality.

2. Go to the ngraph-tf/examples directory and run the following models.
    ```
    cd examples/mnist
    python mnist_fprop_only.py \
        --data_dir <input_data_location> 
    ```

## OS X Instructions

The build and installation instructions are idential for Ubuntu 16.04 and OS X.

### Running tests

1. Add `<path-to-tensorflow-repo>/bazel-out/darwin-py3-opt/bin/tensorflow` and `<path-to-ngraph-tf-repo>/build/ngraph/ngraph_dist/lib` to your `DYLD_LIBRARY_PATH`
2. Follow the C++ and Python instructions from the Linux based testing described above.

## Debugging

See the instructions provided in the [diagnostics] directory.

## Support

Please submit your questions, feature requests and bug reports via [GitHub issues].

## How to Contribute

We welcome community contributions to nGraph. If you have an idea for how to 
improve it:

* Share your proposal via [GitHub issues].
* Make sure your patch is in line with Google style by setting up your git `pre-commit` hooks.  First, ensure `clang-format` is in your path, then:

   ```
   pip install pre-commit autopep8 pylint
   pre-commit install
   ```

* Ensure you can build the product and run all the examples with your patch.
* In the case of a larger feature, create a test.
* Submit a [pull request].
* We will review your contribution and, if any additional fixes or
  modifications are necessary, may provide feedback to guide you. When
  accepted, your pull request will be merged to the repository.


## About Intel® nGraph™

See the full documentation here:  <http://ngraph.nervanasys.com/docs/latest>


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
[0.11.1 also works]:https://github.com/bazelbuild/bazel/releases/tag/0.11.1
[diagnostics]:diagnostics/README.md

 
 
