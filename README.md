# Bridge TensorFlow to run on Intel® nGraph™ backends

This directory contains the code needed to build a TensorFlow 
plugin that can be used with Intel® nGraph™. nGraph is an [open-source C++ 
library, compiler and runtime] that provides developers with the means to 
train and run DNN models efficiently on custom backends: GPU, CPU, and custom 
silicon.


## Linux instructions

### Prepare the build environment

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

   **Note:** You do not need CUDA in order to use the ngraph-tensorflow bridge.

### Installation

1. Once TensorFlow's dependencies are installed, clone the source of the 
   [tensorflow] repo to your machine; this is required for running the unit
   test. If you want to run nGraph without the unit tests then you can 
   install the binary TensorFlow wheel and skip this section.

   ```
   $ git clone https://github.com/tensorflow/tensorflow.git
   $ cd tensorflow
   $ git checkout v1.8.0
   $ git status
   HEAD detached at v1.8.0
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

3. Now run `./configure` and choose all the defaults when prompted to build TensorFlow.

4. Prepare the pip package and the TensorFlow C++ library:

    ```
    bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package
    bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
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

5. Install the pip package, replacing the `tensorflow-1.*` with your 
   version of TensorFlow:

    ```
    pip install -U /tmp/tensorflow_pkg/tensorflow-1.*whl
    ```  

6. Now clone the `ngraph-tf` repo one level above -- in the 
  *parent* directory of the `tensorflow` repo cloned in step 1:

    ```
    cd ..
    git clone https://github.com/NervanaSystems/ngraph-tf.git
    cd ngraph-tf
    ```

7. Next, build and install nGraph bridge:

    ```
    mkdir build
    cd build
    cmake ..
    make -j <your_processor_cores>
    make install 
    pip install python/dist/ngraph-0.0.0-py2-none-any.whl
    ```

This final step automatically downloads the necessary version of `ngraph` and 
the dependencies. The resulting plugin [DSO] is named `libngraph_device.so`.

Once the build and installation steps are complete, you can start experimenting 
with nGraph backends. 


### Running tests

In order to run the unit tests, you need `pytest` installed:
```
pip install pytest
```
Now run the tests using the following instructions:
1. Go to the ngraph-tf/build/test directory
    ```
    cd build/test
    ./gtest_ngtf
    cd python
    pytest
    ```

Next is to run a few DL models to validate the end-to-end functionality.

2. Go to the ngraph-tf/examples directory and run the following models.
    ```
    cd examples
    python mnist_fprop_only.py \
        --data_dir <input_data_location> --select_device NGRAPH
    python tf_cnn_benchmarks.py --model=resnet50 --eval \
        --num_inter_threads=1 --batch_size=1 \
        --train_dir <rep-trained-model-location>/resnet50 \
        --data_format NCHW --select_device NGRAPH \
        --num_batches=10
    ```

<!--
### Run MNIST Softmax with the activated bridge 
TODO
-->


## OS X Instructions (Experimental)

### Prepare the build environment

1. Install [bazel](https://github.com/bazelbuild/bazel/releases).  [0.11.1 works](https://github.com/bazelbuild/bazel/releases/tag/0.11.1), or, if you're feeling adventurous, you could try a later version.
2. `port install coreutils`, then add `/opt/local/libexec/gnubin` **in front** of your `$PATH`.  Both `tensorflow` and `ngraph` assume GNU userland tools, and you'll run into errors otherwise.
3. Make and activate yourself a virtualenv that we'll be using for our custom-built TensorFlow.


### Installation

1. Build TensorFlow and its framework for unit tests. This step is identical to 
how you would build TensorFlow for Linux mentioned above.

	```
	git clone git@github.com:tensorflow/tensorflow.git
	pushd tensorflow
	git checkout r1.8
	./configure # you can disable everything here if you like, or stick with defaults
	bazel run //tensorflow/tools/pip_package:build_pip_package /tmp/tensorflow_pkg
	pip install /tmp/tensorflow_pkg/tensorflow*.whl
	bazel build --config=opt //tensorflow:libtensorflow_cc.so
	popd
	```

2. Prepare `ngraph-tf` for the build:

	```
	git clone git@github.com:NervanaSystems/ngraph-tf.git
	pushd ngraph-tf.git
	ln -s ../tensorflow
	mkdir build && cd build
	cmake -DNGRAPH_USE_PREBUILT_LLVM=False ..
	```

Note: If you want to build a version with no optimization for debugging
then you can use the `-DCMAKE_BUILD_TYPE=Debug` flag during the cmake step
mentioned above.

3. `make -j <your-core-count>`

### Running tests

#### C++

1. `cd test && make -j <your-core-count>`
2. Add `<path-to-tensorflow-repo>/bazel-out/darwin-py3-opt/bin/tensorflow` and `<path-to-ngraph-tf-repo>/build/ngraph/ngraph_dist/lib` to your `LD_LIBRARY_PATH` and `DYLD_LIBRARY_PATH`
3. `./gtest_ngtf`


#### Python

1. `pip install pytest` if you don't already have it, then
2. `cd build/test/python`
3. `pytest`


### Debugging

Don't just use `lldb` -- it likely refers to `/usr/bin/lldb` and OS X security preferences will prevent it from inheriting your `LD_LIBRARY_PATH`.  Instead, alias it to `/Applications/Xcode.app/Contents/Developer/usr/bin/lldb`.


## Support

Please submit your questions, feature requests and bug reports via [GitHub issues].


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
