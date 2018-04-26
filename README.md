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

IMPORTANT: The nGraph TensorFlow bridge currently uses an experimental plugin
API that is not part of mainstream TensorFlow. You must build TensorFlow from
a specific tag from the `NervanaSystems/ngraph-tensorflow` repo. Discussions
about upstreaming the plugin API are ongoing. See the section "Future plans"
below for more details.

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
   [ngraph-tensorflow] repo to your machine; this is the required fork for 
   this integration. Then check out the correct tag for this version of the 
   bridge:

   ```
   git clone https://github.com/NervanaSystems/ngraph-tensorflow.git
   cd ngraph-tensorflow
   git checkout ngraph-tensorflow-preview-0
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

3. Now run `./configure` and choose `y` when prompted to build TensorFlow with 
   "XLA JIT support"; when you come across this question, select `y`.  

    ```
    Do you wish to build TensorFlow with XLA JIT support? [y/N]: y
    ``` 

4. Prepare the pip package:

    ```
    bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package
    bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
    ```

5. Install the pip package, replacing the `tensorflow-1.*` with your 
   version of TensorFlow:

    ```
    pip install -U /tmp/tensorflow_pkg/tensorflow-1.*whl
    ```
   
    **Note** The actual name of the Python wheel file will be updated to the 
    official version of TensorFlow as the `ngraph-tensorflow` repository is 
    synchronized frequently with the original TensorFlow repository.

6. Now clone the `ngraph-tensorflow-bridge` repo one level above -- in the 
  *parent* directory of the `ngraph-tensorflow` repo cloned in step 1:

    ```
    cd ..
    git clone https://github.com/NervanaSystems/ngraph-tensorflow-bridge.git
    cd ngraph-tensorflow-bridge
    ```

7. Next, build and install `ngraph-tensorflow-bridge`:

    ```
    mkdir build
    cd build
    cmake ../
    make install
    ```
8. Finally test the installation by running the following python command
   ```
   cd ../test
   python install_test.py
   ```
   You should see the following output:
   ```
   nGraph device available
   Name:  /device:NGRAPH:0
   ```

This final step automatically downloads the necessary version of `ngraph` and 
the dependencies. The resulting plugin [DSO] named `libngraph_plugin.so` gets 
copied to the following directory inside the TensorFlow installation directory: 
`<Python site-packages>/tensorflow/plugins`

Once the build and installation steps are complete, you can start experimenting 
with nGraph backends. 


### Run MNIST Softmax with the activated bridge 

To see everything working together, you can run MNIST Softmax example with the 
now-activated bridge to nGraph. The script named `mnist_softmax_ngraph.py` 
can be found under the `ngraph-tensorflow-bridge/test` directory. 
It was modified from the example explained in the TensorFlow* tutorial; the 
following changes were made from the original script: 

```python
def main(_):
with tf.device('/device:NGRAPH:0'):
  run_mnist(_)

def run_mnist(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
  ...
```

To run this example, first set the configuration options via command-line:

```bash
export OMP_NUM_THREADS=4
export KMP_AFFINITY=granularity=fine,scatter
```

Then run the script as follows from within the `/test` directory of 
your `ngraph-tensorflow-bridge` clone:

```bash
python mnist_softmax_ngraph.py
```

**Note** The number-of-threads parameter specified in the `OMP_NUM_THREADS` is 
a function of number of CPU cores that are available in your system. 

## Support

Please submit your questions, feature requests and bug reports via [GitHub issues].

### Troubleshooting

If the installation test fails, then run the script again with logging enabled:
```
TF_CPP_MIN_VLOG_LEVEL=1 python install_test.py 
```
This will create debug messages that you can analyze to detect the error or 
when submitting the github issue.


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

Discussions with Google's TensorFlow team are in progress regarding a 
dynamically-loadable XLA plugin scheme, one that can can be loaded 
directly from the [official distribution of TensorFlow] when specified
during build time of a TensorFlow framework.    

Follow those [upstreaming discussions here].


[ngraph-tensorflow]:https://github.com/NervanaSystems/ngraph-tensorflow.git
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