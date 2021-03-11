# Readme for tools

## Build and run openvino_tensorflow in Docker

To run nGraph in Docker, choose one of two ways to create your container:
  1. Use the [`docker_build_and_install_ovtf.sh`](docker_build_and_install_ovtf.sh) script to do a multi-stage build and run nGraph for Ubuntu 18.04 in a single command. 
     This will perform all of the build steps automatically in an intermediate container and provide a final image without all the tools needed to build Tensorflow and nGraph. 
  2. Use `Dockerfile.ubuntu.18.04` by itself to set up a build environment that you can use to then manually build Tensorflow, nGraph, and the bridge in a virtualenv. 

##### Method 1:

- Clone the `openvino_tensorflow` repo:
  
        git clone https://github.com/openvinotoolkit/openvino_tensorflow.git
  
- Navigate into the `tools` directory and run the installation script:
  
        cd openvino_tensorflow/tools
        . docker_build_and_install_ovtf.sh

  If you want to use build options such as `--use_prebuilt_tensorflow` or `--use_grappler_optimizer`, set an input argument when running the installation script.

        . docker_build_and_install_ovtf.sh '--use_prebuilt_tensorflow --use_grappler_optimizer'

  For more information about build options, see [here](/build_ovtf.py).
  There may be some build options not supported with this method, so if your customized build is failing, **Method 2** is recommended. 
  
- When the multi-stage docker build is complete, you will be able to run a container with Tensorflow and nGraph using the `openvino_tensorflow:ovtf` image:

        docker run -it --name ovtf openvino_tensorflow:ovtf
        
  Note: If running behind a proxy, you will need to set `-e http_proxy=<http_proxy>` and `-e https_proxy=<https_proxy>` variables in order to run the test script.

- After running the container, you can perform an inference test by running: 

        python examples/infer_image.py
  
##### Method 2:

- Clone the `openvino_tensorflow` repo:

        git clone https://github.com/openvinotoolkit/openvino_tensorflow.git

- Navigate into the `tools` directory and build the dockerfile:

        cd openvino_tensorflow/tools
        docker build -t openvino_tensorflow:devel -f=Dockerfile.ubuntu18.04 .

- Navigate up one level and run the image with the openvino_tensorflow project mounted to `/workspace`:  

        cd ..
        docker run -it -v ${PWD}:/workspace -w /workspace --name ovtf openvino_tensorflow:devel

- Follow the instructions in [Build an Openvino_Tensoflow](/README.md#build-an-openvino_tensorflow) to execute `python3 build_ovtf.py`.
  You do not need to clone the repo inside the container because it is already mounted to `/workspace`.
  The mounted volume allows you to access the build artifacts (`whl` files) outside the container if you wish to do so.
  
- After the build completes, you will be able to use the virtualenv located at `/workspace/build_cmake/venv-tf-py3` and run a test.

        source build_cmake/venv-tf-py3/bin/activate
        python examples/infer_image.py



## Introducing grappler

#### Normal mode of operations
The normal mode of operation of `openvino_tensorflow` is:
```
import openvino_tensorflow

in0, in1, out0 = construct_network()
sess = tf.Session()
sess.run(out0, feed_dict = {in0:[2,3], in1:[4,5]})
```

The "normal" mode latches onto `Tensorflow` operations by registering `GraphOptimizationPass`es such as `NGraphRewritePass` and `NGraphVariableCapturePass`. It also registers a custom op `NGraphEncapsulate`. Note that the graph rewriting and the actual execution all happens when `session.run` is called.


#### Introducing `Grappler`
`Grappler` can be thought of as a function that accepts a `graphdef` and returns a (most likely) new modified `graphdef`, although in cases of failure to transform, `Tensorflow` will silently continue to run with the input untransformed `graphdef`. Recently we have added a way to build `openvino_tensorflow` with `Grappler`, by using the `--use_grappler_optimizer` flag in `build_ovtf.py`. We register the `OVTFOptimizer` as our `Grappler` pass. It pretty much does the same rewriting that `NGraphRewritePass` and `NGraphVariableCapturePass` was doing earlier, except for some subtle differences. For example, when `grappler` receives the graph in a certain stage of the `Tensorflow` execution pipeline which is different from when the `GraphOptimizationPass`es worked. Also we add `IdentityN` nodes to fetch (outputs), feed (inputs), init and keep ops to ensure we also capture these nodes(if supported), because by default `grappler` leaves them out.

A sample script will look like:
```
import openvino_tensorflow

in0, in1, out0 = construct_network()
sess = tf.Session(config=openvino_tensorflow.update_config(tf.ConfigProto()))
result = sess.run(out0, feed_dict = {in0:[2,3], in1:[4,5]})
```

Notice the new line introduced here wrt the "normal" path. `config_updated = openvino_tensorflow.update_config(config)` must now be passed during session construction to ensure `OVTFOptimizer` `grappler` pass is enabled. Without `grappler` build, we use `GraphOptimizationPass`, in which case just `import openvino_tensorflow` was enough to plug in `ngraph`. But in a `grappler` build, the config needs to be modified to enable the pass that will plugin `ngraph`. Like the "normal" path, in this script too the graph rewriting and actual execution happen when `session.run` is called.
