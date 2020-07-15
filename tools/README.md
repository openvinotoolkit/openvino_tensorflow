# Readme for tools

## Build and run nGraph in Docker

To run nGraph in Docker, choose one of two ways to create your container:
  1. Use the [`docker_build_and_install_ngtf.sh`](docker_build_and_install_ngtf.sh) script to do a multi-stage build and run nGraph for Ubuntu 18.04 in a single command. 
     This will perform all of the build steps automatically in an intermediate container and provide a final image without all the tools needed to build Tensorflow and nGraph. 
  2. Use `Dockerfile.ubuntu.18.04` by itself to set up a build environment that you can use to then manually build Tensorflow, nGraph, and the bridge in a virtualenv. 

##### Method 1:

- Clone the `ngraph-bridge` repo:
  
        git clone https://github.com/tensorflow/ngraph-bridge.git
  
- Navigate into the `tools` directory and run the installation script:
  
        cd ngraph-bridge/tools
        . docker_build_and_install_ngtf.sh

  If you want to use build options such as `--use_prebuilt_tensorflow` or `--use_grappler_optimizer`, set an input argument when running the installation script.

        . docker_build_and_install_ngtf.sh '--use_prebuilt_tensorflow --use_grappler_optimizer'

  For more information about build options, see [here](/build_ngtf.py).
  There may be some build options not supported with this method, so if your customized build is failing, **Method 2** is recommended. 
  
- When the multi-stage docker build is complete, you will be able to run a container with Tensorflow and nGraph using the `ngraph-bridge:ngtf` image:

        docker run -it --name ngtf ngraph-bridge:ngtf
        
  Note: If running behind a proxy, you will need to set `-e http_proxy=<http_proxy>` and `-e https_proxy=<https_proxy>` variables in order to run the test script.

- After running the container, you can perform an inference test by running: 

        python examples/infer_image.py
  
##### Method 2:

- Clone the `ngraph-bridge` repo:

        git clone https://github.com/tensorflow/ngraph-bridge.git

- Navigate into the `tools` directory and build the dockerfile:

        cd ngraph-bridge/tools
        docker build -t ngraph-bridge:devel -f=Dockerfile.ubuntu18.04 .

- Navigate up one level and run the image with the ngraph-bridge project mounted to `/workspace`:  

        cd ..
        docker run -it -v ${PWD}:/workspace -w /workspace --name ngtf ngraph-bridge:devel

- Follow the instructions in [Build an nGraph bridge](/README.md#build-an-ngraph-bridge) to execute `python3 build_ngtf.py`.
  You do not need to clone the repo inside the container because it is already mounted to `/workspace`.
  The mounted volume allows you to access the build artifacts (`whl` files) outside the container if you wish to do so.
  
- After the build completes, you will be able to use the virtualenv located at `/workspace/build_cmake/venv-tf-py3` and run a test.

        source build_cmake/venv-tf-py3/bin/activate
        python examples/infer_image.py



## Introducing grappler and tf2ngraph

#### Normal mode of operations
The normal mode of operation of `ngrph_bridge` is:
```
import ngraph_bridge

in0, in1, out0 = construct_network()
sess = tf.Session()
sess.run(out0, feed_dict = {in0:[2,3], in1:[4,5]})
```

The "normal" mode latches onto `Tensorflow` operations by registering `GraphOptimizationPass`es such as `NGraphRewritePass` and `NGraphVariableCapturePass`. It also registers a custom op `NGraphEncapsulate`. Note that the graph rewriting and the actual execution all happens when `session.run` is called.


#### Introducing `Grappler`
`Grappler` can be thought of as a function that accepts a `graphdef` and returns a (most likely) new modified `graphdef`, although in cases of failure to transform, `Tensorflow` will silently continue to run with the input untransformed `graphdef`. Recently we have added a way to build `ngraph-bridge` with `Grappler`, by using the `--use_grappler_optimizer` flag in `build_ngtf.py`. We register the `NgraphOptimizer` as our `Grappler` pass. It pretty much does the same rewriting that `NGraphRewritePass` and `NGraphVariableCapturePass` was doing earlier, except for some subtle differences. For example, when `grappler` receives the graph in a certain stage of the `Tensorflow` execution pipeline which is different from when the `GraphOptimizationPass`es worked. Also we add `IdentityN` nodes to fetch (outputs), feed (inputs), init and keep ops to ensure we also capture these nodes(if supported), because by default `grappler` leaves them out.

A sample script will look like:
```
import ngraph_bridge

in0, in1, out0 = construct_network()
sess = tf.Session(config=ngraph_bridge.update_config(tf.ConfigProto()))
result = sess.run(out0, feed_dict = {in0:[2,3], in1:[4,5]})
```

Notice the new line introduced here wrt the "normal" path. `config_updated = ngraph_bridge.update_config(config)` must now be passed during session construction to ensure `NgraphOptimizer` `grappler` pass is enabled. Without `grappler` build, we use `GraphOptimizationPass`, in which case just `import ngraph_bridge` was enough to plug in `ngraph`. But in a `grappler` build, the config needs to be modified to enable the pass that will plugin `ngraph`. Like the "normal" path, in this script too the graph rewriting and actual execution happen when `session.run` is called.


#### tf2ngraph

The advantage of `grappler` is that the graph rewriting can happen before `session.run` is called. This is exactly what `tf2ngraph` does. The workflow can now look like this:
```
python tf2ngraph.py --inputpbtxt input.pbtxt --outnodes out_node --outputpbtxt output.pbtxt --ngbackend CPU
```
followed by
```
import ngraph_bridge
from google.protobuf import text_format

with open('output.pbtxt', "r") as f:
    text_format.Merge(f.read(), graph_def)
sess = tf.Session(graph=graph)
in0 = get_tensor_by_name('in0')
in1 = get_tensor_by_name('in1')
out0 = get_tensor_by_name('out0')
result = sess.run(out1, feed_dict={in0:[2,3], in1:[4,5]})
```

Note that in the second phase, we did not need `ngraph_bridge.update_config`. This is because all graph rewrites have finished in the first phase (`tf2ngraph`) and now in the second phase the `import ngraph_bridge` is needed only to identify the custom op `NGraphEncapsulate`, but not for rewrites. If we set the environment flag `NGRAPH_TF_LOG_PLACEMENT=1` in the first phase, we would see the TF nodes with some being placed on `ngraph`. If we use `NGRAPH_TF_LOG_PLACEMENT=1` in the second phase, we would not see most `tensorflow` nodes since they have disappeared inside `NGraphEncapsulate`s. We will only see a few `tensorflow` nodes (those unsupported by `ngraph-bridge` hence not captured inside `NGraphEncapsulate`) and some `NGraphEncapsulate` nodes. All nodes will be shown placed on `HOST` and not on `ngraph` (because `NGraphEncapsulate` runs on `HOST`)

Some features of `tf2ngraph:
1. You can get a comprehensive help string using: `python tf2ngraph.py -h`
2. It supports 3 input formats: `pb`, `pbtxt` and `saved model`
3. It supports 3 output formats: `pb`, `pbtxt` and `saved model`
4. It requires the user to specify the output node names using `--outnodes`. For example: `--outnodes add:0,sub:0`
5. Optionally one can also specify the backend along with config strings after colons `--ngbackend CPU` or `--ngbackend GPU:0` etc. By default we use CPU
6. Its pytest/unit test is present [here]
7. `tf2ngraph` can be used in 2 ways: CLI and python functional api. The CLI is well tested and recommended for now

## build_utils, test_utils
These files contain some useful functions we use for building and testing. For example:
1. **Command_executor**: The function accepts a string and executes it as a shell command
2. **Venv related functions**: `load_venv`, `install_virtual_env`, `setup_venv`
3. **Building functions**: `build_ngraph`, `build_tensorflow`, `install_tensorflow`, `build_ngraph_tf`, `install_ngraph_tf`, `install_ngraph_bridge`
4. **Git related functions**: `download_repo`, `apply_patch`
5. **Test related functions** Some test related functions in `test_utils` used for CI

[here]: https://github.com/tensorflow/ngraph-bridge/blob/master/test/python/test_convert_script.py
