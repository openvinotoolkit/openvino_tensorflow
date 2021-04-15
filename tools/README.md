# Readme for tools

## Build ManyLinux2014 compatible OpenVINO integration with TensorFlow whls

To build whl files compatible with manylinux2014, use the following commands. The build artifacts will be available in your container's /whl/ folder.

        cd builds
        docker build --no-cache -t openvino_tensorflow/pip --build-arg OVTF_BRANCH=releases/v0.5.0-beta-rc0 . -f Dockerfile.manylinux2014

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
