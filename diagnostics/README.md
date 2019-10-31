# Diagnostics tools

What follows here is a collection of tools to help triage 
and diagnose or troubleshoot.

Notation used:

* TensorFlow: `TF`
* TensorBoard: `TB`
* protobuf binary: `pb`
* protobuf string text: `pbtxt`
* NGraph-TF: **NGTF**


## What to do if your network fails?

**TODO**: List steps to generate logs or run diagnostic tools


### Capturing logs in text file

**NGTF** uses the std error to output its logs, so it is necessary to pipe 
it correctly to capture all logs.

    python run_TF_network.py > log.txt 2>&1

### A full dump

To get a **full** dump use the following set of flags
```NGRAPH_ENABLE_SERIALIZE=1 NGRAPH_CPU_TRACING=1 NGRAPH_TF_VLOG_LEVEL=5 NGRAPH_TF_LOG_PLACEMENT=1 NGRAPH_TF_DUMP_CLUSTERS=1 NGRAPH_TF_DUMP_GRAPHS=1 python run_TF_network.py > log.txt 2>&1```


## Debug flags

|Name                          |Description                            |
|------------------------------|---------------------------------------|
| `NGRAPH_ENABLE_SERIALIZE=1`  | Generate nGraph-level serialized graphs|
| `NGRAPH_CPU_TRACING=1`       | Generate nGraph-level function timelines|
| `NGRAPH_TF_VLOG_LEVEL=5`     | Generate ngraph-tf logging info for different passes|
| `NGRAPH_TF_LOG_PLACEMENT=1`  | Generate op placement log at stdout   |
| `NGRAPH_TF_DUMP_CLUSTERS=1`  | Dump Encapsulated TF Graphs `ngraph_cluster_<cluster_num>` |
| `NGRAPH_TF_DUMP_GRAPHS=1`    | Dump TF graphs for different passes: precapture, capture, unmarked, marked, clustered, declustered, encapsulated |
| `TF_CPP_MIN_VLOG_LEVEL=1`    | Enable TF CPP logs                    |
| `NGRAPH_TF_DUMP_DECLUSTERED_GRAPHS=1` | Dump graphs with final clusters assigned. Use this to view TF computation graph with colored nodes indicating clusters|

### Selectively disable or enable ngraph

* In your script, import ngraph_bridge by using: ```import ngraph_bridge```
* Disable ngraph by calling: ```ngraph_bridge.disable()```
* Enable ngraph by calling: ```ngraph_bridge.enable()```
* Check whether ngraph is enabled by calling: ```ngraph.is_enabled()```
* You need to enable ngraph every time you called ```ngraph_bridge.disable()```, so it is good to check whether ngraph is enabled by calling ```ngraph.is_enabled()```
   * _Caution_: The above functions are only effective at the beginning of the execution. Once the session is created and ```run``` is called, the above functions will not be able to disable ngraph. 


For example usage, take a look at the ```model_test/verify_model.py``` in the diagnostics folder


## Protobuf visualization

The [tf2ngraph] script can be used to convert tensorflow graph to an ngraph enabled graph
and dump it as a protobuf (`pb`).
Tensorflow's `import_pb_tensorboard.py` script can then be used to view the
dumped graph on `tensorboard`.

### Example usage

* python tf2ngraph --input_pbtxt test_axpy.pbtxt --output_nodes add --output_pb axpy_ngraph.pb --ng_backend CPU
* Add `import ngraph_bridge` to [import_pb_to_tensorboard.py] script
* `python import_pb_to_tensorboard.py --model_dir axpy_ngraph.pb --log_dir test/`
* Launch Tensorboard by pointing it to the log directory : `tensorboard --logdir=test/`
* Goto to the link to view `.pb` as a graph.

[nGraph Library documentation]: https://ngraph.nervanasys.com/docs/latest/frameworks/generic-configs.html#activate-logtrace-related-environment-variables
[tf2ngraph]: https://github.com/tensorflow/ngraph-bridge/blob/master/tools/tf2ngraph.py
[import_pb_to_tensorboard.py]: https://github.com/tensorflow/tensorflow/blob/8ddd4429f9f7b21c7dc9312f1bad0dbf5377c615/tensorflow/python/tools/import_pb_to_tensorboard.py
