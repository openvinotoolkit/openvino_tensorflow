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
```NGRAPH_ENABLE_SERIALIZE=1 NGRAPH_CPU_TRACING=1 OPENVINO_TF_VLOG_LEVEL=5 OPENVINO_TF_LOG_PLACEMENT=1 OPENVINO_TF_DUMP_CLUSTERS=1 OPENVINO_TF_DUMP_GRAPHS=1 python run_TF_network.py > log.txt 2>&1```


## Debug flags

|Name                          |Description                            |
|------------------------------|---------------------------------------|
| `OPENVINO_TF_VLOG_LEVEL=5`     | Generate ngraph-tf logging info for different passes|
| `OPENVINO_TF_LOG_PLACEMENT=1`  | Generate op placement log at stdout   |
| `OPENVINO_TF_DUMP_CLUSTERS=1`  | Dump Encapsulated TF Graphs `ngraph_cluster_<cluster_num>` |
| `OPENVINO_TF_DUMP_GRAPHS=1`    | Dump TF graphs for different passes: precapture, capture, unmarked, marked, clustered, declustered, encapsulated |
| `TF_CPP_MIN_VLOG_LEVEL=1`    | Enable TF CPP logs                    |
|

### Visualizing encapsulates using TB

* Run your script with this flag: ```OPENVINO_TF_DUMP_GRAPHS=1 python run_TF_network.py```
* Change directory to this diagnostics folder
* Run this script to parse the dumped graphs to know which encapsulate a node belongs to. At this step nodemap.pkl is created: ```python get_node_encapsulate_map.py ./path/to/folder/where/run_TF_network.py/exists/where/the/dumps/were/created/in/the/last/step/ nodemap.pkl```
* Modify the graphdef and dump TB file in ```./vis``` using encapsulate information in ```nodemap.pkl```: ```python ngtf_graph_viewer.py -c nodemap.pkl ./path/to/original_network_pbtxtfile.pbtxt ./vis```. If you do not have the pbtxt of the original tensorflow graph, you can dump it from your script using [write_graph](https://www.tensorflow.org/api_docs/python/tf/train/write_graph)
*  View the original network with encapsulate information by running tensorboard, using the files created in ```./vis```.
* Note: you may need to preprocess your protobuffers (pbtxt) to remove the `_class` attribute if any of the above steps result in an error such as `{NodeType} expects to be colocated with unknown node ...`. In that case, first run ```python remove_protobuf_class_attribute.py -d /path/to/pbtxt/to/process/``` and then follow the above steps again.

### Selectively disable or enable ngraph

* In your script, import ngraph_bridge by using: ```import ngraph_bridge```
* Disable ngraph by calling: ```ngraph_bridge.disable()```
* Enable ngraph by calling: ```ngraph_bridge.enable()```
* Check whether ngraph is enabled by calling: ```ngraph.is_enabled()```
* You need to enable ngraph every time you called ```ngraph_bridge.disable()```, so it is good to check whether ngraph is enabled by calling ```ngraph.is_enabled()```
   * _Caution_: The above functions are only effective at the beginning of the execution. Once the session is created and ```run``` is called, the above functions will not be able to disable ngraph. 


For example usage, take a look at the ```model_test/verify_model.py``` in the diagnostics folder


## Protobuf visualization

A `ngtf_graph_viewer` script can convert a protobuf (`pb` or `pbtxt`) into a 
dot file or a TB log, which can be viewed using TB. If the input is a `pbtxt`, 
it can also sanitize node names to remove underscores from the front of node 
names. It can also prepend strings in front of certain node names, a feature 
which can be used to append encapsulate information for clustering nodes 
together. (Tested on Python 2 TF-1.9; however, it should work with Python 3 
and other versions of TF.)

Run the following for detailed help:

    python ngtf_graph_viewer.py -h


### Examples for command-line use

* `pbtxt` to TB: ```python ngtf_graph_viewer.py pbtxtfile.pbtxt ./vis```
* `pbtxt` to dot: ```python ngtf_graph_viewer.py -v 0 pbtxtfile.pbtxt ./vis```
* `pb` to TB: ```python ngtf_graph_viewer.py -b pbtxtfile.pb ./vis```
* `pb` to dot: ```python ngtf_graph_viewer.py -b -v 0 pbtxtfile.pb ./vis```
* `pbtxt` to TB (after prepending cluster information): ```python ngtf_graph_viewer.py -c nodemap.pkl pbtxtfile.pbtxt ./vis``` 


### Some other use cases

* graphdef to dot: ```from ngtf_graph_viewer import graphdef_to_dot```
* graphdef to TB: ```from ngtf_graph_viewer import graphdef_to_tensorboard```
* pbtxt to TB: ```from ngtf_graph_viewer import protobuf_to_grouped_tensorboardt; protobuf_to_grouped_tensorboard(input_filename, dot_dir, input_binary=False)```
* pbtxt to dot: ```from ngtf_graph_viewer import protobuf_to_dot; protobuf_to_dot(input_filename, dot_dir, input_binary=False)```
* pb to TB: ```from ngtf_graph_viewer import protobuf_to_grouped_tensorboardt; protobuf_to_grouped_tensorboard(input_filename, dot_dir, input_binary=True)```
* pb to dot: ```from ngtf_graph_viewer import protobuf_to_dot; protobuf_to_dot(input_filename, dot_dir, input_binary=True)```
* pb to graphdef: ```from ngtf_graph_viewer import load_file; load_file(input_filename, input_binary=True)```
* pbtxt to graphdef: ```from ngtf_graph_viewer import load_file; load_file(input_filename, input_binary=False)```
* modify a graphdef's nodes names: ```from ngtf_graph_viewer import modify_node_names; modify_node_names(graph_def, node_map={"net1/node1":"e1/net1/node1"})```

[nGraph Library documentation]: https://ngraph.nervanasys.com/docs/latest/frameworks/generic-configs.html#activate-logtrace-related-environment-variables
[ngtf_graph_viewer]: https://github.com/tensorflow/ngraph-bridge/blob/master/diagnostics/ngtf_graph_viewer.py
