# Diagnostics tools

A collection of tools to help triage and diagnose misbehaving networks.

Notation used:
* TensorFlow: TF
* TensorBoard: TB
* protobuf binary: pb
* protobuf string text: pbtxt
* NGraph-TF: NGTF


## What to do if your network fails
**TODO**: List steps to generate logs or run diagnostic tools

### Capturing logs in text file
NGTF uses the std error to output its logs, so it is necessary to pipe it correctly to capture all logs.
```python run_TF_network.py > log.txt 2>&1```

### A full dump
To get a **full** dump use the following set of flags
```NGRAPH_ENABLE_SERIALIZE=1 NGRAPH_CPU_TRACING=1 NGRAPH_VLOG_LEVEL=5 NGRAPH_GENERATE_GRAPHS_PBTXT=1 NGRAPH_TF_LOG_PLACEMENT=1 NGRAPH_TF_VALIDATE_CLUSTER_GRAPHS=1 NGRAPH_TF_DUMP_GRAPHS=1 python run_TF_network.py > log.txt 2>&1```

### Visualizing encapsulates using TB
* Run your script with this flag: ```NGRAPH_TF_DUMP_DECLSUTERED_GRAPHS=1 python run_TF_network.py```
* Change directory to this diagnostics folder
* Run this script to parse the dumped graphs to know which encapsulate a node belongs to. At this step nodemap.pkl is created: ```python get_node_encapsulate_map.py ./path/to/folder/where/run_TF_network.py/exists/where/the/dumps/were/created/in/the/last/step/ nodemap.pkl```
* View the original network with encapsulate information in nodemap.pkl: ```python ngtf_graph_viewer.py -c nodemap.pkl ./path/to/original_network_pbtxtfile.pbtxt```. If you do not have the pbtxt of the original tensorflow graph, you can dump it from your script using [write_graph](https://www.tensorflow.org/api_docs/python/tf/train/write_graph)


## Debug flags
* ```NGRAPH_ENABLE_SERIALIZE=1```: Generate nGraph level serialized graphs .json
* ```NGRAPH_CPU_TRACING=1```: Generate nGraph level function timelines
* ```NGRAPH_VLOG_LEVEL=5```: Generate ngraph-tf logging info for different passes
* ```NGRAPH_GENERATE_GRAPHS_PBTXT=1```: Generate .pbtxt files for different phases in ngraph-tf bridge
* ```NGRAPH_TF_LOG_PLACEMENT=1```: will generate op placement log to stdout
* ```NGRAPH_TF_DUMP_CLUSTERS=1```: Dumps Encapsulated TF Graphs: ngraph_cluster_<cluster_num>
* ```NGRAPH_TF_DUMP_GRAPHS=1```: dumps TF graphs for different passes : precapture, capture, unmarked, marked, clustered, declustered, encapsulated
* ```TF_CPP_MIN_VLOG_LEVEL=1```: Enables TF CPP Logs 
* ```NGRAPH_TF_DUMP_DECLSUTERED_GRAPHS=1```: To view TF computation graph with colored nodes indicating clusters

## Protobuf Visualization
The python script ngtf_graph_viewer.py can convert a protobuf (pb or pbtxt) into a dot file or a TB log, which can be viewed using TB. If the input is a pbtxt then ngtf_graph_viewer can also sanitize node names to remove underscores from the front of node names (which indicate they are internal nodes and might cause TB to complain). It can also prepend strings in front of certain node names, a feature which can be used  to append encapsulate information for clustering nodes together

ngtf_graph_viewer has been tested on Python 2 TF-1.9, but should work with Python 3 and other versions of TF.

Run the following for detailed help:
```
python ngtf_graph_viewer.py -h
```

### Some commandline samples/usecases

* pbtxt to TB: ```python ngtf_graph_viewer.py pbtxtfile.pbtxt ./vis```
* pbtxt to dot: ```python ngtf_graph_viewer.py -v 1 pbtxtfile.pbtxt ./vis```
* pb to TB: ```python ngtf_graph_viewer.py -b pbtxtfile.pb ./vis```
* pb to dot: ```python ngtf_graph_viewer.py -b -v 1 pbtxtfile.pb ./vis```
* pbtxt to TB after prepending cluster information. See **Visualizing encapsulates using TB**: ```python ngtf_graph_viewer.py -c nodemap.pkl pbtxtfile.pbtxt ./vis```


### Some other usecases
* graphdef to dot: ```from ngtf_graph_viewer import graphdef_to_dot```
* graphdef to TB: ```from ngtf_graph_viewer import graphdef_to_tensorboard```
* pbtxt to TB: ```from ngtf_graph_viewer import protobuf_to_grouped_tensorboardt; protobuf_to_grouped_tensorboard(input_filename, dot_dir, input_binary=False)```
* pbtxt to dot: ```from ngtf_graph_viewer import protobuf_to_dot; protobuf_to_dot(input_filename, dot_dir, input_binary=False)```
* pb to TB: ```from ngtf_graph_viewer import protobuf_to_grouped_tensorboardt; protobuf_to_grouped_tensorboard(input_filename, dot_dir, input_binary=True)```
* pb to dot: ```from ngtf_graph_viewer import protobuf_to_dot; protobuf_to_dot(input_filename, dot_dir, input_binary=True)```
* pb to graphdef: ```from ngtf_graph_viewer import load_file; load_file(input_filename, input_binary=True)```
* pbtxt to graphdef: ```from ngtf_graph_viewer import load_file; load_file(input_filename, input_binary=False)```
* modify a graphdef's nodes names: ```from ngtf_graph_viewer import modify_node_names; modify_node_names(graph_def, node_map={"net1/node1":"e1/net1/node1"})```

