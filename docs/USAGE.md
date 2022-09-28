<p>English | <a href="./USAGE_cn.md">简体中文</a></p>

# APIs and environment variables for **OpenVINO™ integration with TensorFlow**

This document describes available Python APIs for **OpenVINO™ integration with TensorFlow**. The first section covers the essential APIs and lines of code required to leverage the functionality of **OpenVINO™ integration with TensorFlow** in TensorFlow applications.

## APIs for essential functionality

To add the **OpenVINO™ integration with TensorFlow** package to your TensorFlow python application, import the package using this line of code:

    import openvino_tensorflow

[Note: This would set **CUDA_VISIBLE_DEVICES** environment variable to -1. Its previous state is restored when **OpenVINO™ integration with TensorFlow** is disabled.]

By default, CPU backend is enabled. You can set a different backend by using the following API:

    openvino_tensorflow.set_backend('<backend_name>')

Supported backends include 'CPU', 'GPU', 'GPU_FP16', 'MYRIAD', and 'VAD-M'.


## Additional APIs

To determine available backends on your system, use the following API:

    openvino_tensorflow.list_backends()

To disable **OpenVINO™ integration with TensorFlow**, use the following API:

    openvino_tensorflow.disable()

To enable **OpenVINO™ integration with TensorFlow**, use the following API:

    openvino_tensorflow.enable()

To check if the **OpenVINO™ integration with TensorFlow** is enabled, use the following API:

    openvino_tensorflow.is_enabled()

To get the assigned backend, use the following API:

    openvino_tensorflow.get_backend()

To enable verbose logs of the execution of the full TensorFlow pipeline and placement stages along with the **OpenVINO™ integration with TensorFlow**, use the following API:

    openvino_tensorflow.start_logging_placement()

To disable verbose logs of the execution of the full TensorFlow pipeline and placement stages along with the **OpenVINO™ integration with TensorFlow**, use the following API:

    openvino_tensorflow.stop_logging_placement()

To check if the placement logs are enabled, use the following API:

    openvino_tensorflow.is_logging_placement()

To check the CXX11_ABI used to compile **OpenVINO™ integration with TensorFlow**, use the following API:

    openvino_tensorflow.cxx11_abi_flag()

To disable execution of certain operators on the OpenVINO™ backend, use the following API to run them on native TensorFlow runtime:

    openvino_tensorflow.set_disabled_ops(<string_of_operators_separated_by_commas>)

 The string of operators should be separated with commas and provided as an argument to the above API.

 To check the list of disabled operators which are declared as supported by the backend, but disabled programmatically, use the following API:

    openvino_tensorflow.get_disabled_ops()

To enable or disable dynamic fallback, use the following API (When enabled, clusters causing errors during runtime can fall back to native TensorFlow although they are assigned to run on OpenVINO™).

    openvino_tensorflow.enable_dynamic_fallback()
    openvino_tensorflow.disable_dynamic_fallback()

To export the translated Intermediate Representation (IR) of the clusters to a directory, use the API below. This API will export and save the IRs from the most recently executed model as ".xml" and ".bin" files, which can be used for an OpenVINO™ application later. The first parameter to this API is the output directory. If there is any pre-existing IR file in the corresponding directory, it will ask the user to confirm before overwriting any of the older files. To disable this check, pass a "False" value as the second parameter(optional). Then, any pre-existing IR files will be overwritten without any confirmation if the IR file name is the same.

    openvino_tensorflow.export_ir("output/directory/path")

or

    openvino_tensorflow.export_ir("output/directory/path", False)

## Environment Variables  

- **OPENVINO_TF_CONVERT_VARIABLES_TO_CONSTANTS:**
This variable is disabled by default, and it freezes variables from TensorFlow's ReadVariableOp as constants during the graph translation phase. Highly recommended to enable it to ensure optimal inference latencies on eagerly executed models. Disable it when model weights are modified after loading the model for inference.

    Example:

        OPENVINO_TF_CONVERT_VARIABLES_TO_CONSTANTS="1"

- **OPENVINO_TF_BACKEND:**
Backend device name can be set using this variable. It should be set to "CPU", "GPU", "GPU_FP16", "MYRIAD", or "VAD-M".

    Example:
    
        OPENVINO_TF_BACKEND="MYRIAD"

- **OPENVINO_TF_DISABLE:**
Disables **OpenVINO™ integration with TensorFlow** if set to 1.

    Example:
    
        OPENVINO_TF_DISABLE="1"

- **OPENVINO_TF_LOG_PLACEMENT:**
If this variable is set to 1, it will print the logs related to cluster formation and encapsulation.

    Example:
    
        OPENVINO_TF_LOG_PLACEMENT="1"

- **OPENVINO_TF_MIN_NONTRIVIAL_NODES:**
This variable sets the minimum number of operators that can exist in a cluster. If the number of operators in a cluster is smaller than the specified number, the cluster will be de-assigned and all the Ops in it are executed using native TensorFlow. By default, it is calculated based on the total graph size, but it cannot be less than 6 unless it is set manually. (No performance benefit is observed by enabling very small clusters). To get a detailed cluster summary set "OPENVINO_TF_LOG_PLACEMENT" to 1. 

    Example:
    
        OPENVINO_TF_MIN_NONTRIVIAL_NODES="10"

- **OPENVINO_TF_MAX_CLUSTERS:**
This variable sets the maximum number of clusters selected for execution using OpenVINO™ backend. The clusters are selected based on the size (from highest to lowest), and this decision is made at the final stage of cluster de-assignment. Ops of remaining clusters are unmarked and are executed using native TensorFlow. Setting this environment variable is useful if there are some large clusters and a number of small clusters, and performance improves by scheduling only the large clusters using OpenVINO™ backend. To get a detailed cluster summary set "OPENVINO_TF_LOG_PLACEMENT" to 1.

    Example:
    
        OPENVINO_TF_MAX_CLUSTERS="3"

- **OPENVINO_TF_VLOG_LEVEL:**
This variable is used to print the execution logs. Setting it to 1 will print the minumum amount of details and setting it to 5 will print the most detailed logs.

    Example:
    
        OPENVINO_TF_VLOG_LEVEL="4"

- **OPENVINO_TF_DISABLED_OPS:**
A list of disabled operators can be passed using this variable. These operators will not be considered for clustering and they will fall back on to native TensorFlow.

    Example:
    
        OPENVINO_TF_DISABLED_OPS="Squeeze,Greater,Gather,Unpack"

- **OPENVINO_TF_DUMP_GRAPHS:**
Setting this will serialize the full graphs in all stages during the optimization pass and save them in the current directory.

    Example:
    
        OPENVINO_TF_DUMP_GRAPHS="1"

- **OPENVINO_TF_DUMP_CLUSTERS:**
Setting this variable to 1 will serialize all the clusters in ".pbtxt" format and save them in the current directory.

    Example:
    
        OPENVINO_TF_DUMP_CLUSTERS="1"

- **OPENVINO_TF_ENABLE_BATCHING:**
If this parameter is set to 1 while using VAD-M as the backend, the backend engine will divide the input into multiple asynchronous requests to utilize all devices in VAD-M to achieve better performance.

    Example:
    
        OPENVINO_TF_ENABLE_BATCHING="1"

- **OPENVINO_TF_DYNAMIC_FALLBACK**
This variable enables or disables dynamic fallback feature. Should be set to "0" to disable and "1" to enable dynamic fallback. When enabled, clusters causing errors during runtime can fallback to native TensorFlow although they are assigned to run on OpenVINO™. Enabled by default.

    Example:
    
        OPENVINO_TF_DYNAMIC_FALLBACK="0"
    
- **OPENVINO_TF_CONSTANT_FOLDING:**
This will enable/disable constant folding pass on the translated clusters (Disabled by default).

    Example:
    
        OPENVINO_TF_CONSTANT_FOLDING="1"

- **OPENVINO_TF_TRANSPOSE_SINKING:**
This will enable/disable transpose sinking pass on the translated clusters (Enabled by default).

    Example:
    
        OPENVINO_TF_TRANSPOSE_SINKING="0"

- **OPENVINO_TF_DISABLE_DEASSIGN_CLUSTERS:**
After clusters are formed, some of the clusters may still fall back to native TensorFlow (e.g a cluster is too small, some conditions are not supported by the target device). If this variable is set, clusters will not be dropped and forced to run on OpenVINO™ backend. This may reduce the performance gain or may lead the execution to crash in some cases.

    Example:
    
        OPENVINO_TF_DISABLE_DEASSIGN_CLUSTERS="1"

- **OPENVINO_TF_DISABLE_TFFE:**
Starting from **OpenVINO™ integration with TensorFlow 2.2.0** release, TensorFlow operations are converted by [TensorFlow Frontend](https://github.com/openvinotoolkit/openvino/tree/master/src/frontends/tensorflow) to the latest available [Operation Set](https://docs.openvino.ai/latest/openvino_docs_ops_opset.html) by OpenVINO™ toolkit except some exceptional cases. By setting **OPENVINO_TF_DISABLE_TFFE** to **1**, TensorFlow Frontend can be disabled. In that case, TensorFlow Importer (the default translator of **OpenVINO™ integration with TensorFlow 2.1.0** and earlier) will be used to translate TensorFlow operations for all backends. If this environment variable is set to **0**, TensorFlow Frontend will be enabled for all backends. As of **OpenVINO™ integration with TensorFlow 2.2.0** release, this environment variable is effective only on Ubuntu and Windows platforms and TensorFlow Frontend is not supported on MacOS yet. The table below shows the translation modules used for each backend and platform by default for **OpenVINO™ integration with TensorFlow 2.2.0**.

    |             | **CPU**     | **GPU**     | **GPU_FP16** | **MYRIAD**  | **VAD-M**   |                                                                |
    |-------------|-------------|-------------|--------------|-------------|-------------|----------------------------------------------------------------|
    | **Ubuntu**  | TF Frontend | TF Frontend | TF Frontend  | TF Importer | TF Importer | _Environment variable changes the default translator_ |
    | **Windows** | TF Frontend | TF Frontend | TF Frontend  | TF Importer | TF Importer | _Environment variable changes the default translator_ |
    | **MacOS**   | TF Importer | TF Importer | TF Importer  | TF Importer | TF Importer | _Environment variable is not effective_                       |
    
    Example:
    
        OPENVINO_TF_DISABLE_TFFE="1"

- **OPENVINO_TF_MODEL_CACHE_DIR:**
Using this environment variable, a cache directory for [OpenVINO™ model caching](https://docs.openvino.ai/latest/openvino_docs_OV_UG_Model_caching_overview.html). Reusing cached model can reduce the model compile time which impacts the first inference latency using **OpenVINO™ integration with TensorFlow**. Model caching is disabled by default. To enable it, the cache directory should be specified using this environment variable.

    Example:
    
        OPENVINO_TF_MODEL_CACHE_DIR=path/to/model/cache/directory

- **OPENVINO_TF_ENABLE_OVTF_PROFILING:**
When this environment variable is set to **1**, additional performance timing information will be printed as part of verbose logs. This environment variable should be used with **OPENVINO_TF_VLOG_LEVEL** environment variable and it is only effective when verbose log level is set to **1** or greater.

    Example:
    
        OPENVINO_TF_VLOG_LEVEL=1
        OPENVINO_TF_ENABLE_OVTF_PROFILING=1

- **OPENVINO_TF_ENABLE_PERF_COUNT:**
This environment variable is used to print operator level performance counter information. This is only supported by the CPU backend.

    Example:
    
        OPENVINO_TF_ENABLE_PERF_COUNT=1

## GPU Precision

The default precision for Intel<sup>®</sup> Integrated GPU (iGPU) is FP32. So, if you set the backend name as **'GPU'**, the execution on iGPU will be operated on FP32 precision. To change the iGPU precision to FP16, use the device name **'GPU_FP16'**.

Example for setting the iGPU precision to FP32:

    openvino_tensorflow.set_backend('GPU')

or

    OPENVINO_TF_BACKEND="GPU"

Example for setting the iGPU precision to FP16:

    openvino_tensorflow.set_backend('GPU_FP16')

or

    OPENVINO_TF_BACKEND="GPU_FP16"
