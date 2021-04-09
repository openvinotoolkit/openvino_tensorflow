# Usage of Intel<sup>®</sup> OpenVINO™ integration with TensorFlow

This document describes the available Python APIs for Intel<sup>®</sup> OpenVINO™ integration with TensorFlow. The first section describes the essential APIs and lines of code required to use the functionality of OpenVINO integration with TensorFlow applications.

## APIs for essential functionality 

To add the OpenVINO-TensorFlow package to the TensorFlow python application, import the package using the below line of code:

    import openvino_tensorflow

By default, CPU backend is enabled. You can substitute the default CPU backend with a different backend by using the following API:

    openvino_tensorflow.set_backend('backend_name')
    
Supported backends include 'CPU', 'GPU', 'MYRIAD', and 'HDDL'.
    
## Additional APIs 

To determine available backends on your system, use the following API:

    openvino_tensorflow.list_backends()
    
To check if the OpenVINO-TensorFlow is enabled, use the following API:
 
    openvino_tensorflow.is_enabled()
    
To get the assigned backend, use the following API:

    openvino_tensorflow.get_backend()
    
To enable verbose logs of the execution of the full TensorFlow pipeline and placement stages along with the OpenVINO-TensorFlow, use the following API:

    openvino_tensorflow.start_logging_placement()
    
To disbale verbose logs of the execution of the full TensorFlow pipeline and placement stages along with the OpenVINO-TensorFlow, use the following API:

    openvino_tensorflow.stop_logging_placement()
    
To check if the placement logs are enabled, use the following API:

    openvino_tensorflow.is_logging_placement()
    
To check the CXX11_ABI used to compile OpenVINO-TensorFlow, use the following API:

    openvino_tensorflow.cxx11_abi_flag()
  
To disable execution of certain operators on the OpenVINO backend, use the following API to run them on native TensorFlow runtime:

    openvino_tensorflow.set_disabled_ops(<string_of_operators_separated_by_commas>)
    
 The string of operators to be separated should be separated with commas and provided as an argument to the above API. 
    
 To check the list of disabled ops which are declared as supported by the backend, but disabled programmatically, use the following API:
 
    openvino_tensorflow.get_disabled_ops()
