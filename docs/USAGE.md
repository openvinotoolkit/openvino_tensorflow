# Usage of Intel<sup>®</sup> OpenVINO™ Add-on for TensorFlow

This document describes the available Python APIs for Intel<sup>®</sup> OpenVINO™ Add-on for TensorFlow. The first section describes the essential APIs and lines of code needed for achieving the functionality of OpenVINO Add-on in TensorFlow applications.

## APIs for essential functionality 

To add the OpenVINO Add-on package to the TensorFlow python application, import the package using the below line of code:

    import openvino_tensorflow

By default, CPU backend is enabled. You can substitute the default CPU backend with a different backend by using the following API:

    openvino_tensorflow.set_backend('backend_name')
    
Supported backends include 'CPU', 'GPU', 'MYRIAD', and 'HDDL'.
    
## Additional APIs 

To determine available backends on your system, use the following API:

    openvino_tensorflow.list_backends()
    
To check if the OpenVINO Add-on is enabled, use the following API:
 
    openvino_tensorflow.is_enabled()
    
To get the assigned backend, use the following API:

    openvino_tensorflow.get_backend()
    
To enable verbose logs for the execution of the full TensorFlow pipeline and placement stages along with the OpenVINO TensorFlow Add-on, use the following API:

    openvino_tensorflow.start_logging_placement()
    
To disbale verbose logs for the execution of the full TensorFlow pipeline and placement stages along with the OpenVINO TensorFlow Add-on, use the following API:

    openvino_tensorflow.stop_logging_placement()
    
To check if the placement logs are enabled, use the following API:

    openvino_tensorflow.is_logging_placement()
    
To check the version of OpenVINO Add-on, use the following API:

    openvino_tensorflow.__version__()
    
To check the CXX11_ABI used to compile OpenVINO Add-on, use the following API:

    openvino_tensorflow.cxx11_abi_flag()
    
To check if OpenVINO Add-on is registered as a Grappler optimizer, use the following API:

    openvino_tensorflow.is_grappler_enabled()
    
To disable execution of certain operators on the OpenVINO backend, use the following API to run them on native TensorFlow runtime:

    openvino_tensorflow.set_disabled_ops(<string_of_operators_separated_by_commas>)
    
 The string of operators to be separated should be separated with commas and provided as an argument to the above API. 
    
 To check the list of disabled ops which are declared as supported by the backend, but disabled programmatically, use the following API:
 
    openvino_tensorflow.get_disabled_ops()
