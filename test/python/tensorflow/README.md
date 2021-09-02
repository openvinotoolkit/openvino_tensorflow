<p>English | <a href="https://github.com/openvino_tensorflow/test/python/tensorflow/I05038-12-test-python-tensorflow-README_cn.md">简体中文</a></p>


## Description
tf_unittest_runner is primarily used to run tensorflow python unit tests using nGraph

## What can be tested

 - Python tests using nGraph built with TensorFlow (using Option 3 in the instructions [top level documentation page](../../../README.md#option-3-using-the-upstreamed-version)). 
 - Python tests using Tensorflow (using Option 2 in the instructions in the [top level documentation page](../../../README.md#option-2-build-ngraph-bridge-from-source-using-tensorflow-source)) by patching 
   TensorFlow as follows:
   ```
   cp tf_unittest_ngraph.patch <your_virtual_env/lib/python<VERSION>/site-packages>
   cd <your_virtual_env/lib/python<VERSION>/site-packages>
   patch -p1 < tf_unittest_ngraph.patch 
   ```
   
   This will update the `tensorflow/python/framework/test_util.py` so that the TensorFlow Python unit tests use `nGraph` to execute the tests.
 - Python tests using nGraph with grappler and using Tensorflow (using Option 2 in the instructions in the [top level documentation page](../../../README.md#option-2-build-ngraph-bridge-from-source-using-tensorflow-source)) by patching 
   TensorFlow as follows:
   ```
   cp tf_unittest_ovtf_with_grappler.patch <your_virtual_env/lib/python<VERSION>/site-packages>
   cd <your_virtual_env/lib/python<VERSION>/site-packages>
   patch -p1 < tf_unittest_ovtf_with_grappler.patch 
   ```
   
   This will update the `tensorflow/python/framework/test_util.py` so that the TensorFlow Python unit tests use `nGraph` to execute the tests.

## Usage

    usage: tf_unittest_runner.py [-h] --tensorflow_path TENSORFLOW_PATH

    [--list_tests TEST_PATTERN] [--list_tests_from_file MANIFEST_FILE]

    [--run_test TEST_PATTERN]  [--run_tests_from_file MANIFEST_FILE]

    [--xml_report XML_REPORT] [--verbose]
      
    required arguments:
    
    --tensorflow_path TENSORFLOW_PATH
    
    Specify the path to Tensorflow source code. Eg: /localdisk/skantama/tf-ngraph/tensorflow
    
    
    optional arguments (choose one of these):
    
    -h, --help show this help message and exit
    
    --list_tests TEST_PATTERN
    Prints the list of test cases in this package. Eg: --list_tests math_ops_test.\*

    --list_tests_from_file MANIFEST_FILE
    Reads the test names/patterns specified in a manifest file and displays a consolidated list. Eg: --list_tests_from_file ./test/python/tensorflow/tests_common.txt
    
    --run_test TEST_PATTERN
    Runs the testcase and returns the output. Eg: --run_test math_ops_test.DivNoNanTest.\*
    
    --run_tests_from_file MANIFEST_FILE
    Reads the test names specified in a file and runs them. Eg: --run_tests_from_file=/path/to/tests_list_file.txt
    Please see comments in `tests_common.txt` to understand the accepted formats of the file.

    --xml_report XML_REPORT
    Generates results in xml file for jenkins to populate in the test result. Need to specify xml file name.

    --verbose
    Prints standard out if specified


## How to run tests

 - `--tensorflow_path` is a required argument and must be passed to
   specify the location of Tensorflow source code
 
 -  OPENVINO_TF_DISABLE_DEASSIGN_CLUSTERS=1 should be set always, else ops might not land
    on ngraph due to reassignment of singleton clusters
    
 -  To get a list of test modules available in Tensorflow, use bazel query
    ```bazel query 'kind(".*_test rule", //tensorflow/python/...)' --output label```
   
 - Tests can be run by specifying one or multiple tests at a time by
   passing the name of the module/test or regular expressions. Few examples of
   supported formats by `--run_test` argument :
 ```  math_ops_test.DivNoNanTest.testBasic
       math_ops_test.DivNoNanTest.*
       math_ops_test.D*
       math_ops_test.*
       math_*_test
       math_*_*_test
       math*_test
   ```
   Note:math_ops_test is used just for an example, it could be any of the availble tensorflow test module
   
 -  List of tests to run can be listed in a text file and pass the file name 
     to  argument `--run_tests_from_file` to run. 
 -  To verify the Op placement on ngraph while running the tests set OPENVINO_TF_LOG_PLACEMENT=1 

