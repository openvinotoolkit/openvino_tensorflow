  
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

## Usage

    usage: tf_unittest_runner.py [-h] --tensorflow_path TENSORFLOW_PATH
    
    [--list_tests LIST_TESTS] [--run_test RUN_TEST]
    
    [--run_tests_from_file RUN_TESTS_FROM_FILE]
    
      
    required arguments:
    
    --tensorflow_path TENSORFLOW_PATH
    
    Specify the path to Tensorflow source code.
    
    Eg:/localdisk/skantama/tf-ngraph/tensorflow
    
    
    optional arguments:
    
    -h, --help show this help message and exit
    
    --list_tests LIST_TESTS
    
    Prints the list of test cases in this package.
    
    Eg:math_ops_test
    
    --run_test RUN_TEST Runs the testcase and returns the output.
    
    Eg:math_ops_test.DivNoNanTest.testBasic
    
    --run_tests_from_file RUN_TESTS_FROM_FILE
    
    Reads the test names specified in a file and runs
    
    them. Eg:--run_tests_from_file=tests_to_run.txt

  

## How to run tests

 - `--tensorflow_path` is a required argument and must be passed to
   specify the location of Tensorflow source code
 
 -  NGRAPH_TF_DISABLE_DEASSIGN_CLUSTERS=1 should be set always, else ops might not land
    on ngraph due to reassignment of singleton clusters
    
 -  To get a list of test modules available in Tensorflow, use bazel query
    ```bazel query 'kind(".*_test rule", //tensorflow/python/...)' --output label```
   
 - Tests can be run by specifying one or multiple tests at a time by
   passing the name of the module/test or regular expressions. Few examples of
   supported formats by `--run_test` argument :
 ``` math_ops_test 
       math_ops_test.DivNanTest
       math_ops_test.DivNoNanTest.testBasic
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
 -  To verify the Op placement on ngraph while running the tests set NGRAPH_TF_LOG_PLACEMENT=1 

