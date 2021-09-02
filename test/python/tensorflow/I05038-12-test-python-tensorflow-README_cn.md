## 说明

Tf\_unittest\_runner 主要用于使用 nGraph 运行 tensorflow python 单元测试

## 测试内容

- 使用 TensorFlow 构建的 nGraph 进行 Python 测试（使用[顶层文档页面](../../../README.md#option-3-using-the-upstreamed-version)操作说明中的选项 3）。

- 通过为 TensorFlow 安装补丁，使用 TensorFlow 进行 Python 测试（使用[顶层文档页面](../../../README.md#option-3-using-the-upstreamed-version)操作说明中的选项 2），如下所示：
  
  ```
  cp tf_unittest_ngraph.patch <your_virtual_env/lib/python<VERSION>/site-packages>
  cd <your_virtual_env/lib/python<VERSION>/site-packages>
  patch -p1 < tf_unittest_ngraph.patch 
  ```
  
  它将更新 `tensorflow/python/framework/test_util.py`，以便 TensorFlow Python 单元测试使用 `nGraph` 执行测试。

- 通过为 TensorFlow 安装补丁，将 nGraph 用于 grappler 并使用 TensorFlow 进行 Python 测试（使用[顶层文档页面](../../../README.md#option-3-using-the-upstreamed-version)操作说明中的选项 2），如下所示：
  
  ```
  cp tf_unittest_ovtf_with_grappler.patch <your_virtual_env/lib/python<VERSION>/site-packages>
  cd <your_virtual_env/lib/python<VERSION>/site-packages>
  patch -p1 < tf_unittest_ovtf_with_grappler.patch 
  ```
  
  它将更新 `tensorflow/python/framework/test_util.py`，以便 TensorFlow Python 单元测试使用 `nGraph` 执行测试。

## 用途

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

## 如何运行测试

- `--tensorflow_path` 是必要参数，必须传递过来，以指定 Tensorflow 源代码的位置。

- OPENVINO\_TF\_DISABLE\_DEASSIGN\_CLUSTERS=1 应始终设置，否则由于单例集群重新分配，操作可能无法前往 ngraph

- 如要获取 Tensorflow 中的测试模块列表，可使用 bazel 查询`bazel query 'kind(".*_test rule", //tensorflow/python/...)' --output label`

- 通过传递模块/测试的名称或正则表达式，可以通过一次指定一项或多项测试来运行测试。`--run_test` 参数支持的格式示例：

```math_ops_test.DivNoNanTest.testBasic
      math_ops_test.DivNoNanTest.*
      math_ops_test.D*
      math_ops_test.*
      math_*_test
      math_*_*_test
      math*_test
```

注：math\_ops\_test 仅用于示例，可以是任何可用的 tensorflow 测试模块。

- 可以在文本文件中列出带运行的测试，并将文件名传递给参数 `--run_tests_from_file` 以供运行。
- 如要在运行测试时验证 ngraph 上的算子分布，可设置 OPENVINO_TF_LOG_PLACEMENT=1