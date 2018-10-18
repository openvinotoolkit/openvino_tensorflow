# ==============================================================================
#  Copyright 2018 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ==============================================================================
import unittest
import sys
import argparse
import os
"""
tf_unittest_runner is primarily used to run tensorflow python 
unit tests using ngraph
"""


def main():
    parser = argparse.ArgumentParser()
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    required.add_argument(
        '--tensorflow_path',
        help=
        "Specify the path where Tensorflow is installed. Eg:/localdisk/skantama/tf-ngraph/tensorflow \n",
        required=True)
    optional.add_argument(
        '--list_tests',
        help="Prints the list of test cases in this package. Eg:math_ops_test \n"
    )
    optional.add_argument(
        '--run_test',
        help=
        "Runs the testcase and returns the output. Eg:math_ops_test.math_ops_test.DivNoNanTest.testBasic"
    )
    parser._action_groups.append(optional)
    arguments = parser.parse_args()

    all_dirs_to_path(arguments.tensorflow_path)

    if (arguments.list_tests):
        list_tests(arguments.list_tests)

    if (arguments.run_test):
        run_test(arguments.run_test)


from fnmatch import fnmatch


def all_dirs_to_path(dirname):
    """
    Adds all the directories under the specified dirname to the system path to 
    be able to import the modules.
    
    Args:
    dir_name: This is the tensorflow_path passed as an argument where 
    tensorflow is installed.
    """
    pattern = "*_test.py"
    for path, subdirs, files in os.walk(dirname):
        for name in files:
            if fnmatch(name, pattern):
                sys.path.append(path)


def list_tests(test_module):
    """
    Generates a list of test suites and test cases from a TF test target 
    specified. 

    Args:
    test_module: This is tensorflow test target name passed as an argument.
    Example --list_tests=math_ops_test
    To get the list of tensorflow python test modules/packages or 
    labels, query using bazel.
    bazel query 'kind(".*_test rule", //tensorflow/python/...)'
    bazel query 'kind(".*_test rule", //tensorflow/python/...)' --output package
    bazel query 'kind(".*_test rule", //tensorflow/python/...)' --output label
    """
    loader = unittest.TestLoader()
    module = __import__(test_module)
    test_modules = loader.loadTestsFromModule(module)
    alltests = []
    for test_class in test_modules:
        alltests.append(([i.id() for i in test_class._tests]))
    print('\n'.join((sorted(sum(alltests, [])))))


def run_test(test_name, verbosity=2):
    """
    Runs a specific test suite or test case given with the fully qualified 
    test name and prints stdout.

    Args:
    test_name: This is the test suite or the test case passed as an argument.
    Example: --run_test=math_ops_test.AccumulateNTest        
    verbosity: Python verbose logging is set to 2. You get the help string 
    of every test and the result.
    """
    loader = unittest.TestLoader()
    tests = loader.loadTestsFromName(test_name)
    test_result = unittest.TextTestRunner(verbosity=verbosity).run(tests)
    testsRun = 0
    tests_run = []
    failures = []
    errors = []
    if test_result.wasSuccessful():
        tests_run.append([testsRun, 1])
        sys.exit()
    elif test_result.errors:
        errors.append([test_name, test_result.errors])
        tests_run.append([testsRun, 0])
    elif test_result.failures:
        failures.append([test_name, test_result.failures])
        tests_run.append([testsRun, -1])


if __name__ == '__main__':
    main()
