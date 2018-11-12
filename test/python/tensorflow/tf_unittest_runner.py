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
import re
import fnmatch
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
        "Runs the testcase and returns the output. Eg:math_ops_test.DivNoNanTest.testBasic"
    )
    optional.add_argument(
        '--run_tests_from_file',
        help="""Reads the test names specified in a file and runs them. 
        Eg:--run_tests_from_file=tests_to_run.txt""")
    parser._action_groups.append(optional)
    arguments = parser.parse_args()

    if (arguments.list_tests):
        test_list = get_test_list(arguments.tensorflow_path,
                                  arguments.list_tests)
        print('\n'.join(test_list))
    if (arguments.run_test):
        test_list = get_test_list(arguments.tensorflow_path, arguments.run_test)
        print('\n'.join(test_list))
        status_list = run_test(test_list)
        print_results(status_list)
    if (arguments.run_tests_from_file):
        all_test_list = []
        list_of_tests = read_tests_from_file(arguments.run_tests_from_file)
        for test in list_of_tests:
            test_list = get_test_list(arguments.tensorflow_path, test)
            test_list = list(set(test_list))
            for test_name in test_list:
                all_test_list.append(test_name)
            print('\n'.join(all_test_list))
        status_list = run_test(all_test_list)
        print_results(status_list)


def get_test_list(tf_path, test_regex):
    accepted_formats = [
        "math_ops_test", "mat_ops_test.DivNoNanTest",
        "math_ops_test.DivNoNanTest.testBasic", "math_ops_test.DivNoNanTest.*",
        "math_ops_test.D*", "math_ops_test.*", "math_*_test", "math_*_*_test",
        "math*_test"
    ]
    try:
        module_list = regex_walk(tf_path, test_regex)
    except:
        module_list = []
        print(
            """\nInvalid module name. Use bazel query below to get list of tensorflow python test modules.
            bazel query 'kind(".*_test rule", //tensorflow/python:nn_test)' --output label\n"""
        )
    try:
        test_list = list_tests(module_list, test_regex)
    except:
        test_list = []
        print(
            "\nEnter a valid argument to --list_tests or --run_test.\n \nLIST OF ACCEPTED FORMATS:"
        )
        print('\n'.join(accepted_formats))
    return test_list


from fnmatch import fnmatch


def regex_walk(dirname, regex_input):
    """
    Adds all the directories under the specified dirname to the system path to 
    be able to import the modules.
    
    Args:
    dirname: This is the tensorflow_path passed as an argument where 
    tensorflow is installed.
    
    regex_input: Regular expression input string to filter and list/run tests.
    Few examples of accepted regex_input are:
    math_ops_test
    math_ops_test.DivNanTest
    math_ops_test.DivNoNanTest.testBasic
    math_ops_test.DivNoNanTest.*
    math_ops_test.D*
    math_ops_test.*
    math_*_test
    math_*_*_test
    math*_test
    """
    if (re.search("\.", regex_input) is None):
        test = regex_input + '.py'
    else:
        test = (re.split("\.", regex_input))[0] + '.py'
    module_list = []
    for path, subdirs, files in os.walk(dirname):
        for name in files:
            if fnmatch(name, test):
                sys.path.append(path)
                name = os.path.splitext(name)[0]
                module_list.append(name)
    if not module_list:
        sys.exit()
    return module_list


def list_tests(module_list, regex_input):
    """
    Generates a list of test suites and test cases from a TF test target 
    specified. 

    Args:
    module_list: This is a list tensorflow test target names passed as an argument.
    Example --list_tests=math_ops_test.R*
    To get the list of tensorflow python test modules, query using bazel.
    bazel query 'kind(".*_test rule", //tensorflow/python/...)' --output label

    regex_input: Regular expression input strings to filter and list tests. 
    Few examples of accepted regex_input are:
    math_ops_test
    math_ops_test.DivNanTest
    math_ops_test.DivNoNanTest.testBasic
    math_ops_test.DivNoNanTest.*
    math_ops_test.D*
    math_ops_test.*
    math_*_test
    math_*_*_test
    math*_test
    """
    loader = unittest.TestLoader()
    alltests = []
    for test_module in module_list:
        module = __import__(test_module)
        if (module is None):
            print("Enter a valid test name to run")
        test_modules = loader.loadTestsFromModule(module)
        for test_class in test_modules:
            for i in test_class:
                alltests.append(i.id())

    if (re.search("\.", regex_input) is None):
        return alltests
    else:
        test_name = (re.split("\*", regex_input))[0]
        listtests = []
        for test in alltests:
            if test_name in test:
                listtests.append(test)
        if not listtests:
            print("\nTest is not a part of this test suite\n")
            sys.exit()
        return listtests


def read_tests_from_file(filename):
    with open(filename) as list_of_tests:
        return [
            line.split('#')[0].rstrip('\n').strip(' ')
            for line in list_of_tests.readlines()
            if line[0] != '#'
        ]


def run_test(test_list, verbosity=2):
    """
    Runs a specific test suite or test case given with the fully qualified 
    test name and prints stdout.

    Args:
    test_list: This is the list of tests to run,filtered based on the 
    regex_input passed as an argument.
    Example: --run_test=math_ops_test.A*   
    verbosity: Python verbose logging is set to 2. You get the help string 
    of every test and the result.
    """
    loader = unittest.TestLoader()
    succeeded = []
    failures = []
    errors = []
    for test in test_list:
        test_result = unittest.TextTestRunner(verbosity=verbosity).run(
            loader.loadTestsFromName(test))
        if test_result.wasSuccessful():
            succeeded.append(test)
        elif test_result.failures:
            failures.append(test)
        elif test_result.errors:
            errors.append(test)
    summary = {"PASSED": succeeded, "FAILED": failures, "ERRORS": errors}
    return summary


def print_results(status_list):
    print('\033[1m' + '\n==SUMMARY==' + '\033[0m')
    for key in ["PASSED", "ERRORS", "FAILED"]:
        test_name = status_list[key]
        for test in test_name:
            if key is "PASSED":
                print(test + '\033[92m' + ' ..PASS' + '\033[0m')
            if key is "FAILED":
                print(test + '\033[91m' + ' ..FAIL' + '\033[0m')
            if key is "ERRORS":
                print(test + '\033[33m' + ' ..ERROR' + '\033[0m')

    print('\033[1m' + '\n==STATS==' + '\033[0m')
    for key in ["PASSED", "ERRORS", "FAILED"]:
        test_class_name = {}
        test_name = status_list[key]
        for test in test_name:
            module, classname, testcase = test.split('.')
            module_classname = module + '.' + classname
            test_class_name[module_classname] = test_class_name.get(
                module_classname, 0) + 1
        for k in test_class_name:
            print('Number of tests ' + key + ' ' + k, test_class_name[k])


if __name__ == '__main__':
    main()
