# ==============================================================================
# Copyright (C) 2021 Intel Corporation

# SPDX-License-Identifier: Apache-2.0
# ==============================================================================
import unittest
import sys
import argparse
import os
import re
import fnmatch
import time
import warnings
import platform

from datetime import timedelta
from fnmatch import fnmatch

if not platform.system() == "Darwin":
    import multiprocessing
    mpmanager = multiprocessing.Manager()
    mpmanager_return_dict = mpmanager.dict()

try:
    import xmlrunner
except:
    os.system('pip install unittest-xml-reporting')
    import xmlrunner
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
        "Specify the path to Tensorflow source code. Eg:openvino_tensorflow/build_cmake/tensorflow \n",
        required=True)
    optional.add_argument(
        '--list_tests',
        help=
        "Prints the list of test cases in this package. Eg:math_ops_test.* \n")
    optional.add_argument(
        '--list_tests_from_file',
        help=
        """Reads the test names/patterns specified in a manifest file and displays a consolidated list. 
        Eg:--list_tests_from_file=tests_linux_cpu.txt""")
    optional.add_argument(
        '--run_test',
        help=
        "Runs the testcase(s), specified by name or pattern. Eg: math_ops_test.DivNoNanTest.testBasic or math_ops_test.*"
    )
    optional.add_argument(
        '--run_tests_from_file',
        help="""Reads the test names specified in a manifest file and runs them. 
        Eg:--run_tests_from_file=tests_to_run.txt""")
    optional.add_argument(
        '--xml_report',
        help=
        "Generates results in xml file for jenkins to populate in the test result \n"
    )
    optional.add_argument(
        '--verbose',
        action="store_true",
        help="Prints standard out if specified \n")
    optional.add_argument(
        '--print_support_vector',
        action="store_true",
        help=
        "Prints support vector from a device specific manifest file in True/False format\n"
    )
    optional.add_argument(
        '--timeout',
        type=int,
        default=60,
        action="store",
        help="Timeout to skip a test if it hangs\n")
    parser._action_groups.append(optional)
    arguments = parser.parse_args()

    xml_report = arguments.xml_report

    if (arguments.list_tests):
        test_list = get_test_list(arguments.tensorflow_path,
                                  arguments.list_tests)
        print('\n'.join(sorted(test_list[0])))
        print('Total:', len(test_list[0]))
        return True

    if (arguments.list_tests_from_file):
        test_list, skip_list = read_tests_from_manifest(
            arguments.list_tests_from_file, arguments.tensorflow_path)
        print('\n'.join(sorted(test_list)))
        print('Total:', len(test_list), 'Skipped:', len(skip_list))

        if (arguments.print_support_vector):
            print("\n----------------------------------\n")
            all_tests = test_list | skip_list
            for test in sorted(all_tests):
                if test in test_list:
                    print("True")
                elif test in skip_list:
                    print("False")

        return True

    if (arguments.run_test):
        invalid_list = []
        start = time.time()
        test_list = get_test_list(arguments.tensorflow_path, arguments.run_test)
        for test in test_list[1]:
            if test is not None:
                invalid_list.append(test_list[1])
                result_str = "\033[91m INVALID \033[0m " + test + \
                '\033[91m' + '\033[0m'
                print('TEST:', result_str)
        test_results = run_test(
            sorted(test_list[0]), xml_report, (2 if arguments.verbose else 0))
        elapsed = time.time() - start
        print("\n\nTesting results\nTime elapsed: ",
              str(timedelta(seconds=elapsed)))
        return check_and_print_summary(test_results, test_list[1])

    if (arguments.run_tests_from_file):
        invalid_list = []
        start = time.time()
        list_of_tests = read_tests_from_manifest(arguments.run_tests_from_file,
                                                 arguments.tensorflow_path)[0]
        test_results = run_test(
            sorted(list_of_tests), xml_report, (2 if arguments.verbose else 0),
            arguments.timeout)
        elapsed = time.time() - start
        print("\n\nTesting results\nTime elapsed: ",
              str(timedelta(seconds=elapsed)))
        return check_and_print_summary(test_results, invalid_list)

    return True


def get_test_list(tf_path, test_regex):
    accepted_formats = [
        "*test*", "math_ops_test.DivNoNanTest.testBasic",
        "math_ops_test.DivNoNanTest.*", "math_ops_test.D*", "math_ops_test.*",
        "math_*_test", "math_*_*_test", "math*_test"
    ]
    try:
        module_list = regex_walk(tf_path, test_regex)
    except Exception as e:
        module_list = []
        print(
            "Exception occured in regex_walk (" + test_regex + ") -> " +
            str(e) +
            """\nInvalid module name. Use bazel query below to get list of tensorflow python test modules.
            bazel query 'kind(".*_test rule", //tensorflow/python:nn_test)' --output label\n"""
        )
    try:
        test_list = list_tests(module_list, test_regex)
    except Exception as e:
        test_list = [[], []]
        print(
            "Exception occured in list_tests. " + str(e) +
            "\nEnter a valid argument to --list_tests or --run_test.\n \nLIST OF ACCEPTED FORMATS:"
        )
        print('\n'.join(accepted_formats))

    return test_list


def regex_walk(dirname, regex_input):
    """
    Adds all the directories under the specified dirname to the system path to 
    be able to import the modules.
    
    Args:
    dirname: This is the tensorflow_path passed as an argument is the path to 
    tensorflow source code.
    
    regex_input: Regular expression input string to filter and list/run tests.
    Few examples of accepted regex_input are:
    math_ops_test.DivNoNanTest.testBasic
    math_ops_test.DivNoNanTest.*
    math_ops_test.D*
    math_ops_test.*
    math_*_test
    math_*_*_test
    math*_test
    """
    if (re.search(r'\.', regex_input) is None):
        # a module name regex was given
        test = regex_input + '.py'
    else:
        # regex has dot(s) e.g. module.class.testfunc
        test = (re.split("\.", regex_input))[0] + '.py'
    module_list = []
    for path, subdirs, files in os.walk(dirname):
        for name in files:
            if fnmatch(name, test):
                if path not in sys.path:
                    sys.path.append(os.path.abspath(path))
                name = os.path.splitext(name)[0]
                module_list.append(name)
    if not module_list:
        print("Test pattern/name does not exist:", regex_input, "dirname",
              dirname)

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
    math_ops_test.DivNoNanTest.testBasic
    math_ops_test.DivNoNanTest.* (or math_ops_test.DivNoNanTest)
    math_ops_test.D*
    math_ops_test.*
    math_*_test
    math_*_*_test
    math*_test
    """
    loader = unittest.TestLoader()
    alltests = []
    listtests = []
    invalidtests = []
    for test_module in module_list:
        try:
            moduleobj = __import__(test_module)
        except Exception as e:
            print("Exception in __import__({})".format(test_module), 'ERROR:',
                  str(e))
            module_list.remove(test_module)
            continue
        try:
            test_suites = loader.loadTestsFromModule(moduleobj)
        except Exception as e:
            print(
                "Exception in loader.loadTestsFromModule({})".format(moduleobj),
                'ERROR:', str(e))
            module_list.remove(test_module)
            continue
        for a_testsuite in test_suites:
            for a_testcase in a_testsuite:
                alltests.append(a_testcase.id())

    # change module.class to module.class.*
    regex_input = regex_input + ('.*' if (regex_input.count('.') == 1) else '')
    regex_pattern = '^' + regex_input + '$'
    regex_pattern = re.sub(r'\.', '\\.', regex_pattern)
    regex_pattern = re.sub(r'\*', '.*', regex_pattern)
    for a_testcase_id in alltests:
        if re.search(regex_pattern, a_testcase_id):
            listtests.append(a_testcase_id)

    if not listtests:
        invalidtests.append(regex_input)

    return listtests, invalidtests


def read_tests_from_manifest(manifestfile,
                             tensorflow_path,
                             g_imported_files=set()):
    """
    Reads a file that has include & exclude patterns,
    Returns a list of leaf-level single testcase, no duplicates
    """
    run_items = set()
    skipped_items = set()
    g_imported_files.add(manifestfile)
    assert os.path.isfile(manifestfile), "Could not find the file"
    with open(manifestfile) as fh:
        curr_section = ''
        for line in fh.readlines():
            line = line.split('#')[0].rstrip('\n').strip(' ')
            if line == '':
                continue
            if re.search(r'\[IMPORT\]', line):
                curr_section = 'import_section'
                continue
            if re.search(r'\[RUN\]', line):
                curr_section = 'run_section'
                continue
            if re.search(r'\[SKIP\]', line):
                curr_section = 'skip_section'
                continue
            if curr_section == 'import_section':
                if not os.path.isabs(line):
                    line = os.path.abspath(
                        os.path.dirname(manifestfile) + '/' + line)
                if line in g_imported_files:
                    sys.exit("ERROR: re-import of manifest " + line + " in " +
                             manifestfile)
                g_imported_files.add(line)
                new_runs, new_skips = read_tests_from_manifest(
                    line, tensorflow_path, g_imported_files)
                assert (new_runs.isdisjoint(new_skips))
                run_items |= new_runs
                skipped_items |= new_skips
                run_items -= skipped_items
                continue
            if curr_section == 'run_section':
                new_runs = set(get_test_list(tensorflow_path, line)[0])
                skipped_items -= new_runs
                run_items |= new_runs
            if curr_section == 'skip_section':
                new_skips = set(get_test_list(tensorflow_path, line)[0])
                new_skips = set([x for x in new_skips if x in run_items])
                run_items -= new_skips
                skipped_items |= new_skips
        assert (run_items.isdisjoint(skipped_items))
        print('\n#Tests to Run={}, Skip={} (manifest = {})\n'.format(
            len(run_items), len(skipped_items), manifestfile))

    return run_items, skipped_items


def func_utrunner_testcase_run(return_dict, runner, a_test):
    # This func runs in a separate process
    try:
        test_result = runner.run(a_test)
        success = test_result.wasSuccessful()
        return_dict[a_test.id()] = {
            'wasSuccessful': success,
            'failures': [] if (success) else [('', test_result.failures[0][1])],
            'errors': [],
            'skipped': []
        }
    except Exception as e:
        #print('DBG: func_utrunner_testcase_run test_result.errors', test_result.errors, '\n')
        return_dict[a_test.id()] = {
            'wasSuccessful': False,
            'failures': [('', test_result.errors[0][1])],
            'errors': [('', test_result.errors[0][1])],
            'skipped': []
        }


def run_singletest_in_new_child_process(runner, a_test):
    mpmanager_return_dict.clear()
    return_dict = mpmanager_return_dict
    p = multiprocessing.Process(
        target=func_utrunner_testcase_run, args=(return_dict, runner, a_test))
    p.start()
    p.join()

    #  A negative exitcode -N indicates that the child was terminated by signal N.
    if p.exitcode != 0:
        error_msg = '!!! RUNTIME ERROR !!! Test ' + a_test.id(
        ) + ' exited with code: ' + str(p.exitcode)
        print(error_msg)
        return_dict[a_test.id()] = {
            'wasSuccessful': False,
            'failures': [('', error_msg)],
            'errors': [('', error_msg)],
            'skipped': []
        }
        return return_dict[a_test.id()]

    test_result_map = return_dict[a_test.id()]
    return test_result_map


def timeout_handler(signum, frame):
    raise Exception("Test took too long to run. Skipping.")


def run_singletest(testpattern, runner, a_test, timeout):
    # This func runs in the same process
    return_dict = {}
    import signal
    signal.signal(signal.SIGALRM, timeout_handler)

    # set timeout here
    signal.alarm(timeout)

    try:
        test_result = runner.run(a_test)
        success = test_result.wasSuccessful()
        return_dict[a_test.id()] = {
            'wasSuccessful': success,
            'failures': [] if (success) else [('', test_result.failures[0][1])],
            'errors': [],
            'skipped': []
        }
    except Exception as e:
        #print('DBG: func_utrunner_testcase_run test_result.errors', test_result.errors, '\n')
        error_msg = '!!! RUNTIME ERROR !!! Test ' + a_test.id()
        print(error_msg)
        return_dict[a_test.id()] = {
            'wasSuccessful': False,
            'failures': [('', test_result.errors[0][1])],
            'errors': [('', test_result.errors[0][1])],
            'skipped': []
        }
    return return_dict[a_test.id()]


def run_test(test_list, xml_report, timeout=60, verbosity=0):
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
    suite = unittest.TestSuite()
    succeeded = []
    failures = []
    skipped = []
    run_test_counter = 0
    if xml_report is not None:
        for testpattern in test_list:
            tests = loader.loadTestsFromName(testpattern)
            suite.addTest(tests)
        assert os.path.isfile(xml_report), "Could not find the file"
        with open(xml_report, 'wb') as output:
            sys.stdout = open(os.devnull, "w")
            sys.stderr = open(os.devnull, "w")
            test_result = xmlrunner.XMLTestRunner(
                output=output, verbosity=verbosity).run(suite)
            sys.stderr = sys.__stderr__
            sys.stdout = sys.__stdout__
            failures.extend(test_result.failures)
            failures.extend(test_result.errors)
            succeeded.extend(test_result.successes)

        summary = {"TOTAL": test_list, "PASSED": succeeded, "FAILED": failures}
        return summary
    else:
        runner = unittest.TextTestRunner(verbosity=verbosity)
        for testpattern in test_list:
            testsuite = loader.loadTestsFromName(testpattern)
            for a_test in testsuite:
                print()
                run_test_counter += 1
                print('>> >> >> >> ({}) Testing: {} ...'.format(
                    run_test_counter, a_test.id()))
                start = time.time()
                test_result_map = run_singletest(testpattern, runner, a_test,
                                                 timeout)
                elapsed = time.time() - start
                elapsed = str(timedelta(seconds=elapsed))

                if test_result_map['wasSuccessful'] == True:
                    succeeded.append(a_test.id())
                    result_str = " \033[92m OK \033[0m " + a_test.id()
                elif 'failures' in test_result_map and bool(
                        test_result_map['failures']):
                    failures.append(test_result_map['failures'])
                    result_str = " \033[91m FAIL \033[0m " + a_test.id() + \
                        '\n\033[91m' + ''.join(test_result_map['failures'][0][1]) + '\033[0m'
                elif 'errors' in test_result_map and bool(
                        test_result_map['errors']):
                    failures.append(test_result_map['errors'])
                    result_str = " \033[91m FAIL \033[0m " + a_test.id() + \
                        '\n\033[91m' + ''.join(test_result_map['errors'][0][1]) + '\033[0m'

                if 'skipped' in test_result_map and bool(
                        test_result_map['skipped']):
                    skipped.append(test_result_map['skipped'])
                print('took', elapsed, 'RESULT =>', result_str)
        summary = {
            "TOTAL": test_list,
            "PASSED": succeeded,
            "SKIPPED": skipped,
            "FAILED": failures,
        }
        return summary


def check_and_print_summary(test_results, invalid_list):
    print('========================================================')
    print("TOTAL: ", len(test_results['TOTAL']))
    print("PASSED: ", len(test_results['PASSED']))
    if len(test_results['SKIPPED']) > 0:
        print("   with skipped: ", len(test_results['SKIPPED']))
    print("FAILED: ", len(test_results['FAILED']))

    if (len(invalid_list) > 0):
        print("INVALID: ", len(invalid_list))

    print('========================================================\n')

    if len(test_results['FAILED']) == 0:
        return True
    else:
        return False


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        status = main()
        if status == False:
            raise Exception("Tests failed")
