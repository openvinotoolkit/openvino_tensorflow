#==============================================================================
#  Copyright 2019 Intel Corporation
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
# =============================================================================

import pdb, time
from subprocess import check_output, call, Popen, PIPE
import json, os, argparse, sys
import sys
# expects tools to be present at this relative location. Need access to build_utils
sys.path.insert(0, os.path.abspath('../../tools'))
from build_utils import download_repo
from tf2ngraph import get_gdef
from log_parser import parse_logs, compare_parsed_values
import atexit


def get_expected_from_json(json_file_name, configuration, strict):
    with open(json_file_name) as f:
        expected_vals = json.load(f)[configuration]
        possible_keys_1 = set(['logparse', 'time'])
        assert all([(k in possible_keys_1) for k in expected_vals])
        for k in expected_vals:
            assert k in possible_keys_1, "Got unexpected key in json: " + k + ". Expected: " + possible_keys_1
        possible_keys_2 = set([
            'num_nodes_in_graph', 'num_nodes_marked_for_clustering',
            'num_ng_clusters'
        ])
        for k in expected_vals.get('logparse', {}):
            current_keys = set(expected_vals['logparse'][k].keys())
            if strict:
                assert len(
                    current_keys.difference(possible_keys_2)
                ) == 0, "Got unexpected keys in json: " + str(
                    current_keys) + ". Expected: " + str(possible_keys_2)
        return expected_vals


def generate_functional_check_checkpoint(loc, chkpoint_save_patch, run_command):
    pass


def command_executor(cmd, verbose=False, msg=None, stdout=None, stderr=None):
    command_executor.commands += ('' if (msg is None) else
                                  '# ' + msg.strip('\n') + '\n') + cmd + '\n'
    if verbose or msg is not None:
        tag = 'Running Command: ' if msg is None else msg
        print(tag + cmd)
    if 'cd ' == cmd[:3]:
        os.chdir(cmd.split(' ')[1])
    else:
        ps = Popen(cmd, stdin=PIPE, stdout=stdout, stderr=stderr, shell=True)
        so, se = ps.communicate()
        errcode = ps.returncode
        assert errcode == 0, "Error in running command: " + cmd + ". Error message: " + se.decode(
        )
        return so, se, errcode


command_executor.commands = ''  # TODO: slightly ugly


def return_to_cwd(f):

    def _helper(*args, **kwargs):
        so, _, __ = command_executor('pwd', stdout=PIPE)
        cwd = so.decode("utf-8").strip('\n')
        try:
            retval = f(*args, **kwargs)
        finally:
            # In both cases (the call to f passes or fails), return to original directory
            command_executor('cd ' + cwd)
        return retval

    return _helper


@return_to_cwd
def apply_patch_and_test(test_folder, env_flags):
    model_dir = os.path.abspath(test_folder + '/..')
    downloaded_repo = os.path.abspath(model_dir + '/downloaded_model')
    command_executor('cd ' + model_dir)
    # To generate the patch use: git diff > enable_ngraph.patch
    patch_in_test_folder = os.path.abspath(test_folder + '/enable_ngraph.patch')
    patch_in_model_folder = os.path.abspath(test_folder +
                                            '/../enable_ngraph.patch')
    if os.path.isfile(patch_in_test_folder):
        patch_file = patch_in_test_folder
    elif os.path.isfile(patch_in_model_folder):
        patch_file = patch_in_model_folder
    else:
        patch_file = None
    assert patch_file is not None, "Did not find any patch file. Looked for " + patch_in_test_folder + ' or ' + patch_in_model_folder

    command_executor('cd ' + downloaded_repo)
    if patch_file is not None:
        command_executor('git apply ' + patch_file)

    command_executor('chmod +x ' + test_folder + '/core_run.sh')
    so, se, errcode = command_executor(
        env_flags + ' ' + test_folder + '/core_run.sh',
        msg="Running test config " + test_folder.split('/')[-1] + ': ',
        stdout=PIPE,
        stderr=PIPE)

    command_executor('git reset --hard')  # remove applied patch (if any)
    return so.decode("utf-8"), se.decode("utf-8")


@return_to_cwd
def ready_repo(model_dir, repo_dl_loc):
    command_executor('cd ' + repo_dl_loc)
    #command_executor('git reset --hard')
    # getting the repo ready is common to both check_rewrite_test and get_checkpoint
    if os.path.isfile(model_dir + '/getting_repo_ready.sh'):
        command_executor('chmod +x ' + model_dir + '/getting_repo_ready.sh')
        command_executor(model_dir + '/getting_repo_ready.sh', verbose=True)


# Currently there are 2 types of tests. parsing the logs and timing the run
def valid_test_types():
    return set(['time', 'logparse'])


# Check if the contents of this iterable contains only valid test types
def check_test_types(iterable):
    return all(map(lambda i: i in valid_test_types(), iterable))


# TODO: this function needs to accept "do-i-dump-pbtxt"? and if so, a cleanup needs to happen later.
# Also this function could return the list of pbtxts it generated (but does it need to? we can infer it)
# TODO: this function should also take the level/intensity of test to run
def run_test_suite(model_dir, configuration, disabled, print_parsed,
                   ignore_test):
    try:
        # TODO: assert TF version. Some models may not run on TF1.12 etc
        model_dir = os.path.abspath(model_dir)
        test_suites = os.listdir(model_dir)

        failed_tests = []
        passed_tests = []
        skipped_tests = []

        # download/prepare repo if needed:
        repo_filename = model_dir + '/repo.txt'
        repo_based = False  # Is this test dir repo based or pb/pbtxt/savedmodel based?
        if os.path.isfile(repo_filename):
            repo_based = True
            repo_info = [
                line.strip()
                for line in open(repo_filename).readlines()
                if len(line.strip()) > 0
            ]
            repo_name = repo_info[0]
            repo_version = repo_info[1] if len(repo_info) == 2 else 'master'
            repo_dl_loc = model_dir + '/downloaded_model'
            assert not os.path.isdir(
                repo_dl_loc
            ), "Did not expect " + repo_dl_loc + " to be present. Maybe a leftover from the last run that was not deleted?"
            download_repo(repo_dl_loc, repo_name, repo_version)
            assert os.path.isdir(
                repo_dl_loc), "Did not manage to download the repo " + repo_name
            ready_repo(model_dir, repo_dl_loc)

        # Iterate through each sub-test
        for flname in test_suites:
            sub_test_dir = model_dir + '/' + flname
            # if its  directory starting with test, and not containing "disabled" in its name
            item_is_a_subtest = not os.path.isfile(
                sub_test_dir) and flname.startswith('test')
            if item_is_a_subtest:
                disabled_by_dir_name = 'disabled' in flname
                disabled_by_cli = flname in disabled
                if (not disabled_by_dir_name) and (not disabled_by_cli):
                    custom_parser_present = os.path.isfile(
                        sub_test_dir + '/custom_log_parser.py')
                    if repo_based:
                        # TODO: shift the timing inside apply_patch_and_test
                        sub_test_dir = model_dir + '/' + flname
                        tstart = time.time()
                        try:
                            so, se = apply_patch_and_test(
                                sub_test_dir, ('NGRAPH_TF_LOG_PLACEMENT=1',
                                               '')[custom_parser_present])
                        except Exception as e:
                            print(e)
                            failed_tests.append(flname)
                            continue
                        tend = time.time()
                        command_executor.commands += '\n'
                    else:
                        model = [
                            i for i in os.listdir(sub_test_dir)
                            if '.md' not in i and '.json' not in i
                        ]
                        assert len(model) == 1
                        model = model[0]
                        split_on_dot = model.split('.')
                        assert len(split_on_dot) <= 2
                        if len(split_on_dot) == 1:
                            model_format = 'savedmodel'
                        elif split_on_dot[1] in ['pb', 'pbtxt']:
                            model_format = split_on_dot[1]
                        else:
                            assert False, "Unknown input format. Expected savedmodel, pb or pbtxt"
                        # TODO: support checkpoint too later
                        gdef = get_gdef(model_format,
                                        sub_test_dir + '/' + model)
                        # TODO: run Level1 tests on gdef. needs another json for that (one which specifies input shapes etc)

                    expected_json_file = sub_test_dir + '/expected.json'
                    expected_json_present = os.path.isfile(expected_json_file)
                    if print_parsed or expected_json_present:
                        # parse logs in this case
                        if custom_parser_present:
                            sys.path.insert(0, os.path.abspath(sub_test_dir))
                            from custom_log_parser import custom_parse_logs
                            parsed_vals = custom_parse_logs(so)
                            sys.path.pop(0)
                        else:
                            parsed_vals = parse_logs(so)
                        if print_parsed:
                            to_be_printed = {
                                configuration: {
                                    'logparse': parsed_vals,
                                    'time': tend - tstart
                                }
                            }
                            replaced_single_with_double_quotes = json.loads(
                                to_be_printed.__str__().replace("\'", "\""))
                            print(
                                json.dumps(
                                    replaced_single_with_double_quotes,
                                    sort_keys=True,
                                    indent=4,
                                    separators=(',', ': ')))
                    # If expected.json is present, run some extra tests. If not present we deem the test passed if it ran apply_patch_and_test without raising any errors
                    if expected_json_present:
                        try:
                            expected = get_expected_from_json(
                                expected_json_file, configuration,
                                not custom_parser_present)
                        except:
                            assert False, 'Failed to parse ' + expected_json_file
                        assert check_test_types(expected.keys(
                        )), "Got unexpected key in " + expected.keys(
                        ) + ". Should have been " + ','.join(valid_test_types)
                        # We run the test if 'logparse' is present in the expected values to check
                        # for and it is not in the ignore list
                        if ('logparse' in expected) and (
                                'logparse' not in ignore_test):
                            passed, fail_help_string = compare_parsed_values(
                                parsed_vals, expected['logparse'])
                            if not passed:
                                print('Failed in test ' + flname +
                                      '. Help message: ' + fail_help_string)
                                failed_tests.append(flname)
                                continue
                        if ('time' in expected) and ('time' not in ignore_test):
                            actual_runtime = tend - tstart
                            # TODO: decide this criteria. time can be pretty variable
                            # TODO: the percentage (0.1) for the time bound might be passed through `expected.json`
                            time_check = (actual_runtime - expected['time']
                                         ) / expected['time'] < 0.1
                            if not time_check:
                                print("Expected run time for test " + flname +
                                      " is " + str(expected['time']) +
                                      " but it actually took " +
                                      str(actual_runtime))
                                failed_tests.append(flname)
                                continue
                    passed_tests.append(flname)
                else:
                    skipped_tests.append(flname)
                # Make sure the test is exactly one of passed, skipped or failed
                assert sum([
                    flname in skipped_tests, flname in passed_tests,
                    flname in failed_tests
                ]) == 1, str(
                    flname
                ) + ' does not appear exactly once in passed, skipped or failed test lists'

        # Clean up if needed
        cleanup_script = model_dir + '/cleanup.sh'
        if os.path.isfile(cleanup_script):
            assert repo_based, 'Did not expect a cleanup script in non-repo based test'
            command_executor('chmod +x ' + cleanup_script)
            command_executor(cleanup_script)
        command_executor.commands += '# Exiting. Done with tests in ' + model_dir.split(
            '/')[-1]
        return passed_tests, failed_tests, skipped_tests
        # TODO: use gdef to run
        # TODO: add axpy test folders for pb. pbtxt and savedmodel
        # TODO integrate the if-else paths as much as possible

        # TODO: check throughput/latency
    except Exception as e:
        print(e)
        return passed_tests, failed_tests, skipped_tests
    finally:
        if (os.path.isdir(repo_dl_loc)):
            command_executor('rm -rf ' + repo_dl_loc)


def dump_commands_in_shellscript(dir):
    with open(dir + '/dump.sh', 'w') as f:
        f.write(command_executor.commands)


def get_test_list_string(string, disabled_test_suite, disabled_sub_test):
    available_dirs = os.listdir('./models')
    dirs_to_scan = available_dirs if string == '' else string.split(',')
    help_string = ''
    for dir in dirs_to_scan:
        assert dir in available_dirs, "Requested to list " + dir + ", but that directory is not present in available directories: " + ','.join(
            available_dirs)
        help_string += ("\033[1;32m", "\033[1;31m")[dir in disabled_test_suite]
        help_string += 'Test suite: ' + dir + '\033[0;0m\n'
        currdir = './models/' + dir
        if os.path.isfile(currdir + '/README.md'):
            with open(currdir + '/README.md') as f:
                help_string += '\n'.join(f.readlines()) + '\n'
        for c in os.listdir(currdir):
            if c.startswith('test'):
                sub_test = currdir + '/' + c
                if os.path.isdir(sub_test):
                    help_string += ("\033[1;32m",
                                    "\033[1;31m")['disabled' in c or
                                                  dir in disabled_test_suite or
                                                  c in disabled_sub_test.get(
                                                      dir, [])]
                    help_string += '\tSub test: ' + c + '\033[0;0m\n'
                    currtest_readme = sub_test + '/README.md'
                    if os.path.isfile(currtest_readme):
                        with open(currtest_readme) as f:
                            help_string += '\n'.join(
                                ['\t' + i for i in f.readlines()]) + '\n'
        help_string += '\n' + '*' * 50 + '\n\n'
    return help_string


def get_disabled_tests_info():
    available_test_suites = os.listdir('./models/')
    # TODO assert we are in model_level_tests
    disabled_test_suite = []
    disabled_sub_test = {}
    if len(args.disable) > 0:
        for item in args.disable.split(','):
            if '.' in item:
                test_suite, sub_test = item.split('.')
                assert test_suite in available_test_suites, 'Request to disable ' + item + ' but ' + test_suite + ' is not a directory in models'
                assert sub_test in os.listdir(
                    './models/' + test_suite
                ), 'Expected ' + sub_test + ' to be in ' + test_suite
                disabled_sub_test[test_suite] = disabled_sub_test.get(
                    test_suite, []) + [sub_test]
            else:
                assert item in available_test_suites, 'Request to disable ' + item + ' which is not a directory in models'
                disabled_test_suite.append(item)
    return disabled_test_suite, disabled_sub_test


if __name__ == '__main__':
    cwd = os.getcwd()
    atexit.register(dump_commands_in_shellscript, cwd)
    parser = argparse.ArgumentParser(
        description=
        'Testing framework for TF models. Performs 2 types of tests. A) run_only and B) functional'
    )

    parser.add_argument(
        '--run_basic_tests',
        action='store_true',
        help='Perform type A tests (Log parsing)')
    parser.add_argument(  # TODO: revisit this flag
        '--run_functional_tests',
        action='store_true',
        help='Perform type B tests (functional, random input)')
    # TODO: if needed we can pass an arg here that indicates the dir where all the test-suites are. Currently its assumed to be `models`
    parser.add_argument(
        '--models',
        action='store',
        type=str,
        help='Comma separated list of model names',
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help=
        'List all tests or only those tests selected by --models and --disable')
    # TODO: add some pre-set configuration types. We already have "default", add "grappler", "var-opt", etc
    parser.add_argument(
        '--configuration',
        action='store',
        type=str,
        help=
        "The configuration in which the test is run (to choose which expected values current run's results will be compared against)",
        default='default')
    parser.add_argument(
        '--disable',
        action='store',
        type=str,
        help=
        'Comma separated list of model/test-suite names or sub-test names to be disabled. Eg: "MLP,DenseNet.test1"',
        default='')
    parser.add_argument(
        '--print_parsed',
        action='store_true',
        help=
        'Print the parsed values from log parsing. Useful when checking in a new model and we want to know its expected values'
    )
    parser.add_argument(
        '--ignore_test',
        type=str,
        default=None,
        help=
        'Comma separated string. Given an expected json file, ignore these tests. Can take values "", "logparse", "time", "logparse,time", "time,logparse"'
    )

    # This script must be run from this location
    assert cwd.split('/')[-1] == 'model_level_tests'

    args = parser.parse_args()

    available_test_suites = os.listdir('./models/')

    disabled_test_suite, disabled_sub_test = get_disabled_tests_info()

    if args.list:
        print(
            get_test_list_string(args.models, disabled_test_suite,
                                 disabled_sub_test))
        exit(0)

    assert (
        args.run_basic_tests or args.run_functional_tests
    ), 'No type of test enabled. Please choose --run_basic_tests, --run_functional_tests or both'

    ignore_test = [] if (
        args.ignore_test is None) else args.ignore_test.split(',')
    assert ((ignore_test=='') or check_test_types(ignore_test)
           ), "Types of possible tests: " + ','.join(valid_test_types()) + \
    ", but requested to skip " + args.ignore_test

    requested_test_suites = os.listdir(
        'models') if args.models == '' else args.models.split(',')

    assert len(requested_test_suites) != 0, "Number of tests expected to be > 0"
    assert len(
        set(requested_test_suites).difference(set(
            available_test_suites))) == 0, "The requested tests are not present"
    assert all(
        [not os.path.isfile(i) for i in available_test_suites]
    ), "Expected that all the contents of models to be directories, but found files there"

    passed_tests = {}
    failed_tests = {}
    skipped_tests = {}
    for test_suite in requested_test_suites:
        print('\n' + '=' * 20 + 'Testing model/test-suite: ' + test_suite +
              '=' * 20)
        if test_suite not in disabled_test_suite:
            if args.run_basic_tests:
                passed_tests_in_suite, failed_tests_in_suite, skipped_tests_in_suite = run_test_suite(
                    './models/' + test_suite, args.configuration,
                    disabled_sub_test.get(test_suite, []), args.print_parsed,
                    ignore_test)
                passed_tests[test_suite] = passed_tests_in_suite
                failed_tests[test_suite] = failed_tests_in_suite
                skipped_tests[test_suite] = skipped_tests_in_suite
            if args.run_functional_tests:
                assert False, 'Functional tests not implemented yet!!'
    print_format = lambda d: '\n'.join(
        ['\n\t'.join([k] + d[k]) for k in d if len(d[k]) > 0])
    print('Passed:\n' + '\033[92m' + print_format(passed_tests) + '\033[0m')
    print('Skipped:\n' + '\033[93m' + print_format(skipped_tests) + '\033[0m')
    print('Failed:\n' + '\033[91m' + print_format(failed_tests) + '\033[0m')
    all_tests_passed = all([len(failed_tests[k]) == 0 for k in failed_tests])
    exit(0 if all_tests_passed else 1)

# TODO add a test comparing with TF run?
# TODO verbose or quiet?

# TODO: what happens in case of shrestha's change. maybe expected number of clusters etc is different for normal path and var-opt path. Can be taken care of by --configuration. However user will have to decide if grappler, then use this config. it could possibly be auto-detected

# TODO: we have a way to control which model/test-dirs run (using --models). But we do not have a flag for test "intensity".
# each subtest folder has a "enable" patch and a run command.
# Level1: These are run with "parse the NGRAPH_TF_LOG_PLACEMENT=1". These tests should be short
# Level2: Dump pbtxts and run verify models (needs an input file that specifies certain layers. (what about all layers?)). These tests should be short
# When Level1 is running, dump out pbtxts that can be used for Level2 tests
# Level3: parse prints we put. These tests are run without "NGRAPH_TF_LOG_PLACEMENT=1". the framework can provide some default parsers, but users are free to add pyscripts that provide functions for custom script parsers
# These tests can be long
# So we can offer options to do: {1}, {1,2}, {1,2,3}, {3}  (or do we allow options for any combination of tests?)
# NOTE: Level3 and Level1 test are same (mechanics wise). We have only 2 types of tests, though Level2 is unimplemented for now
