# ==============================================================================
# Copyright (C) 2021 Intel Corporation

# SPDX-License-Identifier: Apache-2.0
# ==============================================================================
import os
import re
import pytest
from tensorflow.python.framework import ops

# Note: to see a list of tests, run:
# .../test/python$ python -m pytest --collect-only <optional-args-to-specify-tests>
# e.g. ROOT=/localdisk/WS1 PYTHONPATH=$ROOT:$ROOT/test/python:$ROOT/tools:$ROOT/examples:$ROOT/examples/mnist python -m pytest --collect-only test_elementwise_ops.py


@pytest.fixture(autouse=True)
def reset_graph():
    yield
    ops.reset_default_graph()


@pytest.fixture(scope='session', autouse=True)
def cleanup():
    yield


def pattern_to_regex(pattern):
    no_param = (re.search(r'\[.*\]$', pattern) is None)
    pattern_noparam = re.sub(r'\[.*$', '', pattern)
    pattern = re.sub(r'\-', '\\-', pattern)
    pattern = re.sub(r'\.', '\\.', pattern)
    pattern = re.sub(r'\[', '\\[', pattern)
    pattern = re.sub(r'\]', '\\]', pattern)
    pattern = re.sub(r'\*', '.*', pattern)
    # special case for M.C.F when it possibly matches with parameterized tests
    if pattern_noparam.count('.') == 2 and no_param:
        if no_param:
            pattern = '^' + pattern + '$'
        else:
            pattern = '^' + pattern + r'\[.*'
    if pattern_noparam.count('.') == 0:
        pattern = '^' + pattern + r'\..*\..*' + '$'
    if pattern_noparam.count('.') == 1:
        pattern = '^' + pattern + r'\..*' + '$'
    return pattern


def testfunc_matches_manifest(item, pattern):
    itemfullname = get_item_fullname(item)  # Module.Class.Func
    if pattern == itemfullname:  # trivial case
        return True
    pattern = pattern_to_regex(pattern)
    if re.search(pattern, itemfullname):
        return True
    return False


# must be called after pytest.all_test_items has been set
def list_matching_tests(manifest_line):
    items = set()
    if (not pytest.all_test_items) or (len(pytest.all_test_items) == 0):
        return items
    for item in pytest.all_test_items:
        # item: Function type
        if testfunc_matches_manifest(item, manifest_line):
            items.add(get_item_fullname(item))
    return items


def read_tests_from_manifest(manifestfile):
    """
    Reads a file that has include & exclude patterns,
    Returns a set of leaf-level single testcase, no duplicates
    """
    run_items = set()
    skipped_items = set()
    pytest.g_imported_files.add(manifestfile)
    assert os.path.exists(manifestfile), "File doesn't exist {0}".format(manifestfile)
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
                if line in pytest.g_imported_files:
                    sys.exit("ERROR: re-import of manifest " + line + " in " +
                             manifestfile)
                pytest.g_imported_files.add(line)
                new_runs, new_skips = read_tests_from_manifest(line)
                assert new_runs.isdisjoint(new_skips)
                run_items |= new_runs
                skipped_items |= new_skips
                run_items -= skipped_items
                continue
            if curr_section == 'run_section':
                new_runs = list_matching_tests(line)
                skipped_items -= new_runs
                run_items |= new_runs
            if curr_section == 'skip_section':
                new_skips = list_matching_tests(line)
                run_items -= new_skips
                skipped_items |= new_skips
        assert run_items.isdisjoint(skipped_items)
        print('#Tests to Run={}, Skip={} (manifest = {})'.format(
            len(run_items), len(skipped_items), manifestfile))

    return run_items, skipped_items  # 2 sets


# item -> Function
def get_item_fullname(item):
    return item.module.__name__ + "." + item.cls.__qualname__ + "." + item.name


def attach_run_markers():
    for item in pytest.all_test_items:
        itemfullname = get_item_fullname(item)
        if itemfullname in pytest.tests_to_run or itemfullname not in pytest.tests_to_skip:
            item.add_marker(pytest.mark.temp_run_via_manifest)


#--------------------------------------------------------------------------
#--------------------------------------------------------------------------


# PyTestHook: ahead of command line option parsing
def pytest_cmdline_preparse(args):
    if 'OPENVINO_TF_TEST_MANIFEST' in os.environ:
        args[:] = ["-m", 'temp_run_via_manifest'] + args


# PyTestHook: called at early stage of pytest setup
def pytest_configure(config):
    if 'OPENVINO_TF_TEST_MANIFEST' in os.environ:
        pytest.tests_to_skip = set()
        pytest.tests_to_run = set()
        pytest.g_imported_files = set()

        print("\npytest args=", config.invocation_params.args, "dir=",
              config.invocation_params.dir)
        pytest.arg_collect_only = (
            '--collect-only' in config.invocation_params.args)

        # register an additional marker
        config.addinivalue_line(
            "markers",
            "temp_run_via_manifest: temporarily mark test to run via manifest filters"
        )


# PyTestHook: called after collection has been performed, but
# we may modify or re-order the items in-place
def pytest_collection_modifyitems(items):
    if 'OPENVINO_TF_TEST_MANIFEST' in os.environ:
        # Get list of tests to run/skip
        filename = os.path.abspath(os.environ['OPENVINO_TF_TEST_MANIFEST'])
        pytest.all_test_items = items
        print('\nChecking manifest...')
        pytest.tests_to_run, pytest.tests_to_skip = read_tests_from_manifest(
            filename)

        attach_run_markers()
        # summary
        print("\n\nTotal Available Tests:", len(items))
        print("Enabled via manifest:", len(pytest.tests_to_run))
        print("Skipped via manifest:", len(pytest.tests_to_skip))
