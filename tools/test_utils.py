#!/usr/bin/env python3
# ==============================================================================
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
# ==============================================================================

import argparse
import errno
import os
from subprocess import check_output, call
import sys
import shutil
import glob
import platform
from distutils.sysconfig import get_python_lib

from tools.build_utils import load_venv, command_executor


def get_os_type():
    if platform.system() == 'Darwin':
        return 'Darwin'

    if platform.linux_distribution():
        return platform.linux_distribution()[0]


def install_ngraph_bridge(artifacts_dir):
    # Determine the ngraph whl
    ngtf_wheel_files = glob.glob(artifacts_dir +
                                 "/ngraph_tensorflow_bridge-*.whl")
    if (len(ngtf_wheel_files) != 1):
        print("Multiple Python whl files exist. Please remove old wheels")
        for whl in ngtf_wheel_files:
            print("Existing Wheel: " + whl)
        raise Exception("Error getting the ngraph-tf wheel file")

    # First ensure that we have nGraph installed
    ng_whl = os.path.join(artifacts_dir, ngtf_wheel_files[0])
    command_executor(["pip", "install", "-U", ng_whl])


#@depricated
def run_ngtf_gtests(build_dir, filters):
    root_pwd = os.getcwd()
    build_dir = os.path.abspath(build_dir)

    os.environ['GTEST_OUTPUT'] = 'xml:%s/xunit_gtest.xml' % build_dir

    if not os.path.isdir(build_dir):
        raise Exception("build directory doesn't exist: " + build_dir)

    # First run the C++ gtests
    os.chdir(os.path.join(build_dir, "test"))
    if (filters != None):
        gtest_filters = "--gtest_filter=" + filters
        cmd = ['./gtest_ngtf', gtest_filters]
    else:
        cmd = ['./gtest_ngtf']

    command_executor(cmd, verbose=True)

    os.chdir(root_pwd)


def run_ngtf_cpp_gtests(artifacts_dir, log_dir, filters):
    root_pwd = os.getcwd()
    artifacts_dir = os.path.abspath(artifacts_dir)
    log_dir = os.path.abspath(log_dir)

    os.environ['GTEST_OUTPUT'] = 'xml:%s/xunit_gtest.xml' % log_dir

    if not os.path.isdir(artifacts_dir):
        raise Exception("Artifacts directory doesn't exist: " + artifacts_dir)

    # First run the C++ gtests
    lib_dir = 'lib'
    if 'CentOS' in get_os_type():
        lib_dir = 'lib64'

    os.environ['LD_LIBRARY_PATH'] = os.path.join(artifacts_dir, lib_dir)
    os.chdir(os.path.join(artifacts_dir, "test"))
    if (filters != None):
        gtest_filters = "--gtest_filter=" + filters
        cmd = ['./gtest_ngtf', gtest_filters]
    else:
        cmd = ['./gtest_ngtf']

    command_executor(cmd, verbose=True)

    os.chdir(root_pwd)


def run_ngtf_pytests(venv_dir, build_dir):
    root_pwd = os.getcwd()

    build_dir = os.path.abspath(build_dir)
    venv_dir = os.path.abspath(venv_dir)

    test_dir = os.path.join(build_dir, "test")
    test_dir = os.path.join(test_dir, "python")

    if not os.path.isdir(test_dir):
        raise Exception("test directory doesn't exist: " + test_dir)

    # Change the directory to the test_dir
    os.chdir(test_dir)

    # Load venv
    load_venv(venv_dir)

    # Next run the ngraph-tensorflow python tests
    command_executor(["pip", "install", "-U", "pytest"])
    command_executor(["pip", "install", "-U", "psutil"])
    command_executor([
        "python", "-m", "pytest", ('--junitxml=%s/xunit_pytest.xml' % build_dir)
    ],
                     verbose=True)

    os.chdir(root_pwd)


def run_ngtf_pytests_from_artifacts(artifacts_dir):
    root_pwd = os.getcwd()

    artifacts_dir = os.path.abspath(artifacts_dir)
    install_ngraph_bridge(artifacts_dir)

    test_dir = os.path.join(artifacts_dir, "test")
    test_dir = os.path.join(test_dir, "python")

    if not os.path.isdir(test_dir):
        raise Exception("test directory doesn't exist: " + test_dir)

    # Change the directory to the test_dir
    os.chdir(test_dir)

    # Next run the ngraph-tensorflow python tests
    command_executor(["pip", "install", "-U", "pytest"])
    command_executor(["pip", "install", "-U", "psutil"])
    command_executor([
        "python", "-m", "pytest",
        ('--junitxml=%s/xunit_pytest.xml' % artifacts_dir)
    ])

    os.chdir(root_pwd)


def run_tensorflow_pytests(venv_dir, build_dir, ngraph_tf_src_dir, tf_src_dir):
    root_pwd = os.getcwd()

    build_dir = os.path.abspath(build_dir)
    venv_dir = os.path.abspath(venv_dir)
    ngraph_tf_src_dir = os.path.abspath(ngraph_tf_src_dir)

    patch_file = os.path.abspath(
        os.path.join(ngraph_tf_src_dir,
                     "test/python/tensorflow/tf_unittest_ngraph.patch"))

    # Load the virtual env
    venv_dir_absolute = load_venv(venv_dir)

    # Next patch the TensorFlow so that the tests run using ngraph_bridge
    pwd = os.getcwd()

    # Go to the site-packages
    os.chdir(glob.glob(venv_dir_absolute + "/lib/py*/site-packages")[0])
    print("CURRENT DIR: " + os.getcwd())

    print("Patching TensorFlow using: %s" % patch_file)
    result = call(["patch", "-p1", "-N", "-i", patch_file])
    print("Patch result: %d" % result)
    os.chdir(pwd)

    # Now run the TensorFlow python tests
    test_src_dir = os.path.join(ngraph_tf_src_dir, "test/python/tensorflow")
    test_script = os.path.join(test_src_dir, "tf_unittest_runner.py")
    test_manifest_file = os.path.join(test_src_dir, "python_tests_list.txt")
    test_xml_report = '%s/junit_tensorflow_tests.xml' % build_dir

    import psutil
    num_cores = int(psutil.cpu_count(logical=False))
    print("OMP_NUM_THREADS: %s " % str(num_cores))
    os.environ['OMP_NUM_THREADS'] = str(num_cores)
    os.environ['NGRAPH_TF_DISABLE_DEASSIGN_CLUSTERS'] = '1'

    command_executor([
        "python", test_script, "--tensorflow_path", tf_src_dir,
        "--run_tests_from_file", test_manifest_file, "--xml_report",
        test_xml_report
    ])

    os.chdir(root_pwd)


def run_tensorflow_pytests_from_artifacts(ngraph_tf_src_dir, tf_src_dir,
                                          xml_output):
    root_pwd = os.getcwd()

    ngraph_tf_src_dir = os.path.abspath(ngraph_tf_src_dir)

    patch_file = os.path.abspath(
        os.path.join(ngraph_tf_src_dir,
                     "test/python/tensorflow/tf_unittest_ngraph.patch"))

    # Next patch the TensorFlow so that the tests run using ngraph_bridge
    pwd = os.getcwd()

    # Go to the location of TesorFlow install directory
    import tensorflow as tf
    tf_dir = tf.sysconfig.get_lib()
    os.chdir(os.path.join(tf_dir, '../'))
    print("CURRENT DIR: " + os.getcwd())

    print("Patching TensorFlow using: %s" % patch_file)
    result = call(["patch", "-p1", "-N", "-i", patch_file])
    print("Patch result: %d" % result)
    os.chdir(pwd)

    # Now run the TensorFlow python tests
    test_src_dir = os.path.join(ngraph_tf_src_dir, "test/python/tensorflow")
    test_script = os.path.join(test_src_dir, "tf_unittest_runner.py")
    test_manifest_file = os.path.join(test_src_dir, "python_tests_list.txt")
    test_xml_report = './junit_tensorflow_tests.xml'

    import psutil
    num_cores = int(psutil.cpu_count(logical=False))
    print("OMP_NUM_THREADS: %s " % str(num_cores))
    os.environ['OMP_NUM_THREADS'] = str(num_cores)
    os.environ['NGRAPH_TF_DISABLE_DEASSIGN_CLUSTERS'] = '1'

    cmd = [
        "python",
        test_script,
        "--tensorflow_path",
        tf_src_dir,
        "--run_tests_from_file",
        test_manifest_file,
    ]
    if xml_output:
        cmd.extend(["--xml_report", test_xml_report])
    command_executor(cmd)

    os.chdir(root_pwd)


def run_resnet50(build_dir):

    root_pwd = os.getcwd()
    build_dir = os.path.abspath(build_dir)
    os.chdir(build_dir)

    call(['git', 'clone', 'https://github.com/tensorflow/benchmarks.git'])
    os.chdir('benchmarks/scripts/tf_cnn_benchmarks/')

    call(['git', 'checkout', '4c7b09ad87bbfc4b1f89650bcee40b3fc5e7dfed'])

    junit_script = os.path.abspath('%s/test/ci/junit-wrap.sh' % root_pwd)

    # Update the script by adding `import ngraph_bridge`
    with open('convnet_builder.py', 'a') as outfile:
        call(['echo', 'import ngraph_bridge'], stdout=outfile)

    # Setup the env flags
    import psutil
    num_cores = int(psutil.cpu_count(logical=False))
    print("OMP_NUM_THREADS: %s " % str(num_cores))

    os.environ['OMP_NUM_THREADS'] = str(num_cores)
    os.environ["KMP_AFFINITY"] = 'granularity=fine,compact,1,0'

    # Delete the temporary model save directory
    model_save_dir = os.getcwd() + '/modelsavepath'
    if os.path.exists(model_save_dir) and os.path.isdir(model_save_dir):
        shutil.rmtree(model_save_dir)

    os.environ['JUNIT_WRAP_FILE'] = "%s/junit_training_test.xml" % build_dir
    os.environ['JUNIT_WRAP_SUITE'] = 'models'
    os.environ['JUNIT_WRAP_TEST'] = 'resnet50-training'

    # Run training job
    cmd = [
        junit_script, 'python', 'tf_cnn_benchmarks.py', '--data_format', 'NCHW',
        '--num_inter_threads', '1', '--train_dir=' + model_save_dir,
        '--num_batches', '10', '--model=resnet50', '--batch_size=128'
    ]
    command_executor(cmd)

    os.environ['JUNIT_WRAP_FILE'] = "%s/junit_inference_test.xml" % build_dir
    os.environ['JUNIT_WRAP_SUITE'] = 'models'
    os.environ['JUNIT_WRAP_TEST'] = 'resnet50-inference'

    # Run inference job
    cmd = [
        junit_script, 'python', 'tf_cnn_benchmarks.py', '--data_format', 'NCHW',
        '--num_inter_threads', '1', '--train_dir=' + model_save_dir,
        '--model=resnet50', '--batch_size=128', '--num_batches', '10', '--eval'
    ]
    command_executor(cmd)
    os.chdir(root_pwd)


def run_resnet50_from_artifacts(artifact_dir, batch_size, iterations):

    root_pwd = os.getcwd()
    artifact_dir = os.path.abspath(artifact_dir)

    install_ngraph_bridge(artifact_dir)

    # Now clone the repo and proceed
    call(['git', 'clone', 'https://github.com/tensorflow/benchmarks.git'])
    os.chdir('benchmarks/scripts/tf_cnn_benchmarks/')

    call(['git', 'checkout', '4c7b09ad87bbfc4b1f89650bcee40b3fc5e7dfed'])

    # junit_script = os.path.abspath('%s/test/ci/junit-wrap.sh' % root_pwd)

    # Update the script by adding `import ngraph_bridge`
    with open('convnet_builder.py', 'a') as outfile:
        call(['echo', 'import ngraph_bridge'], stdout=outfile)

    # Setup the env flags
    import psutil
    num_cores = int(psutil.cpu_count(logical=False))
    print("OMP_NUM_THREADS: %s " % str(num_cores))

    os.environ['OMP_NUM_THREADS'] = str(num_cores)
    os.environ["KMP_AFFINITY"] = 'granularity=fine,compact,1,0'

    # Delete the temporary model save directory
    model_save_dir = os.getcwd() + '/modelsavepath'
    if os.path.exists(model_save_dir) and os.path.isdir(model_save_dir):
        shutil.rmtree(model_save_dir)

    eval_eventlog_dir = os.getcwd() + '/eval_eventlog_dir'
    if os.path.exists(eval_eventlog_dir) and os.path.isdir(eval_eventlog_dir):
        shutil.rmtree(eval_eventlog_dir)

    # os.environ['JUNIT_WRAP_FILE'] = "%s/junit_training_test.xml" % build_dir
    # os.environ['JUNIT_WRAP_SUITE'] = 'models'
    # os.environ['JUNIT_WRAP_TEST'] = 'resnet50-training'

    # Run training job
    # cmd = [
    #     junit_script, 'python', 'tf_cnn_benchmarks.py', '--data_format',
    #     'NCHW', '--num_inter_threads', '1', '--train_dir=' + model_save_dir,
    #     '--num_batches', '10', '--model=resnet50', '--batch_size=128'
    # ]

    cmd = [
        'python', 'tf_cnn_benchmarks.py', '--data_format', 'NCHW',
        '--num_inter_threads', '1', '--train_dir=' + model_save_dir,
        '--num_batches',
        str(iterations), '--model=resnet50', '--batch_size=' + str(batch_size),
        '--eval_dir=' + eval_eventlog_dir
    ]
    command_executor(cmd)

    # os.environ['JUNIT_WRAP_FILE'] = "%s/junit_inference_test.xml" % build_dir
    # os.environ['JUNIT_WRAP_SUITE'] = 'models'
    # os.environ['JUNIT_WRAP_TEST'] = 'resnet50-inference'

    # Run inference job
    # cmd = [
    #     junit_script, 'python', 'tf_cnn_benchmarks.py', '--data_format',
    #     'NCHW', '--num_inter_threads', '1', '--train_dir=' + model_save_dir,
    #     '--model=resnet50', '--batch_size=128', '--num_batches', '10', '--eval'
    # ]
    cmd = [
        'python', 'tf_cnn_benchmarks.py', '--data_format', 'NCHW',
        '--num_inter_threads', '1', '--train_dir=' + model_save_dir,
        '--model=resnet50', '--batch_size=' + str(batch_size), '--num_batches',
        str(iterations), '--eval', '--eval_dir=' + eval_eventlog_dir
    ]
    command_executor(cmd)

    os.chdir(root_pwd)


def run_cpp_example_test(build_dir):

    root_pwd = os.getcwd()
    build_dir = os.path.abspath(build_dir)
    os.chdir(build_dir)

    # Create the example workspace directory and chdir there
    path = 'cpp_example'
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
    os.chdir(path)

    # Copy the files
    files = [
        '../../examples/tf_cpp_examples/hello_tf.cpp',
        '../../examples/tf_cpp_examples/Makefile'
    ]
    command_executor(['cp', files[0], './'])
    command_executor(['cp', files[1], './'])

    # Now execute Make
    command_executor(['make'])

    # Now run the hello_tf example
    # First setup the LD_LIB_PATH
    if (platform.system() == 'Darwin'):
        ld_path_name = 'DYLD_LIBRARY_PATH'
    else:
        ld_path_name = 'LD_LIBRARY_PATH'

    os.environ[ld_path_name] = '../artifacts/lib:../artifacts/tensorflow'
    command_executor('./hello_tf')

    # Return to the original directory
    os.chdir(root_pwd)


def run_bazel_build_test(venv_dir, build_dir):
    # Load the virtual env
    venv_dir_absolute = load_venv(venv_dir)

    # Next patch the TensorFlow so that the tests run using ngraph_bridge
    root_pwd = os.getcwd()

    # Now run the configure
    command_executor(['bash', 'configure_bazel.sh'])

    # Build the bridge
    command_executor(['bazel', 'build', 'libngraph_bridge.so'])

    # Build the backend
    command_executor(['bazel', 'build', '@ngraph//:libinterpreter_backend.so'])

    # Return to the original directory
    os.chdir(root_pwd)


def run_bazel_build():
    # Next patch the TensorFlow so that the tests run using ngraph_bridge
    root_pwd = os.getcwd()

    # Now run the configure
    command_executor(['bash', 'configure_bazel.sh'])

    # Build the bridge
    command_executor(['bazel', 'build', 'libngraph_bridge.so'])

    # Build the backend
    command_executor(['bazel', 'build', '@ngraph//:libinterpreter_backend.so'])

    # Return to the original directory
    os.chdir(root_pwd)
