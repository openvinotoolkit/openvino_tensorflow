#!/usr/bin/env python3
# ==============================================================================
#  Copyright 2019-2020 Intel Corporation
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
from subprocess import check_output, call, Popen
import sys
import shutil
import glob
import platform
import subprocess
from distutils.sysconfig import get_python_lib

from tools.build_utils import load_venv, command_executor, apply_patch


# Abstract all Platform/OS specific information within this class
class TestEnv:

    @staticmethod
    def get_platform_type():
        return platform.system()  # 'Linux', 'Windows', 'Darwin', or 'Java'

    @staticmethod
    def get_linux_type():
        if platform.linux_distribution():
            return platform.linux_distribution()[0]  # Ubuntu or CentOS
        else:
            return ''

    @staticmethod
    def get_platform_lib_dir():
        lib_dir = 'lib'
        if 'CentOS' in TestEnv.get_linux_type():
            lib_dir = 'lib64'
        return lib_dir

    @staticmethod
    def is_osx():
        return platform.system() == 'Darwin'

    @staticmethod
    def is_mac():
        return TestEnv.is_osx()

    @staticmethod
    def is_linux():
        return platform.system() == 'Linux'

    @staticmethod
    def get_test_manifest_filename():
        if ('NGRAPH_TF_TEST_MANIFEST' in os.environ):
            return os.environ['NGRAPH_TF_TEST_MANIFEST']
        else:
            # test manifest files are named like this:
            # tests_${PLATFORM}_${NGRAPH_TF_BACKEND}.txt
            return 'tests_' + TestEnv.PLATFORM().lower(
            ) + '_' + TestEnv.BACKEND().lower() + '.txt'

    @staticmethod
    def PLATFORM():
        return platform.system()  # 'Linux', 'Windows', 'Darwin', or 'Java'

    @staticmethod
    def BACKEND():
        if 'NGRAPH_TF_BACKEND' in os.environ:
            return os.environ['NGRAPH_TF_BACKEND']
        else:
            return 'CPU'


def install_ngraph_bridge(artifacts_dir):
    ngtf_wheel_files = glob.glob(artifacts_dir +
                                 "/ngraph_tensorflow_bridge-*.whl")

    if (len(ngtf_wheel_files) != 1):
        print("Multiple Python whl files exist. Please remove old wheels")
        for whl in ngtf_wheel_files:
            print("Existing Wheel: " + whl)
        raise Exception("Error getting the ngraph-tf wheel file")

    ng_whl = os.path.join(artifacts_dir, ngtf_wheel_files[0])
    command_executor(["pip", "install", "-U", ng_whl])


def run_ngtf_cpp_gtests(artifacts_dir, log_dir, filters):
    root_pwd = os.getcwd()
    artifacts_dir = os.path.abspath(artifacts_dir)
    log_dir = os.path.abspath(log_dir)

    # Check if we can run C++ tests
    if not os.path.exists(os.path.join(artifacts_dir, "test/gtest_ngtf")):
        print("gtest_ngtf not found. Skipping C++ unit tests...")
        return

    os.environ['GTEST_OUTPUT'] = 'xml:%s/xunit_gtest.xml' % log_dir

    if not os.path.isdir(artifacts_dir):
        raise Exception("Artifacts directory doesn't exist: " + artifacts_dir)

    # First run the C++ gtests
    lib_dir = TestEnv.get_platform_lib_dir()

    os.environ['LD_LIBRARY_PATH'] = os.getenv(
        "LD_LIBRARY_PATH", "") + ':' + os.path.join(artifacts_dir, lib_dir)
    os.chdir(os.path.join(artifacts_dir, "test"))
    if (filters != None):
        gtest_filters = "--gtest_filter=" + filters
        cmd = ['./gtest_ngtf', gtest_filters]
    else:
        cmd = ['./gtest_ngtf']

    command_executor(cmd)
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

    test_manifest_file = TestEnv.get_test_manifest_filename()
    # export the env-var for pytest to process manifest in conftest.py
    os.environ['NGRAPH_TF_TEST_MANIFEST'] = test_manifest_file

    command_executor([
        "python", "-m", "pytest",
        ('--junitxml=%s/xunit_pytest.xml' % artifacts_dir)
    ])

    os.chdir(root_pwd)


def run_tensorflow_pytests_from_artifacts(ngraph_tf_src_dir, tf_src_dir,
                                          xml_output):
    root_pwd = os.getcwd()
    ngraph_tf_src_dir = os.path.abspath(ngraph_tf_src_dir)

    # Check to see if we need to apply the patch for Grappler
    import ngraph_bridge
    patch_file_name = "test/python/tensorflow/tf_unittest_ngraph" + (
        "_with_grappler"
        if ngraph_bridge.is_grappler_enabled() else "") + ".patch"
    patch_file = os.path.abspath(
        os.path.join(ngraph_tf_src_dir, patch_file_name))

    # Next patch the TensorFlow so that the tests run using ngraph_bridge
    pwd = os.getcwd()

    # Go to the location of TesorFlow install directory
    import tensorflow as tf
    tf_dir = tf.sysconfig.get_lib()
    os.chdir(tf_dir + '/python/framework')
    print("CURRENT DIR: " + os.getcwd())

    print("Patching TensorFlow using: %s" % patch_file)
    cmd = subprocess.Popen(
        'patch -N -i ' + patch_file, shell=True, stdout=subprocess.PIPE)
    printed_lines = cmd.communicate()
    # Check if the patch is being applied for the first time, in which case
    # cmd.returncode will be 0 or if the patch has already been applied, in
    # which case the string will be found, in all other cases the assertion
    # will fail
    assert cmd.returncode == 0 or 'patch detected!  Skipping patch' in str(
        printed_lines[0]), "Error applying the patch."
    os.chdir(pwd)

    # Now run the TensorFlow python tests
    test_src_dir = os.path.join(ngraph_tf_src_dir, "test/python/tensorflow")
    test_script = os.path.join(test_src_dir, "tf_unittest_runner.py")

    test_manifest_file = TestEnv.get_test_manifest_filename()
    if not os.path.isabs(test_manifest_file):
        test_manifest_file = os.path.join(test_src_dir, test_manifest_file)

    test_xml_report = './junit_tensorflow_tests.xml'

    import psutil
    num_cores = int(psutil.cpu_count(logical=False))
    print("OMP_NUM_THREADS: %s " % str(num_cores))
    os.environ['OMP_NUM_THREADS'] = str(num_cores)
    os.environ['NGRAPH_TF_DISABLE_DEASSIGN_CLUSTERS'] = '1'

    cmd = [
        "python", test_script, "--tensorflow_path", tf_src_dir,
        "--run_tests_from_file", test_manifest_file
    ]
    if xml_output:
        cmd.extend(["--xml_report", test_xml_report])
    command_executor(cmd, verbose=True)

    os.chdir(root_pwd)


def run_resnet50_from_artifacts(ngraph_tf_src_dir, artifact_dir, batch_size,
                                iterations):
    root_pwd = os.getcwd()
    artifact_dir = os.path.abspath(artifact_dir)
    ngraph_tf_src_dir = os.path.abspath(ngraph_tf_src_dir)
    install_ngraph_bridge(artifact_dir)

    # Now clone the repo and proceed
    call(['git', 'clone', 'https://github.com/tensorflow/benchmarks.git'])
    os.chdir('benchmarks')
    call(['git', 'checkout', 'aef6daa90a467a1fc7ce8395cd0067e5fda1ecff'])

    # Check to see if we need to patch the repo for Grappler
    # benchmark_cnn.patch will only work for the CPU backend
    patch_file = os.path.abspath(
        os.path.join(ngraph_tf_src_dir, "test/grappler/benchmark_cnn.patch"))
    import ngraph_bridge
    if ngraph_bridge.is_grappler_enabled():
        print("Patching repo using: %s" % patch_file)
        apply_patch(patch_file)

    os.chdir('scripts/tf_cnn_benchmarks/')

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

    cmd = [
        'python', 'tf_cnn_benchmarks.py', '--data_format', 'NHWC',
        '--num_inter_threads', '1', '--train_dir=' + model_save_dir,
        '--num_batches',
        str(iterations), '--model=resnet50', '--batch_size=' + str(batch_size),
        '--eval_dir=' + eval_eventlog_dir
    ]
    command_executor(cmd, verbose=True)
    cmd = [
        'python', 'tf_cnn_benchmarks.py', '--data_format', 'NHWC',
        '--num_inter_threads', '1', '--train_dir=' + model_save_dir,
        '--model=resnet50', '--batch_size=' + str(batch_size), '--num_batches',
        str(iterations), '--eval', '--eval_dir=' + eval_eventlog_dir
    ]
    # Commenting the eval since it currently fails with TF2.0
    command_executor(cmd, verbose=True)

    os.chdir(root_pwd)


def run_resnet50_infer_from_artifacts(artifact_dir, batch_size, iterations):
    root_pwd = os.getcwd()
    artifact_dir = os.path.abspath(artifact_dir)
    if not os.path.exists(artifact_dir):
        raise Exception("Can't find artifact dir: " + artifact_dir)
    if (len(glob.glob(artifact_dir + "/ngraph_tensorflow_bridge-*.whl")) == 0):
        install_ngraph_bridge(artifact_dir)

    # Check/download pretrained model
    pretrained_models_dir = os.path.abspath(
        os.path.join(root_pwd, '../pretrained_models'))
    if not os.path.exists(pretrained_models_dir):
        os.mkdir(pretrained_models_dir, 0o755)
    os.chdir(pretrained_models_dir)
    pretrained_model = os.path.join(pretrained_models_dir, 'resnet50_v1.pb')
    if not os.path.exists(pretrained_model):
        # wget https://zenodo.org/record/2535873/files/resnet50_v1.pb
        command_executor(
            ['wget', 'https://zenodo.org/record/2535873/files/resnet50_v1.pb'],
            verbose=True)
        if not os.path.exists(pretrained_model):
            raise Exception("Can't download pretrained model: " +
                            pretrained_model)
    else:
        print("Using existing pretrained model file: " + pretrained_model)

    # Setup the env flags
    import psutil
    num_cores = int(psutil.cpu_count(logical=False))
    print("OMP_NUM_THREADS: %s " % str(num_cores))
    os.environ['OMP_NUM_THREADS'] = str(num_cores)
    os.environ["KMP_AFFINITY"] = 'granularity=fine,compact,1,0'

    os.chdir(root_pwd)
    cmd = [
        'python',
        artifact_dir + '/test/python/test_rn50_infer.py',
        '--input-graph',
        pretrained_model,
        '--batch-size',
        str(batch_size),
        '--num-images',
        str(batch_size * iterations),
    ]
    command_executor(cmd, verbose=True)
    os.chdir(root_pwd)