#!/usr/bin/env python3
# ==============================================================================
# Copyright (C) 2021 Intel Corporation

# SPDX-License-Identifier: Apache-2.0
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
        linux_distro = subprocess.check_output(
            """awk -F= '$1=="ID" { print $2 ;}' /etc/os-release""", shell=True)
        if "ubuntu" in linux_distro.decode("utf-8"):
            return 'Ubuntu'
        elif "centos" in linux_distro.decode("utf-8"):
            return 'CentOS'
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
        if ('OPENVINO_TF_TEST_MANIFEST' in os.environ):
            return os.environ['OPENVINO_TF_TEST_MANIFEST']
        else:
            # test manifest files are named like this:
            # tests_${PLATFORM}_${OPENVINO_TF_BACKEND}.txt
            return 'tests_' + TestEnv.PLATFORM().lower(
            ) + '_' + TestEnv.BACKEND().lower() + '.txt'

    @staticmethod
    def PLATFORM():
        return platform.system()  # 'Linux', 'Windows', 'Darwin', or 'Java'

    @staticmethod
    def BACKEND():
        if 'OPENVINO_TF_BACKEND' in os.environ:
            return os.environ['OPENVINO_TF_BACKEND']
        else:
            return 'CPU'


def install_openvino_tensorflow(artifacts_dir):
    ovtf_wheel_files = glob.glob(artifacts_dir + "/openvino_tensorflow*.whl")

    if (len(ovtf_wheel_files) != 1):
        print("Multiple Python whl files exist. Please remove old wheels")
        for whl in ovtf_wheel_files:
            print("Existing Wheel: " + whl)
        raise Exception("Error getting the openvino_tensorflow wheel file")

    ng_whl = os.path.join(artifacts_dir, ovtf_wheel_files[0])
    command_executor(["pip", "install", "-U", ng_whl])


def run_ovtf_cpp_gtests(artifacts_dir, log_dir, filters):
    root_pwd = os.getcwd()
    artifacts_dir = os.path.abspath(artifacts_dir)
    log_dir = os.path.abspath(log_dir)

    # Check if we can run C++ tests
    if not os.path.exists(os.path.join(artifacts_dir, "test/gtest_ovtf")):
        print("gtest_ovtf not found. Skipping C++ unit tests...")
        return

    os.environ['GTEST_OUTPUT'] = 'xml:%s/xunit_gtest.xml' % log_dir

    if not os.path.isdir(artifacts_dir):
        raise Exception("Artifacts directory doesn't exist: " + artifacts_dir)

    # First run the C++ gtests
    lib_dir = TestEnv.get_platform_lib_dir()

    os.environ['LD_LIBRARY_PATH'] = os.getenv(
        "LD_LIBRARY_PATH", "") + ':' + os.path.join(artifacts_dir, lib_dir)
    assert os.path.exists(artifacts_dir), "Could not find directory"
    os.chdir(os.path.join(artifacts_dir, "test"))
    if (filters != None):
        gtest_filters = "--gtest_filter=" + filters
        cmd = ['./gtest_ovtf', gtest_filters]
    else:
        cmd = ['./gtest_ovtf']

    command_executor(cmd)
    assert os.path.exists(root_pwd), "Could not find directory"
    os.chdir(root_pwd)


def run_ovtf_pytests_from_artifacts(artifacts_dir):
    root_pwd = os.getcwd()

    artifacts_dir = os.path.abspath(artifacts_dir)
    install_openvino_tensorflow(artifacts_dir)

    test_dir = os.path.join(artifacts_dir, "test")
    test_dir = os.path.join(test_dir, "python")

    if not os.path.isdir(test_dir):
        raise Exception("test directory doesn't exist: " + test_dir)

    # Change the directory to the test_dir
    assert os.path.exists(test_dir), "Could not find directory: {}".format(
        test_dir)
    os.chdir(test_dir)

    # Next run the ngraph-tensorflow python tests
    command_executor(["pip", "install", "-U", "pytest"])
    command_executor(["pip", "install", "-U", "psutil"])

    test_manifest_file = TestEnv.get_test_manifest_filename()
    # export the env-var for pytest to process manifest in conftest.py
    os.environ['OPENVINO_TF_TEST_MANIFEST'] = test_manifest_file

    command_executor([
        "python", "-m", "pytest",
        ('--junitxml=%s/xunit_pytest.xml' % artifacts_dir)
    ])

    assert os.path.exists(root_pwd), "Could not find directory"
    os.chdir(root_pwd)


def run_tensorflow_pytests_from_artifacts(openvino_tf_src_dir, tf_src_dir,
                                          xml_output):
    root_pwd = os.getcwd()
    openvino_tf_src_dir = os.path.abspath(openvino_tf_src_dir)

    # Patch TensorFlow so that the tests run using openvino_tensorflow
    pwd = os.getcwd()
    assert os.path.exists(pwd), "Could not find directory: {}".format(pwd)

    # Go to the location of TesorFlow install directory
    import tensorflow as tf
    tf_dir = tf.sysconfig.get_lib()
    assert os.path.exists(
        tf_dir + '/python/framework'), "Could not find directory: {}".format(
            tf_dir + '/python/framework')
    os.chdir(tf_dir + '/python/framework')
    print("CURRENT DIR: " + os.getcwd())

    f_test_util = "test_util.py"
    if not "openvino_tensorflow" in open(f_test_util).read():
        print("Adding `import openvino_tensorflow` to TensorFlow tests")

        with open(f_test_util, "r") as f:
            fi = f.read()

        import_ovtf_ = fi.replace("super(TensorFlowTestCase, self).__init__(methodName)", \
                "super(TensorFlowTestCase, self).__init__(methodName)\n    import openvino_tensorflow")

        with open(f_test_util, "w") as f:
            f.write(import_ovtf_)
    os.chdir(pwd)

    # Now run the TensorFlow python tests
    test_src_dir = os.path.join(openvino_tf_src_dir, "test/python/tensorflow")
    test_script = os.path.join(test_src_dir, "tf_unittest_runner.py")

    test_manifest_file = TestEnv.get_test_manifest_filename()
    assert os.path.exists(test_src_dir), "Path doesn't exist {}".format(
        test_src_dir)
    if not os.path.isabs(test_manifest_file):
        test_manifest_file = os.path.join(test_src_dir, test_manifest_file)
    assert os.path.exists(test_manifest_file), "Could not find file"

    test_xml_report = './junit_tensorflow_tests.xml'

    import psutil
    num_cores = int(psutil.cpu_count(logical=False))
    print("OMP_NUM_THREADS: %s " % str(num_cores))
    os.environ['OMP_NUM_THREADS'] = str(num_cores)

    openvino_tf_disable_deassign_clusters = os.environ.pop(
        'OPENVINO_TF_DISABLE_DEASSIGN_CLUSTERS', None)
    os.environ['OPENVINO_TF_DISABLE_DEASSIGN_CLUSTERS'] = '1'

    cmd = [
        "python", test_script, "--tensorflow_path", tf_src_dir,
        "--run_tests_from_file", test_manifest_file
    ]

    if xml_output:
        cmd.extend(["--xml_report", test_xml_report])
    command_executor(cmd, verbose=True)

    os.environ.pop('OPENVINO_TF_DISABLE_DEASSIGN_CLUSTERS', None)

    if openvino_tf_disable_deassign_clusters is not None:
        os.environ['OPENVINO_TF_DISABLE_DEASSIGN_CLUSTERS'] = \
            openvino_tf_disable_deassign_clusters

    assert os.path.exists(root_pwd), "Could not find the path"
    os.chdir(root_pwd)


def run_resnet50_from_artifacts(openvino_tf_src_dir, artifact_dir, batch_size,
                                iterations):
    root_pwd = os.getcwd()
    artifact_dir = os.path.abspath(artifact_dir)
    openvino_tf_src_dir = os.path.abspath(openvino_tf_src_dir)
    install_openvino_tensorflow(artifact_dir)

    # Now clone the repo and proceed
    call(['git', 'clone', 'https://github.com/tensorflow/benchmarks.git'])
    assert os.path.exists('benchmarks'), "Could not find directory: {}".format(
        'benchmarks')
    os.chdir('benchmarks')
    call(['git', 'checkout', 'aef6daa90a467a1fc7ce8395cd0067e5fda1ecff'])

    # Check to see if we need to patch the repo for Grappler
    # benchmark_cnn.patch will only work for the CPU backend
    patch_file = os.path.abspath(
        os.path.join(openvino_tf_src_dir, "test/grappler/benchmark_cnn.patch"))
    import openvino_tensorflow
    if openvino_tensorflow.is_grappler_enabled():
        print("Patching repo using: %s" % patch_file)
        apply_patch(patch_file)
    assert os.path.exists(
        'scripts/tf_cnn_benchmarks/'), "Could not find directory: {}".format(
            'scripts/tf_cnn_benchmarks/')
    os.chdir('scripts/tf_cnn_benchmarks/')

    # junit_script = os.path.abspath('%s/test/ci/junit-wrap.sh' % root_pwd)

    # Update the script by adding `import openvino_tensorflow`
    with open('convnet_builder.py', 'a') as outfile:
        call(['echo', 'import openvino_tensorflow'], stdout=outfile)

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

    assert os.path.exists(root_pwd), "Could not find the path"
    os.chdir(root_pwd)


def run_resnet50_infer_from_artifacts(artifact_dir, batch_size, iterations):
    root_pwd = os.getcwd()
    artifact_dir = os.path.abspath(artifact_dir)
    if not os.path.exists(artifact_dir):
        raise Exception("Can't find artifact dir: " + artifact_dir)
    if (len(glob.glob(artifact_dir + "/openvino_tensorflow-*.whl")) == 0):
        install_openvino_tensorflow(artifact_dir)

    # Check/download pretrained model
    pretrained_models_dir = os.path.abspath(
        os.path.join(root_pwd, '../pretrained_models'))
    if not os.path.exists(pretrained_models_dir):
        os.mkdir(pretrained_models_dir, 0o755)
    assert os.path.exists(
        pretrained_models_dir), "Could not find the path: {}".format(
            pretrained_models_dir)
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

    assert os.path.exists(root_pwd), "Could not find the path"
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
    assert os.path.exists(root_pwd), "Could not find the path"
    os.chdir(root_pwd)
