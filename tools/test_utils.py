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

    ng_whl = os.path.join(artifacts_dir, ngtf_wheel_files[0])
    command_executor(["pip", "install", "-U", ng_whl])


#@depricated
def run_ngtf_gtests(build_dir, filters):
    root_pwd = os.getcwd()
    build_dir = os.path.abspath(build_dir)

    # Check if we can run C++ tests
    if not os.path.exists(os.path.join(build_dir, "test/gtest_ngtf")):
        print("gtest_ngtf not found. Skipping C++ unit tests...")
        return

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

    command_executor(cmd)

    os.chdir(root_pwd)


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

    command_executor(cmd)
    os.chdir(root_pwd)


def run_ngtf_pytests(venv_dir, build_dir):
    root_pwd = os.getcwd()

    build_dir = os.path.abspath(build_dir)
    venv_dir = os.path.abspath(venv_dir)
    mnist_dir = os.path.abspath(build_dir + '/examples/mnist/')
    axpy_dir = os.path.abspath(build_dir + '/examples/')
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
    command_executor(["pip", "install", "-U", "keras==2.3.1"])
    command_executor(["pip", "install", "-U", "psutil"])

    cmd = 'python -m pytest ' + ('--junitxml=%s/xunit_pytest.xml' % build_dir)
    env = os.environ.copy()
    new_paths = venv_dir + '/bin/python3:' + os.path.abspath(
        build_dir) + ":" + os.path.abspath(axpy_dir) + ":" + os.path.abspath(
            mnist_dir)
    if 'PYTHONPATH' in env:
        env["PYTHONPATH"] = new_paths + ":" + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = new_paths
    ps = Popen(cmd, shell=True, env=env)
    so, se = ps.communicate()
    errcode = ps.returncode
    assert errcode == 0, "Error in running command: " + cmd
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

    # Go to the site-packages/tensorflow_core_python/framework
    os.chdir(
        glob.glob(venv_dir_absolute +
                  "/lib/py*/site-packages/tensorflow_core/python/framework")[0])
    print("CURRENT DIR: " + os.getcwd())

    print("Patching TensorFlow using: %s" % patch_file)
    apply_patch(patch_file, level=0)
    os.chdir(pwd)

    # Now run the TensorFlow python tests
    test_src_dir = os.path.join(ngraph_tf_src_dir, "test/python/tensorflow")
    test_script = os.path.join(test_src_dir, "tf_unittest_runner.py")
    if get_os_type() == 'Darwin':
        test_manifest_file = os.path.join(test_src_dir,
                                          "python_tests_list_mac.txt")
    else:
        test_manifest_file = os.path.join(test_src_dir, "python_tests_list.txt")
    test_xml_report = '%s/junit_tensorflow_tests.xml' % build_dir

    import psutil
    num_cores = int(psutil.cpu_count(logical=False))
    print("OMP_NUM_THREADS: %s " % str(num_cores))
    os.environ['OMP_NUM_THREADS'] = str(num_cores)
    os.environ['NGRAPH_TF_DISABLE_DEASSIGN_CLUSTERS'] = '1'

    # command_executor([
    #     "python", test_script, "--tensorflow_path", tf_src_dir,
    #     "--run_tests_from_file", test_manifest_file, "--xml_report",
    #     test_xml_report
    # ], verbose=True)

    command_executor([
        "python", test_script, "--tensorflow_path", tf_src_dir,
        "--run_tests_from_file", test_manifest_file
    ],
                     verbose=True)

    os.chdir(root_pwd)


def run_tensorflow_pytests_from_artifacts(backend, ngraph_tf_src_dir,
                                          tf_src_dir, xml_output):
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
    if get_os_type() == 'Darwin':
        test_manifest_file = os.path.join(test_src_dir,
                                          "python_tests_list_mac.txt")
    else:
        test_manifest_file = os.path.join(test_src_dir, "python_tests_list.txt")
    if backend is not None:
        if 'INTERPRETER' in backend:
            test_manifest_file = os.path.join(test_src_dir,
                                              "python_tests_list_int.txt")

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


def run_resnet50(build_dir):

    root_pwd = os.getcwd()
    build_dir = os.path.abspath(build_dir)
    ngraph_tf_src_dir = os.path.abspath(build_dir + '/../')
    os.chdir(build_dir)

    call(['git', 'clone', 'https://github.com/tensorflow/benchmarks.git'])
    os.chdir('benchmarks')
    call(['git', 'checkout', '4c7b09ad87bbfc4b1f89650bcee40b3fc5e7dfed'])

    junit_script = os.path.abspath('%s/test/ci/junit-wrap.sh' % root_pwd)

    # Check to see if we need to patch the repo for Grappler
    # benchmark_cnn.patch will only work for the CPU backend
    patch_file = os.path.abspath(
        os.path.join(ngraph_tf_src_dir, "test/grappler/benchmark_cnn.patch"))
    import ngraph_bridge
    if ngraph_bridge.is_grappler_enabled():
        print("Patching repo using: %s" % patch_file)
        apply_patch(patch_file)

    os.chdir('scripts/tf_cnn_benchmarks/')
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
        junit_script,
        'python',
        'tf_cnn_benchmarks.py',
        '--data_format',
        'NHWC',
        '--num_inter_threads',
        '1',
        '--train_dir=' + model_save_dir,
        '--num_batches',
        '10',
        '--model=resnet50',
        '--batch_size=128',
    ]
    command_executor(cmd, verbose=True)

    os.environ['JUNIT_WRAP_FILE'] = "%s/junit_inference_test.xml" % build_dir
    os.environ['JUNIT_WRAP_SUITE'] = 'models'
    os.environ['JUNIT_WRAP_TEST'] = 'resnet50-inference'

    # Run inference job
    cmd = [
        junit_script, 'python', 'tf_cnn_benchmarks.py', '--data_format', 'NHWC',
        '--num_inter_threads', '1', '--train_dir=' + model_save_dir,
        '--model=resnet50', '--batch_size=128', '--num_batches', '10', '--eval'
    ]
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

    # os.environ['JUNIT_WRAP_FILE'] = "%s/junit_training_test.xml" % build_dir
    # os.environ['JUNIT_WRAP_SUITE'] = 'models'
    # os.environ['JUNIT_WRAP_TEST'] = 'resnet50-training'

    # Run training job
    # cmd = [
    #     junit_script, 'python', 'tf_cnn_benchmarks.py', '--data_format',
    #     'NHWC', '--num_inter_threads', '1', '--train_dir=' + model_save_dir,
    #     '--num_batches', '10', '--model=resnet50', '--batch_size=128'
    # ]

    cmd = [
        'python', 'tf_cnn_benchmarks.py', '--data_format', 'NHWC',
        '--num_inter_threads', '1', '--train_dir=' + model_save_dir,
        '--num_batches',
        str(iterations), '--model=resnet50', '--batch_size=' + str(batch_size),
        '--eval_dir=' + eval_eventlog_dir
    ]
    command_executor(cmd, verbose=True)

    # os.environ['JUNIT_WRAP_FILE'] = "%s/junit_inference_test.xml" % build_dir
    # os.environ['JUNIT_WRAP_SUITE'] = 'models'
    # os.environ['JUNIT_WRAP_TEST'] = 'resnet50-inference'

    # Run inference job
    # cmd = [
    #     junit_script, 'python', 'tf_cnn_benchmarks.py', '--data_format',
    #     'NHWC', '--num_inter_threads', '1', '--train_dir=' + model_save_dir,
    #     '--model=resnet50', '--batch_size=128', '--num_batches', '10', '--eval'
    # ]
    cmd = [
        'python', 'tf_cnn_benchmarks.py', '--data_format', 'NHWC',
        '--num_inter_threads', '1', '--train_dir=' + model_save_dir,
        '--model=resnet50', '--batch_size=' + str(batch_size), '--num_batches',
        str(iterations), '--eval', '--eval_dir=' + eval_eventlog_dir
    ]
    # Commenting the eval since it currently fails with TF2.0
    command_executor(cmd, verbose=True)

    os.chdir(root_pwd)


def run_resnet50_forward_pass(build_dir):

    root_pwd = os.getcwd()
    build_dir = os.path.abspath(build_dir)
    ngraph_tf_src_dir = os.path.abspath(build_dir + '/../')
    os.chdir(build_dir)

    call(['git', 'clone', 'https://github.com/tensorflow/benchmarks.git'])
    os.chdir('benchmarks')
    call(['git', 'checkout', '4c7b09ad87bbfc4b1f89650bcee40b3fc5e7dfed'])

    junit_script = os.path.abspath('%s/test/ci/junit-wrap.sh' % root_pwd)

    # Check to see if we need to patch the repo for Grappler
    # benchmark_cnn.patch will only work for the CPU backend
    patch_file = os.path.abspath(
        os.path.join(ngraph_tf_src_dir, "test/grappler/benchmark_cnn.patch"))
    import ngraph_bridge
    if ngraph_bridge.is_grappler_enabled():
        print("Patching repo using: %s" % patch_file)
        apply_patch(patch_file)

    os.chdir('scripts/tf_cnn_benchmarks/')
    # Update the script by adding `import ngraph_bridge`
    with open('convnet_builder.py', 'a') as outfile:
        call(['echo', 'import ngraph_bridge'], stdout=outfile)

    # Setup the env flags
    import psutil
    num_cores = int(psutil.cpu_count(logical=False))
    print("OMP_NUM_THREADS: %s " % str(num_cores))

    os.environ['OMP_NUM_THREADS'] = str(num_cores)
    os.environ["KMP_AFFINITY"] = 'granularity=fine,compact,1,0'

    os.environ['JUNIT_WRAP_FILE'] = "%s/junit_inference_test.xml" % build_dir
    os.environ['JUNIT_WRAP_SUITE'] = 'models'
    os.environ['JUNIT_WRAP_TEST'] = 'resnet50-inference'

    # Run inference job
    cmd = [
        junit_script, 'python', 'tf_cnn_benchmarks.py', '--data_format', 'NHWC',
        '--num_inter_threads', '2', '--freeze_when_forward_only=True',
        '--model=resnet50', '--batch_size=1', '--num_batches', '32'
    ]
    command_executor(cmd, verbose=True)
    os.chdir(root_pwd)


def run_resnet50_forward_pass_from_artifacts(ngraph_tf_src_dir, artifact_dir,
                                             batch_size, iterations):

    root_pwd = os.getcwd()
    artifact_dir = os.path.abspath(artifact_dir)
    ngraph_tf_src_dir = os.path.abspath(ngraph_tf_src_dir)
    install_ngraph_bridge(artifact_dir)

    # Now clone the repo and proceed
    call(['git', 'clone', 'https://github.com/tensorflow/benchmarks.git'])
    os.chdir('benchmarks')
    call(['git', 'checkout', '4c7b09ad87bbfc4b1f89650bcee40b3fc5e7dfed'])

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

    cmd = [
        'python',
        'tf_cnn_benchmarks.py',
        '--data_format',
        'NHWC',
        '--num_inter_threads',
        '2',
        '--freeze_when_forward_only=True',
        '--model=resnet50',
        '--batch_size=' + str(batch_size),
        '--num_batches',
        str(iterations),
    ]
    command_executor(cmd, verbose=True)

    os.chdir(root_pwd)


# See https://github.com/IntelAI/models/blob/master/benchmarks/image_recognition/tensorflow/resnet50v1_5/README.md#fp32-inference-instructions
def run_intelaimodels_resnet50_infer_from_artifacts(
        ngraph_tf_src_dir, artifact_dir, batch_size, iterations):
    root_pwd = os.getcwd(
    )  # e.g. /localdisk/buildkite-agent/builds/aipg-ra-skx-168-2/ngraph/ngtf-cpu-ubuntu
    artifact_dir = os.path.abspath(artifact_dir)
    if not os.path.exists(artifact_dir):
        raise Exception("Can't find artifact dir: " + artifact_dir)
    ngraph_tf_src_dir = os.path.abspath(ngraph_tf_src_dir)
    if (len(glob.glob(artifact_dir + "/ngraph_tensorflow_bridge-*.whl")) == 0):
        install_ngraph_bridge(artifact_dir)

    # Check/download pretrained model
    pretrained_models_dir = root_pwd + '/pretrained_models'
    if not os.path.exists(pretrained_models_dir):
        os.mkdir(pretrained_models_dir, 0o755)
    os.chdir(pretrained_models_dir)
    pretrained_model = pretrained_models_dir + '/resnet50_v1.pb'
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

    # Now clone the IntelAI repo and proceed
    os.chdir(root_pwd)
    IntelAIModels_dir = root_pwd + '/IntelAI-MODELS'
    if not os.path.exists(IntelAIModels_dir):
        call([
            'git', 'clone', 'https://github.com/IntelAI/models.git',
            'IntelAI-MODELS'
        ])
        os.chdir('IntelAI-MODELS')
        call(['git', 'checkout', 'ae52473'])  # == master as of 2020-04-21

    # Update file: ./models/image_recognition/tensorflow/resnet50v1_5/inference/eval_image_classifier_inference.py
    script_file = IntelAIModels_dir + '/models/image_recognition/tensorflow/resnet50v1_5/inference/eval_image_classifier_inference.py'
    if not os.path.exists(script_file):
        raise Exception("Can't find script file: " + script_file)
    # ^[ ]*import[ ]*ngraph_bridge[ ]*$
    res = subprocess.run(
        'grep \'import ngraph_bridge\' ' + script_file + ' | wc -l',
        shell=True,
        stdout=subprocess.PIPE)
    if res.stdout.decode('utf-8').rstrip() == '0':
        print('Updating script with \'import ngraph_bridge\' ...')
        with open(script_file, 'r') as filedesc:
            data = filedesc.readlines()  # data[0], data[1], ...
        data[166] = data[166] + '    import ngraph_bridge\n'
        with open(script_file, 'w') as filedesc:
            filedesc.writelines(data)
        res = subprocess.run(
            'grep \'import ngraph_bridge\' ' + script_file + ' | wc -l',
            shell=True,
            stdout=subprocess.PIPE)
        if res.stdout.decode('utf-8').rstrip() == 0:
            raise Exception("Can't add 'import ngraph_bridge' to script file: "
                            + script_file)

    # Setup the env flags
    import psutil
    num_cores = int(psutil.cpu_count(logical=False))
    print("OMP_NUM_THREADS: %s " % str(num_cores))
    os.environ['OMP_NUM_THREADS'] = str(num_cores)
    os.environ["KMP_AFFINITY"] = 'granularity=fine,compact,1,0'

    # Delete older logs
    log_dir = IntelAIModels_dir + '/benchmarks/common/tensorflow/logs'
    if os.path.exists(log_dir):
        #os.rmdir(log_dir)
        filelist = glob.glob(os.path.join(log_dir, "*.log"))
        for f in filelist:
            os.remove(f)

    os.chdir(IntelAIModels_dir + '/benchmarks/')
    # python launch_benchmark.py  --in-graph resnet50_v1.pb  --model-name resnet50v1_5
    # --framework tensorflow  --precision fp32  --mode inference  --batch-size=1  --socket-id=0
    cmd = [
        'python',
        'launch_benchmark.py',
        '--in-graph',
        pretrained_model,
        '--model-name resnet50v1_5',
        '--framework tensorflow',
        '--precision fp32',
        '--mode inference',
        '--batch-size=1',
        '--socket-id=0',
    ]
    command_executor(cmd, verbose=True)
    os.chdir(root_pwd)


def run_resnet50_infer_from_artifacts(ngraph_tf_src_dir, artifact_dir,
                                      batch_size, iterations):
    root_pwd = os.getcwd(
    )  # e.g. /localdisk/buildkite-agent/builds/aipg-ra-skx-168-2/ngraph/ngtf-cpu-ubuntu
    artifact_dir = os.path.abspath(artifact_dir)
    if not os.path.exists(artifact_dir):
        raise Exception("Can't find artifact dir: " + artifact_dir)
    ngraph_tf_src_dir = os.path.abspath(ngraph_tf_src_dir)
    if (len(glob.glob(artifact_dir + "/ngraph_tensorflow_bridge-*.whl")) == 0):
        install_ngraph_bridge(artifact_dir)

    # Check/download pretrained model
    pretrained_models_dir = root_pwd + '/pretrained_models'
    if not os.path.exists(pretrained_models_dir):
        os.mkdir(pretrained_models_dir, 0o755)
    os.chdir(pretrained_models_dir)
    pretrained_model = pretrained_models_dir + '/resnet50_v1.pb'
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
        ngraph_tf_src_dir + '/test/python/test_rn50_infer.py',
        '--input-graph',
        pretrained_model,
        '--batch-size',
        str(batch_size),
        '--num-images',
        str(batch_size * iterations),
    ]
    command_executor(cmd, verbose=True)
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
