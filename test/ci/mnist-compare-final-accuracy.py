#!/usr/bin/env python

# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import argparse
import os
import os.path
import re
import subprocess
import sys

import mnist_softmax_util as msu

#===================================================================================================


def ensure_mnist_data_local_copy(data_download_script_path, data_dir):

    cmd = [data_download_script_path, data_dir]
    print "script: ", data_download_script_path
    print "data_dir: ", data_dir
    p = subprocess.Popen(
        cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.wait()
    if p.returncode != 0:
        raise Exception("ERROR: Subprocess command failed: {}".format(cmd))


#===================================================================================================


def main(args):
    sys.stdout.write('\n')
    sys.stdout.write('ABOUT TO DOWNLOAD MNIST DATA (IF NECESSARY)\n')
    sys.stdout.write('\n')

    download_mnist_data_script_path = './download-mnist-data.sh'
    mnist_data_dir = './data_dir'

    ensure_mnist_data_local_copy(download_mnist_data_script_path,
                                 mnist_data_dir)

    enable_global_jit = True
    allow_soft_placement = True
    log_device_placement = False
    batch_size = 100
    graph_summary_dir_or_None = None
    final_iteration_chrome_trace_filename_or_None = None
    return_per_iteration_cost_value = False
    return_per_iteration_accuracy_value = False
    log_stream = sys.stdout

    # To help use decide what tolerances to require for our tests, here's some
    # observed data picked up from the console logs of the CI job
    # http://cje.amr.corp.intel.com/aipg-algo/job/ngraph-tensorflow-unittest/116/console
    #
    # With 300 training iterations and graph_random_seed = 0:
    #    XLA_CPU accuracy:    0.908499956131
    #                         0.90649998188
    #
    #    XLA_NGRAPH accuracy (INTERPRETER): 0.911899983883
    #    XLA_NGRAPH accuracy (CPU): 0.911899983883
    #

    tests = [
        {
            'dev': '/job:localhost/replica:0/task:0/device:XLA_CPU:0',
            'num_iterations': 300,
            'min_final_accuracy': 0.90,
            'max_final_accuracy': 0.92,
        },
        {
            'dev': '/job:localhost/replica:0/task:0/device:NGRAPH:0',
            'num_iterations': 300,
            'min_final_accuracy': 0.90,
            'max_final_accuracy': 0.92,
        },
    ]

    all_tests_pass = True

    for t in tests:
        sys.stdout.write('\n')
        sys.stdout.write('BEGIN MNIST MLP RUN FOR DEVICE {}\n'.format(
            t['dev']))
        sys.stdout.write('\n')

        tf_device_str_or_None = t['dev']
        num_training_iterations = t['num_iterations']

        results = msu.train_and_evaluate_model(
            mnist_data_dir,
            tf_device_str_or_None,
            enable_global_jit,
            allow_soft_placement,
            log_device_placement,
            num_training_iterations,
            batch_size,
            graph_summary_dir_or_None,
            final_iteration_chrome_trace_filename_or_None,
            return_per_iteration_cost_value,
            return_per_iteration_accuracy_value,
            log_stream,
        )

        test_passes = \
            (results['final_accuracy'] >= t['min_final_accuracy']) and \
            (results['final_accuracy'] <= t['max_final_accuracy'])

        if not test_passes:
            all_tests_pass = False

        sys.stdout.write('\nFinal accuracy: {}\n'.format(
            results['final_accuracy']))
        sys.stdout.write('  Min accuracy: {}\n'.format(
            t['min_final_accuracy']))
        sys.stdout.write('  Max accuracy: {}\n'.format(
            t['max_final_accuracy']))
        sys.stdout.write('  {}\n'.format('OK' if test_passes else 'FAILED'))

    if all_tests_pass:
        sys.stdout.write('\nALL TESTS PASSED\n')
    else:
        sys.stdout.write('\nSOME TESTS FAILED\n')
        sys.exit(1)


#===================================================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='''
This script performs two MNIST runs that differ only regarding which XLA device they prefer:
'XLA_NGRAPH' or 'XLA_CPU'.  If the reported accuracy of their final trained models is sufficiently
close then this script exits with code 0.  Otherwise it exits with a non-zero exit code.
''')

    parser.add_argument(
        '--data_dir',
        type=str,
        default='/tmp/tensorflow/mnist/input_data',
        help='Directory for storing input data')

    args = parser.parse_args()

    main(args)
