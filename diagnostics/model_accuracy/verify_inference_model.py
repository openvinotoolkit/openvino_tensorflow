# ******************************************************************************
# Copyright (C) 2021 Intel Corporation
 
# SPDX-License-Identifier: Apache-2.0
# ******************************************************************************

import pdb
from subprocess import check_output, call, Popen, PIPE, STDOUT
import re
import json, shlex, os, argparse, sys


def parse_json(json_file_name):
    with open(json_file_name) as f:
        return json.load(f)


def command_executor(cmd, verbose=True, msg=None):
    if verbose or msg is not None:
        tag = 'Running COMMAND: ' if msg is None else msg
        print(tag + cmd)

    ps = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
    so, se = ps.communicate()
    return so


def download_repo(repo, target_name=None, version='master'):
    # First download repo
    command_executor("git clone " + repo)


def run_inference(model_name, models_dir, json_file_name):

    try:
        data = parse_json(args.json_file_name)
    except:
        print("Pass a valid model prameters dictionary")
        sys.exit(1)
    pwd = os.getcwd()
    for i, d in enumerate(data):

        if (model_name in data[i]["model_name"]):
            if (data[i]["model_type"] == "Image Recognition"):
                if models_dir is None:
                    os.chdir(pwd + "/models/research/slim")
                else:
                    os.chdir(models_dir + "research/slim")
                command_executor("export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`")
                command_executor('git apply ' + pwd +
                                 '/image_recognition.patch')
            p = command_executor(data[i]["cmd"])
            os.chdir(pwd)
            return model_name, p


def check_accuracy(model, p, json_file_name, tolerance=0.001):
    #check if the accuracy of the model inference matches with the published numbers
    #Accuracy values picked up from here https://github.com/tensorflow/models/tree/master/research/slim

    data = parse_json(args.json_file_name)
    for line in p.splitlines():
        print(line.decode())
        if ('eval/Accuracy'.encode() in line):
            is_match = re.search('eval/Accuracy\[([0-9.]+)]', line.decode())
            if is_match and len(is_match.groups()) > 0:
                top1_accuracy = is_match.group(1)

        #for now we just validate top 1 accuracy, but calculating top5 anyway.
        if ('eval/Recall_5'.encode() in line):
            is_match = re.search('.+eval/Recall_5\[([0-9.]+)]', line.decode())
            if is_match and len(is_match.groups()) > 0:
                top5_accuracy = is_match.group(1)

    for i, d in enumerate(data):
        if (model in data[i]["model_name"]):
            # Tolerance check
            diff = float(data[i]["accuracy"]) - float(top1_accuracy)
            print('\033[1m' + '\nModel Accuracy Verification' + '\033[0m')
            if (diff > tolerance):
                print('\033[91m' + 'FAIL' + '\033[0m' +
                      " Functional accuracy " + top1_accuracy +
                      " is not as expected for " + data[i]["model_name"] +
                      "\nExpected accuracy = " + data[i]["accuracy"])
                return False
            else:
                print('\033[92m' + 'PASS' + '\033[0m' +
                      " Functional accuracy " + top1_accuracy +
                      " is as expected for " + data[i]["model_name"])
                return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Accuracy verification for TF models using ngraph')

    parser.add_argument(
        '--model_name',
        help=
        'Model name to run inference. Availble models are Inception_v4, MobileNet_v1, ResNet50_v1',
        required=True)

    parser.add_argument(
        '--json_file_name',
        help=
        'json file with model parameters to run inference and accuracy values to verify',
        required=True)
    parser.add_argument(
        '--models_dir',
        help='Source of the model repository location on disk \
        If not specified, the local repository will be used \
        Example:ngraph-tf/diagnostics/model_accuracy/models')
    cwd = os.getcwd()

    args = parser.parse_args()

    if (args.models_dir):
        models_dir = args.models_dir
    else:
        models_dir = None
        repo = "https://github.com/tensorflow/models.git"
        download_repo(repo)

    #Just takes in one model at a time for now
    #TODO(Sindhu): Run multiple or ALL models at once and compare accuracy.

    try:
        model_name, p = run_inference(args.model_name, models_dir,
                                      args.json_file_name)
        check_accuracy(model_name, p, args.json_file_name)
        if check_accuracy(model_name, p, args.json_file_name):
            sys.exit(0)
        else:
            sys.exit(1)

    except Exception as ex:
        print("Model accuracy verification failed. Exception: %s" % str(ex))
        sys.exit(1)
