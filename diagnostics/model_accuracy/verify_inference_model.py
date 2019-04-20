# Copyright 2019 Intel Corporation
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


def run_inference(model_name, models_dir):
    parameters = \
    '[{"model_type" : "Image Recognition", "model_name" : "Inception_v4", \
        "cmd" : "OMP_NUM_THREADS=28 KMP_AFFINITY=granularity=fine,compact,1,0 \
                python eval_image_classifier.py --alsologtostderr \
                --checkpoint_path=/nfs/fm/disks/aipg_trained_dataset/ngraph_tensorflow/fully_trained/inception4_slim/inception_v4.ckpt \
                --dataset_dir=/mnt/data/TF_ImageNet_latest/ --dataset_name=imagenet \
                --dataset_split_name=validation --model_name=inception_v4"},\
     {"model_type" : "Image Recognition", "model_name" : "MobileNet_v1", \
        "cmd" : "OMP_NUM_THREADS=28 KMP_AFFINITY=granularity=fine,compact,1,0 \
                python eval_image_classifier.py --alsologtostderr \
                --checkpoint_path=/nfs/fm/disks/aipg_trained_dataset/ngraph_tensorflow/fully_trained/mobilenet_v1/mobilenet_v1_1.0_224.ckpt \
                --dataset_dir=/mnt/data/TF_ImageNet_latest/ --dataset_name=imagenet \
                --dataset_split_name=validation --model_name=mobilenet_v1"}, \
     {"model_type" : "Image Recognition", "model_name" : "ResNet50_v1", \
        "cmd" : "OMP_NUM_THREADS=28 KMP_AFFINITY=granularity=fine,compact,1,0 \
                python eval_image_classifier.py --alsologtostderr \
                --checkpoint_path=/nfs/fm/disks/aipg_trained_dataset/ngraph_tensorflow/fully_trained/resnet50_v1_slim/resnet_v1_50.ckpt \
                --dataset_dir=/mnt/data/TF_ImageNet_latest/ --dataset_name=imagenet \
                --dataset_split_name=validation --model_name=resnet_v1_50 --labels_offset=1"}, \
     {"model_type" : "Object Detection", "model_name" : "SSD-MobileNet_v1", \
        "cmd" : "OMP_NUM_THREADS=28 KMP_AFFINITY=granularity=fine,compact,1,0 \
                python object_detection/model_main.py --logtostderr \
                --pipeline_config_path=object_detection/samples/configs/ssd_mobilenet_v1_coco.config \
                --checkpoint_dir=/nfs/site/disks/aipg_trained_dataset/ngraph_tensorflow/fully_trained/ssd_mobilenet_v1_coco_2018_01_28/ \
                --run_once=True"}]'

    try:
        data = json.loads(parameters)
    except:
        print("Pass a valid model prameters dictionary")
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


def check_accuracy(model, p, tolerance=0.001):
    #check if the accuracy of the model inference matches with the published numbers
    #Accuracy values picked up from here https://github.com/tensorflow/models/tree/master/research/slim
    accuracy = \
    '[{"model_name" : "Inception_v4", "accuracy" : "0.802"},\
     {"model_name" : "ResNet50_v1", "accuracy" : "0.752"}, \
     {"model_name" : "MobileNet_v1", "accuracy" : "0.709"}]'

    data = json.loads(accuracy)

    for line in p.splitlines():
        print(line.decode())
        if ('eval/Accuracy'.encode() in line):
            accuracy = re.split("eval/Accuracy", line.decode())[1]
            top1_accuracy = re.search(r'\[(.*)\]', accuracy).group(1)
        #for now we just validate top 1 accuracy, but calculating top5 anyway.
        if ('eval/Recall_5'.encode() in line):
            accuracy = re.split("eval/Recall_5", line.decode())[1]
            top5_accuracy = float(re.search("\[(.*?)\]", accuracy).group(1))

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
        model_name, p = run_inference(args.model_name, models_dir)
        check_accuracy(model_name, p)
    except Exception as ex:
        print("Model accuracy verification failed. Exception: %s" % str(ex))
