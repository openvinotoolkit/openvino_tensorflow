This document provides a list of all validated models that are supported by Intel<sup>®</sup> OpenVINO™ integration with TensorFlow. This list is continuously evolving as we enable more operators and models. 

## TensorFlow-Slim Image Classification Library
| Model Name | Supported Devices |
|---|---|
| [Inception V3](http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz)  | CPU, iGPU, MYRIAD, VAD-M |
| [Inception_V4](http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz)  | CPU, iGPU, MYRIAD, VAD-M |
| [Resnet V1 50](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz)  | CPU, iGPU, MYRIAD, VAD-M |
| [Resnet V2 152](http://download.tensorflow.org/models/resnet_v2_152_2017_04_14.tar.gz)| CPU, iGPU, MYRIAD, VAD-M |
| [Resnet V2 50](http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz)  | CPU, iGPU, MYRIAD, VAD-M |
| [VGG 16](http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz)              | CPU, iGPU, MYRIAD, VAD-M |
| [VGG 19](http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz)              | CPU, iGPU, MYRIAD, VAD-M |
| [MobileNet_v1_1.0_224](http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz)| CPU, iGPU, MYRIAD, VAD-M |
| [MobileNet_v2_1.4_224](https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.4_224.tgz)| CPU |
| [CifarNet](https://github.com/tensorflow/models/blob/master/research/slim/nets/cifarnet.py)| CPU, iGPU, MYRIAD, VAD-M |
| [LeNet](https://github.com/tensorflow/models/blob/master/research/slim/nets/lenet.py)| CPU, iGPU, MYRIAD, VAD-M |

The links to the TensorFlow-Slim models include the pre-trained checkpoint files only. You should refer to TensorFlow-Slim models [instructions page](https://github.com/tensorflow/models/tree/master/research/slim) to run inference or freeze the models. (No pre-trained checkpoint files provided for CifarNet and LeNet.)

## Tensorflow Object Detection Model Zoo
| Model Name | Supported Devices |
|---|---|
| [faster_rcnn_inception_resnet_v2_atrous_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28.tar.gz)| CPU, iGPU, MYRIAD, VAD-M |
| [faster_rcnn_inception_v2_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz)| CPU, iGPU, MYRIAD, VAD-M |
| [faster_rcnn_resnet50_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gzs)| CPU, iGPU, MYRIAD, VAD-M |
| [faster_rcnn_resnet101_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_2018_01_28.tar.gz)| CPU, iGPU, MYRIAD, VAD-M |
| [faster_rcnn_resnet50_lowproposals_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_lowproposals_coco_2018_01_28.tar.gz)| CPU, iGPU, MYRIAD, VAD-M |
| [ssd_inception_v2](http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz)| CPU, iGPU, MYRIAD, VAD-M |
| [ssd_mobilenet_v1](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz)| CPU, iGPU, MYRIAD, VAD-M |
| [ssd_mobilenet_v1_fpn](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz)| CPU, iGPU, MYRIAD, VAD-M |
| [ssd_mobilenet_v2](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz)| CPU, iGPU, MYRIAD, VAD-M |
| [ssd_resnet_50_fpn](http://download.tensorflow.org/models/object_detection/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz)| CPU, iGPU, MYRIAD, VAD-M |
| [ssdlite_mobilenet_v2](http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz)| CPU, iGPU, MYRIAD, VAD-M |
| [mask_rcnn_inception_resnet_v2_atrous_coco](http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28.tar.gz)| CPU, iGPU, MYRIAD, VAD-M |
| [mask_rcnn_inception_v2_coco](http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_v2_coco_2018_01_28.tar.gz)| CPU, iGPU, MYRIAD, VAD-M |

Pre-trained frozen models are provided for these models.

## TF Keras Applications
| Model Name | Supported Devices |
|---|---|
| [DenseNet121](https://www.tensorflow.org/api_docs/python/tf/keras/applications/DenseNet121)| CPU, iGPU, MYRIAD, VAD-M |
| [DenseNet169](https://www.tensorflow.org/api_docs/python/tf/keras/applications/DenseNet169)| CPU, iGPU, MYRIAD, VAD-M |
| [DenseNet201](https://www.tensorflow.org/api_docs/python/tf/keras/applications/DenseNet201)| CPU, iGPU, MYRIAD, VAD-M |
| [EfficientnetB0](https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB0)| CPU, iGPU, MYRIAD, VAD-M |
| [EfficientnetB1](https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB1)| CPU, iGPU, MYRIAD, VAD-M |
| [EfficientnetB2](https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB2)| CPU, iGPU, MYRIAD, VAD-M |
| [EfficientnetB3](https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB3)| CPU, iGPU, MYRIAD, VAD-M |
| [EfficientnetB4](https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB4)| CPU, iGPU, MYRIAD, VAD-M |
| [EfficientnetB5](https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB5)| CPU, iGPU, MYRIAD, VAD-M |
| [EfficientnetB6](https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB6)| CPU, iGPU, MYRIAD, VAD-M |
| [EfficientnetB7](https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB7)| CPU, iGPU, MYRIAD, VAD-M |
| [InceptionV3](https://www.tensorflow.org/api_docs/python/tf/keras/applications/InceptionV3)| CPU, iGPU, MYRIAD, VAD-M |
| [NASNetLarge](https://www.tensorflow.org/api_docs/python/tf/keras/applications/NASNetLarge)| CPU, iGPU, MYRIAD, VAD-M |
| [NASNetMobile](https://www.tensorflow.org/api_docs/python/tf/keras/applications/NASNetMobile)| CPU, iGPU, MYRIAD, VAD-M |
| [ResNet50v2](https://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet50V2)| CPU, iGPU, MYRIAD, VAD-M |

Please follow the instructions on [Keras Applications](https://keras.io/api/applications/) page for further information about using these pre-trained models.

## Other Models
| Model Name | Supported Devices |
|---|---|
| [3d-pose-baseline](https://drive.google.com/file/d/0BxWzojlLp259MF9qSFpiVjl0cU0/view?usp=sharing)| CPU, iGPU, MYRIAD |
| [CRNN](https://github.com/MaybeShewill-CV/CRNN_Tensorflow)| CPU, iGPU, MYRIAD, VAD-M |
| [densenet161](https://drive.google.com/file/d/0B_fUSpodN0t0NmZvTnZZa2plaHc/view)| CPU, iGPU, MYRIAD, VAD-M |
| [east_resnet_v1_50](https://github.com/argman/EAST#download)| CPU, iGPU, MYRIAD, VAD-M |
| [efficientnet-b7_auto_aug](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/ckptsaug/efficientnet-b7.tar.gz)| CPU, iGPU, MYRIAD, VAD-M |
| [efficientnet-b0_auto_aug](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/ckptsaug/efficientnet-b0.tar.gz)| CPU, iGPU |
| [fc_densenet_103](https://github.com/AI-slam/FC-DenseNet-Tiramisu)| CPU |
| [fcrn-dp-nyu-depth-v2-tf](http://campar.in.tum.de/files/rupprecht/depthpred/NYU_FCRN-checkpoint.zip)| CPU, iGPU, MYRIAD, VAD-M |
| [googlenet-v1](http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz)| CPU, iGPU, MYRIAD, VAD-M |
| [googlenet-v2](http://download.tensorflow.org/models/inception_v2_2016_08_28.tar.gz)| CPU, iGPU, MYRIAD, VAD-M |
| [googlenet-v4](http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz)| CPU, iGPU, MYRIAD, VAD-M |
| [i3d-rgb](https://download.01.org/opencv/public_models/032020/i3d-rgb/rgb.frozen.pb)| CPU, iGPU, MYRIAD |
| [inception_resnet_v2](https://github.com/openvinotoolkit/open_model_zoo/blob/2021.2/models/public/inception-resnet-v2-tf/model.yml)| CPU, iGPU, MYRIAD, VAD-M |
| [license-plate-recognition-barrier-0007](https://download.01.org/openvinotoolkit/training_toolbox_tensorflow/models/lpr/chinese_lp/license-plate-recognition-barrier-0007.tar.gz)| CPU, iGPU, MYRIAD |
| [mobilenet_v1_0.25_128](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.25_128.tgz)| CPU, iGPU, MYRIAD, VAD-M |
| [mobilenet_v1_0.50_160](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.5_160.tgz)| CPU, iGPU, MYRIAD, VAD-M |
| [mobilenet_v1_0.50_224](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.5_224.tgz)| CPU, iGPU, MYRIAD, VAD-M |
| [mobilenet_v1_1.0_224](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz)| CPU, iGPU, MYRIAD, VAD-M |
| [mobilenet-v3-large-1.0-224-tf](https://storage.googleapis.com/mobilenet_v3/checkpoints/v3-large_224_1.0_float.tgz)| CPU, iGPU, MYRIAD, VAD-M |
| [mobilenet-v3-small-1.0-224-tf](https://storage.googleapis.com/mobilenet_v3/checkpoints/v3-small_224_1.0_float.tgz)| CPU, iGPU, MYRIAD, VAD-M |
| [mozilla-deepspeech-0.7.1](https://github.com/mozilla/DeepSpeech/archive/v0.7.1.tar.gz)| CPU, iGPU |
| [netvlad](http://rpg.ifi.uzh.ch/datasets/netvlad/vd16_pitts30k_conv5_3_vlad_preL2_intra_white.zip)| CPU, MYRIAD |
| [NiftyNet](https://github.com/NifTK/NiftyNetModelZoo/tree/5-reorganising-with-lfs/mr_ct_regression)| CPU, iGPU, MYRIAD |
| [openpose-pose](http://www.mediafire.com/file/qlzzr20mpocnpa3/graph_opt.pb)| CPU, iGPU, MYRIAD, VAD-M |
| [PRNet](https://github.com/YadiraF/PRNet)| CPU, iGPU, MYRIAD, VAD-M |
| [resnet_50](https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/resnet50_fp32_pretrained_model.pb)| CPU, iGPU, MYRIAD, VAD-M |
| [resnet-101](http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz)| CPU, iGPU, MYRIAD, VAD-M |
| [resnet-152](http://download.tensorflow.org/models/resnet_v1_152_2016_08_28.tar.gz)| CPU, iGPU, MYRIAD, VAD-M |
| [resnet_50_v1.5](https://zenodo.org/record/2535873/files/resnet50_v1.pb)| CPU, iGPU, MYRIAD, VAD-M |
| [se-resnext-50](https://drive.google.com/drive/folders/1k5MtfqbNRA8ziE3f18vu00Q1FQCzk4__)| CPU, iGPU, MYRIAD, VAD-M |
| [srgan](https://github.com/tensorlayer/srgan)| CPU, iGPU |
| [STN](https://github.com/oarriaga/STN.keras)| CPU, iGPU, MYRIAD, VAD-M |
| [vehicle-license-plate-detection-barrier-0123](https://download.01.org/opencv/openvino_training_extensions/models/ssd_detector/ssd-mobilenet-v2-0.35.1-barrier-256x256-0123.tar.gz)| CPU, iGPU, MYRIAD, VAD-M |
| [vggvox](https://github.com/linhdvu14/vggvox-speaker-identification)| CPU, iGPU, MYRIAD |
| [yolo_v2](https://github.com/david8862/keras-YOLOv3-model-set.git)| CPU, iGPU, MYRIAD, VAD-M |
| [yolo_v2_tiny](https://github.com/david8862/keras-YOLOv3-model-set.git)| CPU, iGPU, MYRIAD, VAD-M |
| [yolo_v3_darknet](https://github.com/mystic123/tensorflow-yolo-v3.git)| CPU, iGPU, MYRIAD, VAD-M |
| [yolo-v3](https://download.01.org/opencv/public_models/022020/yolo_v3/yolov3.pb)| CPU, iGPU, MYRIAD, VAD-M |
| [yolo-v3-tiny-tf](https://download.01.org/opencv/public_models/082020/yolo-v3-tiny-tf/yolo-v3-tiny-tf.zip)| CPU, iGPU, MYRIAD, VAD-M |
| [yolo-v4](https://github.com/david8862/keras-YOLOv3-model-set)| CPU, iGPU, MYRIAD, VAD-M |

Pre-trained frozen model files are provided for some of these models. For the rest, please refer to the links provided.
