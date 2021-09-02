本文列出了所有经过验证，且 **OpenVINO™ integration with TensorFlow** 支持的模型。随着我们支持越来越多的算子和模型，该列表将会持续更新。

## TensorFlow-Slim 图像分类库

| 模型名称| 支持的设备
|----------|----------
| [Inception V3](http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz)| CPU，iGPU，MYRIAD，VAD-M
| [Inception\_V4](http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz)| CPU，iGPU，MYRIAD，VAD-M
| [Resnet V1 50](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz)| CPU，iGPU，MYRIAD，VAD-M
| [Resnet V2 152](http://download.tensorflow.org/models/resnet_v2_152_2017_04_14.tar.gz)| CPU，iGPU，MYRIAD，VAD-M
| [Resnet V2 50](http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz)| CPU，iGPU，MYRIAD，VAD-M
| [VGG 16](http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz)| CPU，iGPU，MYRIAD，VAD-M
| [VGG 19](http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz)| CPU，iGPU，MYRIAD，VAD-M
| [MobileNet\_v1\_1.0\_224](http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz)| CPU，iGPU，MYRIAD，VAD-M
| [MobileNet\_v2\_1.4\_224](https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.4_224.tgz)| CPU
| [CifarNet](https://github.com/tensorflow/models/blob/master/research/slim/nets/cifarnet.py)| CPU，iGPU，MYRIAD，VAD-M
| [LeNet](https://github.com/tensorflow/models/blob/master/research/slim/nets/lenet.py)| CPU，iGPU，MYRIAD，VAD-M

TensorFlow-Slim 模型链接仅包含预训练checkpoint文件。您可以参考 TensorFlow-Slim 模型[说明页面](https://github.com/tensorflow/models/tree/master/research/slim)，以便运行推理或冻结模型。（不提供针对 CifarNet 和 LeNet 的预训练checkpoint文件。）

## TensorFlow 对象检测 Model Zoo

| 模型名称| 支持的设备
|----------|----------
| [faster\_rcnn\_inception\_resnet\_v2\_atrous\_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28.tar.gz)| CPU，iGPU，MYRIAD，VAD-M
| [faster\_rcnn\_inception\_v2\_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz)| CPU，iGPU，MYRIAD，VAD-M
| [faster\_rcnn\_resnet50\_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz)| CPU，iGPU，MYRIAD，VAD-M
| [faster\_rcnn\_resnet101\_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_2018_01_28.tar.gz)| CPU，iGPU，MYRIAD，VAD-M
| [faster\_rcnn\_resnet50\_lowproposals\_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_lowproposals_coco_2018_01_28.tar.gz)| CPU，iGPU，MYRIAD，VAD-M
| [ssd\_inception\_v2](http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz)| CPU，iGPU，MYRIAD，VAD-M
| [ssd\_mobilenet\_v1](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz)| CPU，iGPU，MYRIAD，VAD-M
| [ssd\_mobilenet\_v1\_fpn](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz)| CPU，iGPU，MYRIAD，VAD-M
| [ssd\_mobilenet\_v2](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz)| CPU，iGPU，MYRIAD，VAD-M
| [ssd\_resnet\_50\_fpn](http://download.tensorflow.org/models/object_detection/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz)| CPU，iGPU，MYRIAD，VAD-M
| [ssdlite\_mobilenet\_v2](http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz)| CPU，iGPU，MYRIAD，VAD-M
| [mask\_rcnn\_inception\_resnet\_v2\_atrous\_coco](http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28.tar.gz)| CPU，iGPU，MYRIAD，VAD-M
| [mask\_rcnn\_inception\_v2\_coco](http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_v2_coco_2018_01_28.tar.gz)| CPU，iGPU，MYRIAD，VAD-M

为这些模型提供了预训练frozen模型。

## TensorFlow Keras 应用

| 模型名称| 支持的设备
|----------|----------
| [DenseNet121](https://www.tensorflow.org/api_docs/python/tf/keras/applications/DenseNet121)| CPU，iGPU，MYRIAD，VAD-M
| [DenseNet169](https://www.tensorflow.org/api_docs/python/tf/keras/applications/DenseNet169)| CPU，iGPU，MYRIAD，VAD-M
| [DenseNet201](https://www.tensorflow.org/api_docs/python/tf/keras/applications/DenseNet201)| CPU，iGPU，MYRIAD，VAD-M
| [EfficientnetB0](https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB0)| CPU，iGPU，MYRIAD，VAD-M
| [EfficientnetB1](https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB1)| CPU，iGPU，MYRIAD，VAD-M
| [EfficientnetB2](https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB2)| CPU，iGPU，MYRIAD，VAD-M
| [EfficientnetB3](https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB3)| CPU，iGPU，MYRIAD，VAD-M
| [EfficientnetB4](https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB4)| CPU，iGPU，MYRIAD，VAD-M
| [EfficientnetB5](https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB5)| CPU，iGPU，MYRIAD，VAD-M
| [EfficientnetB6](https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB6)| CPU，iGPU，MYRIAD，VAD-M
| [EfficientnetB7](https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB7)| CPU，iGPU，MYRIAD，VAD-M
| [InceptionV3](https://www.tensorflow.org/api_docs/python/tf/keras/applications/InceptionV3)| CPU，iGPU，MYRIAD，VAD-M
| [NASNetLarge](https://www.tensorflow.org/api_docs/python/tf/keras/applications/NASNetLarge)| CPU，iGPU，MYRIAD，VAD-M
| [NASNetMobile](https://www.tensorflow.org/api_docs/python/tf/keras/applications/NASNetMobile)| CPU，iGPU，MYRIAD，VAD-M
| [ResNet50v2](https://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet50V2)| CPU，iGPU，MYRIAD，VAD-M

请参考 [Keras 应用](https://keras.io/api/applications/)页面上的说明，了解更多关于如何使用这些预训练模型的信息。

## 其他模型

| 模型名称| 支持的设备
|----------|----------
| [3d-pose-baseline](https://drive.google.com/file/d/0BxWzojlLp259MF9qSFpiVjl0cU0/view?usp=sharing)| CPU，iGPU，MYRIAD
| [cpm-person](https://github.com/CMU-Perceptual-Computing-Lab/convolutional-pose-machines-release)| CPU，iGPU，MYRIAD，VAD-M
| [cpm-pose](https://github.com/CMU-Perceptual-Computing-Lab/convolutional-pose-machines-release)| CPU，iGPU
| [CRNN](https://github.com/MaybeShewill-CV/CRNN_Tensorflow)| CPU，iGPU，MYRIAD，VAD-M
| [ctpn](https://github.com/eragonruan/text-detection-ctpn/releases/download/untagged-48d74c6337a71b6b5f87/ctpn.pb)| MYRIAD，VAD-M
| [densenet161](https://drive.google.com/file/d/0B_fUSpodN0t0NmZvTnZZa2plaHc/view)| CPU，iGPU，MYRIAD，VAD-M
| [deeplabv3](http://download.tensorflow.org/models/deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz)| CPU，MYRIAD
| [dilation](https://github.com/fyu/dilation)| CPU，iGPU
| [east\_resnet\_v1\_50](https://github.com/argman/EAST#download)| CPU，iGPU，MYRIAD，VAD-M
| [efficientnet-b7\_auto\_aug](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/ckptsaug/efficientnet-b7.tar.gz)| CPU，iGPU，MYRIAD，VAD-M
| [efficientnet-b0\_auto\_aug](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/ckptsaug/efficientnet-b0.tar.gz)| CPU，iGPU
| [faster\_rcnn\_nas\_coco](https://github.com/shaoqingren/faster_rcnn)| CPU
| [faster\_rcnn\_nas\_lowproposals\_coco](https://github.com/rbgirshick/py-faster-rcnn)| CPU
| [fc\_densenet\_103](https://github.com/AI-slam/FC-DenseNet-Tiramisu)| CPU
| [fcrn-dp-nyu-depth-v2-tf](http://campar.in.tum.de/files/rupprecht/depthpred/NYU_FCRN-checkpoint.zip)| CPU，iGPU，MYRIAD，VAD-M
| [googlenet-v1](http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz)| CPU，iGPU，MYRIAD，VAD-M
| [googlenet-v2](http://download.tensorflow.org/models/inception_v2_2016_08_28.tar.gz)| CPU，iGPU，MYRIAD，VAD-M
| [googlenet-v3](https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py)| CPU，iGPU，MYRIAD，VAD-M
| [googlenet-v4](http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz)| CPU，iGPU，MYRIAD，VAD-M
| [i3d-flow](https://github.com/deepmind/kinetics-i3d)| CPU，iGPU，MYRIAD
| [i3d-rgb](https://download.01.org/opencv/public_models/032020/i3d-rgb/rgb.frozen.pb)| CPU，iGPU，MYRIAD
| [inception\_resnet\_v2](https://github.com/openvinotoolkit/open_model_zoo/blob/2021.2/models/public/inception-resnet-v2-tf/model.yml)| CPU，iGPU，MYRIAD，VAD-M
| [license-plate-recognition-barrier-0007](https://download.01.org/openvinotoolkit/training_toolbox_tensorflow/models/lpr/chinese_lp/license-plate-recognition-barrier-0007.tar.gz)| CPU，iGPU，MYRIAD
| [mask\_rcnn\_resnet50\_atrous](https://github.com/facebookresearch/Detectron)| CPU，iGPU，MYRIAD
| [mask\_rcnn\_resnet101\_atrous\_coco](https://github.com/facebookresearch/Detectron)| CPU，iGPU，MYRIAD
| [mobilenet\_v1\_0.25\_128](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.25_128.tgz)| CPU，iGPU，MYRIAD，VAD-M
| [mobilenet\_v1\_0.50\_160](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.5_160.tgz)| CPU，iGPU，MYRIAD，VAD-M
| [mobilenet\_v1\_0.50\_224](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.5_224.tgz)| CPU，iGPU，MYRIAD，VAD-M
| [mobilenet\_v1\_1.0\_224](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz)| CPU，iGPU，MYRIAD，VAD-M
| [mobilenet\_v2\_1.0\_224](https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_224.tgz)| CPU，iGPU，MYRIAD
| [mobilenet\_v2\_fpn\_ssdlite\_crossroad](https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet)| CPU，MYRIAD
| [mobilenet-v3-large-1.0-224-tf](https://storage.googleapis.com/mobilenet_v3/checkpoints/v3-large_224_1.0_float.tgz)| CPU，iGPU，MYRIAD，VAD-M
| [mobilenet-v3-small-1.0-224-tf](https://storage.googleapis.com/mobilenet_v3/checkpoints/v3-small_224_1.0_float.tgz)| CPU，iGPU，MYRIAD，VAD-M
| [mozilla-deepspeech-0.6.1](https://github.com/mozilla/DeepSpeech/releases/download/v0.6.1/deepspeech-0.6.1-models.tar.gz)| iGPU，MYRIAD
| [mozilla-deepspeech-0.7.1](https://github.com/mozilla/DeepSpeech/archive/v0.7.1.tar.gz)| CPU，iGPU
| [mozilla-deepspeech-0.8.2](https://github.com/mozilla/DeepSpeech)| CPU，MYRIAD
| [netvlad](http://rpg.ifi.uzh.ch/datasets/netvlad/vd16_pitts30k_conv5_3_vlad_preL2_intra_white.zip)| CPU，MYRIAD
| [NiftyNet](https://github.com/NifTK/NiftyNetModelZoo/tree/5-reorganising-with-lfs/mr_ct_regression)| CPU，iGPU，MYRIAD
| [openpose-pose](http://www.mediafire.com/file/qlzzr20mpocnpa3/graph_opt.pb)| CPU，iGPU，MYRIAD，VAD-M
| [pose-ae-refinement](https://github.com/umich-vl/pose-ae-demo)| CPU
| [pose-ae-multiperson](https://github.com/umich-vl/pose-ae-demo)| CPU
| [PRNet](https://github.com/YadiraF/PRNet)| CPU，iGPU，MYRIAD，VAD-M
| [resnet\_50](https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/resnet50_fp32_pretrained_model.pb)| CPU，iGPU，MYRIAD，VAD-M
| [resnet-50-tf](http://download.tensorflow.org/models/official/20181001_resnet/savedmodels/resnet_v1_fp32_savedmodel_NHWC_jpg.tar.gz)| CPU，iGPU，MYRIAD，VAD-M
| [resnet-101](http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz)| CPU，iGPU，MYRIAD，VAD-M
| [resnet-152](http://download.tensorflow.org/models/resnet_v1_152_2016_08_28.tar.gz)| CPU，iGPU，MYRIAD，VAD-M
| [resnet\_50\_v1.5](https://zenodo.org/record/2535873/files/resnet50_v1.pb)| CPU，iGPU，MYRIAD，VAD-M
| [resnet\_v2\_101](http://download.tensorflow.org/models/resnet_v2_152_2017_04_14.tar.gz)| CPU，iGPU，MYRIAD
| [rfcn\_resnet101\_coco](https://download.pytorch.org/models/resnet50-19c8e357.pth)| CPU，iGPU，MYRIAD
| [se-resnext-50](https://drive.google.com/drive/folders/1k5MtfqbNRA8ziE3f18vu00Q1FQCzk4__)| CPU，iGPU，MYRIAD，VAD-M
| [squeezenet1.1](https://github.com/forresti/SqueezeNet)| CPU，iGPU，MYRIAD，VAD-M
| [srgan](https://github.com/tensorlayer/srgan)| CPU，iGPU
| [STN](https://github.com/oarriaga/STN.keras)| CPU，iGPU，MYRIAD，VAD-M
| [vehicle-attributes-barrier-0103](https://download.01.org/opencv/openvino_training_extensions/models/vehicle_attributes/vehicle-attributes-barrier-0103.tar.gz)| CPU，iGPU，MYRIAD，VAD-M
| [vehicle-license-plate-detection-barrier-0123](https://download.01.org/opencv/openvino_training_extensions/models/ssd_detector/ssd-mobilenet-v2-0.35.1-barrier-256x256-0123.tar.gz)| CPU，iGPU，MYRIAD，VAD-M
| [vggvox](https://github.com/linhdvu14/vggvox-speaker-identification)| CPU，iGPU，MYRIAD
| [yolo-v1-tiny-tf](https://www.npmjs.com/package/tfjs-yolo)| CPU，iGPU，MYRIAD
| [yolo\_v2](https://github.com/david8862/keras-YOLOv3-model-set.git)| CPU，iGPU，MYRIAD，VAD-M
| [yolo\_v2\_tiny](https://github.com/david8862/keras-YOLOv3-model-set.git)| CPU，iGPU，MYRIAD，VAD-M
| [yolo\_v3\_darknet](https://github.com/mystic123/tensorflow-yolo-v3.git)| CPU，iGPU，MYRIAD，VAD-M
| [yolo-v3](https://download.01.org/opencv/public_models/022020/yolo_v3/yolov3.pb)| CPU，iGPU，MYRIAD，VAD-M
| [yolo-v3-tiny-tf](https://download.01.org/opencv/public_models/082020/yolo-v3-tiny-tf/yolo-v3-tiny-tf.zip)| CPU，iGPU，MYRIAD，VAD-M
| [yolo-v4](https://github.com/david8862/keras-YOLOv3-model-set)| CPU，iGPU，MYRIAD，VAD-M

仅为其中部分模型提供了预训练frozen模型文件。如欲获取其他模型的预训练frozen模型文件，请访问所提供的链接。