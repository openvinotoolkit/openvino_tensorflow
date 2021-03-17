This document provides a list of all validated models that are supported by Intel<sup>®</sup> OpenVINO™ integration with TensorFlow. This list is continuously evolving as we enable more operators and models. 

**Following models are supported on CPU, GPU and MYRIAD**

## TensorFlow-Slim Image Classification Library

* [Inception V3](http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz)
* [Inception_V4](http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz)
* [Resnet V1 50](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz)
* [Resnet V2 152](http://download.tensorflow.org/models/resnet_v2_152_2017_04_14.tar.gz)
* [Resnet V2 50](http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz)
* [VGG 16](http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz)
* [VGG 19](http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz)
* [MobileNet_v1_1.0_224](http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz)
* [CifarNet](https://github.com/tensorflow/models/blob/master/research/slim/nets/cifarnet.py)
* [LeNet](https://github.com/tensorflow/models/blob/master/research/slim/nets/lenet.py)

The links to the TensorFlow-Slim models include the checkpoint files only. You should refer to TensorFlow-Slim models [instructions page](https://github.com/tensorflow/models/tree/master/research/slim) to run inference or freeze the models. (No pre-trained checkpoint files provided for CifarNet and LeNet.)

## Tensorflow Object Detection Model Zoo
* [faster_rcnn_inception_resnet_v2_atrous_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28.tar.gz)
* [faster_rcnn_inception_v2_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz)
* [faster_rcnn_resnet50_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gzs)
* [faster_rcnn_resnet101_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_2018_01_28.tar.gz)
* [faster_rcnn_resnet50_lowproposals_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_lowproposals_coco_2018_01_28.tar.gz)
* [ssd_inception_v2](http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz)
* [ssd_mobilenet_v1](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz)
* [ssd_mobilenet_v1_fpn](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz)
* [ssd_mobilenet_v2](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz)
* [ssd_resnet_50_fpn](http://download.tensorflow.org/models/object_detection/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz)
* [ssdlite_mobilenet_v2](http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz)
* [mask_rcnn_inception_resnet_v2_atrous_coco](http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28.tar.gz)
* [mask_rcnn_inception_v2_coco](http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_v2_coco_2018_01_28.tar.gz)

## TF Keras Applications
* [DenseNet121](https://www.tensorflow.org/api_docs/python/tf/keras/applications/DenseNet121)
* [DenseNet169](https://www.tensorflow.org/api_docs/python/tf/keras/applications/DenseNet169)
* [DenseNet201](https://www.tensorflow.org/api_docs/python/tf/keras/applications/DenseNet201)
* [EfficientnetB0](https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB0)
* [EfficientnetB1](https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB1)
* [EfficientnetB2](https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB2)
* [EfficientnetB3](https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB3)
* [EfficientnetB4](https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB4)
* [EfficientnetB5](https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB5)
* [EfficientnetB6](https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB6)
* [EfficientnetB7](https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB7)
* [InceptionV3](https://www.tensorflow.org/api_docs/python/tf/keras/applications/InceptionV3)
* [NASNetLarge](https://www.tensorflow.org/api_docs/python/tf/keras/applications/NASNetLarge)
* [NASNetMobile](https://www.tensorflow.org/api_docs/python/tf/keras/applications/NASNetMobile)
* [ResNet50v2](https://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet50V2)

Please follow the instructions on [Keras Applications](https://keras.io/api/applications/) page for further information about using these models.

## Other Models
* [inception_resnet_v2](https://github.com/openvinotoolkit/open_model_zoo/blob/2021.2/models/public/inception-resnet-v2-tf/model.yml)
* [mobilenet_v1_0.25_128](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.25_128.tgz)
* [mobilenet_v1_0.50_160](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.5_160.tgz)
* [mobilenet_v1_0.50_224](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.5_224.tgz)
* [mobilenet_v1_1.0_224](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz)
* [mobilenet-v3-large-1.0-224-tf](https://storage.googleapis.com/mobilenet_v3/checkpoints/v3-large_224_1.0_float.tgz)
* [mobilenet-v3-small-1.0-224-tf](https://storage.googleapis.com/mobilenet_v3/checkpoints/v3-small_224_1.0_float.tgz)
* [PRNet](https://github.com/YadiraF/PRNet)
* [resnet_50](https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/resnet50_fp32_pretrained_model.pb)
* [resnet_50_v1.5](https://zenodo.org/record/2535873/files/resnet50_v1.pb)
* [yolo_v2](https://github.com/david8862/keras-YOLOv3-model-set.git)
* [yolo_v2_tiny](https://github.com/david8862/keras-YOLOv3-model-set.git)
* [yolo_v3_darknet](https://github.com/mystic123/tensorflow-yolo-v3.git)
* [yolo-v3](https://download.01.org/opencv/public_models/022020/yolo_v3/yolov3.pb)
* [yolo-v3-tiny-tf](https://download.01.org/opencv/public_models/082020/yolo-v3-tiny-tf/yolo-v3-tiny-tf.zip)
* [yolo-v4](https://github.com/david8862/keras-YOLOv3-model-set)
* [CRNN](https://github.com/MaybeShewill-CV/CRNN_Tensorflow)
* [densenet161](https://drive.google.com/file/d/0B_fUSpodN0t0NmZvTnZZa2plaHc/view)
* resnext50v2
* squeezenet1.1
