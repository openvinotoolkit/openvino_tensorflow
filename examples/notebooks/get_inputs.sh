mkdir output_images
mkdir inputs
cd inputs
wget "https://github.com/tensorflow/models/raw/master/research/object_detection/test_images/image2.jpg"
wget "https://raw.githubusercontent.com/openvinotoolkit/openvino_tensorflow/master/examples/notebooks/mscoco_label_map.txt"
wget "http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d6_coco17_tpu-32.tar.gz"
tar -zxvf efficientdet_d6_coco17_tpu-32.tar.gz
wget "http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet152_v1_1024x1024_coco17_tpu-8.tar.gz"
tar -zxvf faster_rcnn_resnet152_v1_1024x1024_coco17_tpu-8.tar.gz
wget "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz"
tar -zxvf ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz