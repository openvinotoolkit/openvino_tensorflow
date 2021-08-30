import os
import time
import cv2
import argparse
import tensorflow as tf
import numpy as np
from PIL import Image
import openvino_tensorflow as ovtf
from common.utils import get_input_mode, load_graph, get_colors, draw_boxes, get_anchors
from common.pre_process import preprocess_image_yolov3
from common.post_process import yolo3_postprocess_np
from common.yolov3 import load_model
# from keras.models import load_model

anchors = get_anchors()
num_anchors = len(anchors)


def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.io.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label


def load_coco_names(file_name):
    names = {}
    assert os.path.exists(file_name), "could not find label file path"
    with open(file_name) as f:
        for coco_id, name in enumerate(f):
            names[coco_id] = name
    return names


if __name__ == "__main__":
    input_file = "examples/data/grace_hopper.jpg"
    model_file = "examples/data/darknet53.h5"
    label_file = "examples/data/coco.names"
    input_height = 416
    input_width = 416
    input_mean = 0
    input_std = 255

    supported_backends = ['CPU', 'GPU', 'MYRIAD', 'VAD-M']
    backend_name = "CPU"
    output_dir = "."
    conf_threshold = 0.6
    iou_threshold = 0.5

    # overlay parameters
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = .6
    color = (0, 0, 0)
    font_thickness = 2

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--graph", help="Optional. Path to graph/model to be executed.")

    parser.add_argument(
        "--labels", help="Optional. Path to labels mapping file.")
    parser.add_argument(
        "--input",
        help=
        "Optional. The input to be processed. Path to an image or video or directory. Use 'cam' or 'camera' for using camera as input."
    )
    parser.add_argument(
        "--input_height",
        type=int,
        help="Optional. Specify input height value.")
    parser.add_argument(
        "--input_width", type=int, help="Optional. Specify input width value.")
    parser.add_argument(
        "--input_mean", type=int, help="Optional. Specify input mean value.")
    parser.add_argument(
        "--input_std", type=int, help="Optional. Specify input std value.")
    parser.add_argument(
        "--backend",
        help="Optional. Specify the target device to infer on; "
        "CPU, GPU, MYRIAD, or VAD-M is acceptable. Default value is CPU.")
    parser.add_argument(
        "--no_show", help="Optional. Don't show output.", action='store_true')
    parser.add_argument(
        "--conf_threshold",
        type=float,
        help="Optional. Specify confidence threshold. Default is 0.6.")
    parser.add_argument(
        "--iou_threshold",
        type=float,
        help="Optional. Specify iou threshold. Default is 0.5.")
    parser.add_argument(
        "--disable_ovtf",
        help="Optional."
        "Disable openvino_tensorflow pass and run on stock TF.",
        action='store_true')
    args = parser.parse_args()
    if args.graph:
        model_file = args.graph
        if args.labels:
            label_file = args.labels
        else:
            label_file = None
    if args.input:
        input_file = args.input
    if args.input_height:
        input_height = args.input_height
    if args.input_width:
        input_width = args.input_width
    if args.input_mean:
        input_mean = args.input_mean
    if args.input_std:
        input_std = args.input_std
    if args.backend:
        backend_name = args.backend
    if args.conf_threshold:
        conf_threshold = args.conf_threshold
    if args.iou_threshold:
        iou_threshold = args.iou_threshold

    # Load the labels
    if label_file:
        classes = load_coco_names(label_file)
        num_classes = len(classes)

    # Load graph and process input image
    yolov3_model = load_model(model_file, num_anchors, num_classes,
                              (input_height, input_width))
    # yolov3_model = load_model(model_file)

    if not args.disable_ovtf:
        # Print list of available backends
        print('Available Backends:')
        backends_list = ovtf.list_backends()
        backends_list = [b for b in backends_list if b in supported_backends]
        for backend in backends_list:
            print(backend)
        ovtf.set_backend(backend_name)
    else:
        ovtf.disable()

    cap = None
    images = []
    if label_file:
        labels = load_coco_names(label_file)
    colors = get_colors(labels)
    input_mode = get_input_mode(input_file)
    if input_mode == "video":
        cap = cv2.VideoCapture(input_file)
    elif input_mode == "camera":
        cap = cv2.VideoCapture(0)
    elif input_mode == 'image':
        images = [input_file]
    elif input_mode == 'directory':
        images = [os.path.join(input_file, i) for i in os.listdir(input_file)]
    else:
        raise Exception("Unable to find the input mode")
    images_len = len(images)
    image_id = -1
    # Initialize session and run

    while True:
        image_id += 1
        if input_mode in ['camera', 'video']:
            if cap.isOpened():
                ret, frame = cap.read()
                if ret is True:
                    pass
                else:
                    break
            else:
                break
        if input_mode in ['image', 'directory']:
            if image_id < images_len:
                frame = cv2.imread(images[image_id])
            else:
                break
        img = frame
        image = Image.fromarray(img)

        #preprocess the input frame
        image_data = preprocess_image_yolov3(image, (input_height, input_width))
        image_shape = tuple((frame.shape[0], frame.shape[1]))

        #running the inference
        print("starting the inference")
        start = time.time()
        results = yolov3_model.predict(image_data)
        elapsed = time.time() - start
        fps = 1 / elapsed
        print('Inference time in ms: %.2f' % (elapsed * 1000))

        #post processing the results
        out_boxes, out_classes, out_scores = yolo3_postprocess_np(
            results,
            image_shape,
            get_anchors(),
            len(labels), (input_height, input_width),
            max_boxes=100,
            elim_grid_sense=True)

        # modified draw_boxes function to return an openCV formatted image
        img_bbox = draw_boxes(img, out_boxes, out_classes, out_scores, labels,
                              colors)
        # draw information overlay onto the frames
        cv2.putText(img_bbox, 'Inference Running on : {0}'.format(backend_name),
                    (30, 50), font, font_size, color, font_thickness)
        cv2.putText(
            img_bbox, 'FPS : {0} | Inference Time : {1}ms'.format(
                int(fps), round((elapsed * 1000), 2)), (30, 80), font,
            font_size, color, font_thickness)
        if not args.no_show:
            cv2.imshow("detections", img_bbox)
            if cv2.waitKey(1) & 0XFF == ord('q'):
                break
        print(image_id)
    if cap:
        cap.release()
    cv2.destroyAllWindows()
