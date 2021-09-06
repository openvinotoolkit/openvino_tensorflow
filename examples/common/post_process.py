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

# ==============================================================================
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ==============================================================================
# Modified from the repository (https://github.com/david8862/keras-YOLOv3-model-set):
# https://github.com/david8862/keras-YOLOv3-model-set/blob/master/common/wbf_postprocess.py
# https://github.com/david8862/keras-YOLOv3-model-set/blob/master/common/yolo_postprocess_np.py

import numpy as np
import copy
import warnings
from scipy.special import expit, softmax


def bb_intersection_over_union(A, B) -> float:
    xA = max(A[0], B[0])
    yA = max(A[1], B[1])
    xB = min(A[2], B[2])
    yB = min(A[3], B[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    if interArea == 0:
        return 0.0

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (A[2] - A[0]) * (A[3] - A[1])
    boxBArea = (B[2] - B[0]) * (B[3] - B[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def prefilter_boxes(boxes, scores, labels, image_shape, weights, thr):
    # Create dict with boxes stored by its label
    new_boxes = dict()
    height, width = image_shape

    for t in range(len(boxes)):

        if len(boxes[t]) != len(scores[t]):
            print(
                'Error. Length of boxes arrays not equal to length of scores array: {} != {}'
                .format(len(boxes[t]), len(scores[t])))
            exit()

        if len(boxes[t]) != len(labels[t]):
            print(
                'Error. Length of boxes arrays not equal to length of labels array: {} != {}'
                .format(len(boxes[t]), len(labels[t])))
            exit()

        for j in range(len(boxes[t])):
            score = scores[t][j]
            if score < thr:
                continue
            label = int(labels[t][j])
            box_part = boxes[t][j]

            # input boxes in raw (x,y,w,h) format,
            # convert to normalized (x1,y1,x2,y2)
            x1 = max(0, float(box_part[0]) / width)
            y1 = max(0, float(box_part[1]) / height)
            x2 = min(1, float(box_part[2] + box_part[0]) / width)
            y2 = min(1, float(box_part[3] + box_part[1]) / height)

            if (x2 - x1) * (y2 - y1) == 0.0:
                warnings.warn("Zero area box skipped: {}.".format(box_part))
                continue

            b = [int(label), float(score) * weights[t], x1, y1, x2, y2]
            if label not in new_boxes:
                new_boxes[label] = []
            new_boxes[label].append(b)

    # Sort each list in dict by score and transform it to numpy array
    for k in new_boxes:
        current_boxes = np.array(new_boxes[k])
        new_boxes[k] = current_boxes[current_boxes[:, 1].argsort()[::-1]]

    return new_boxes


def get_weighted_box(boxes, conf_type='avg'):
    """
    Create weighted box for set of boxes
    :param boxes: set of boxes to fuse
    :param conf_type: type of confidence one of 'avg' or 'max'
    :return: weighted box
    """

    box = np.zeros(6, dtype=np.float32)
    conf = 0
    conf_list = []
    for b in boxes:
        box[2:] += (b[1] * b[2:])
        conf += b[1]
        conf_list.append(b[1])
    box[0] = boxes[0][0]
    if conf_type == 'avg':
        box[1] = conf / len(boxes)
    elif conf_type == 'max':
        box[1] = np.array(conf_list).max()
    box[2:] /= conf
    return box


def find_matching_box(boxes_list, new_box, match_iou):
    best_iou = match_iou
    best_index = -1
    for i in range(len(boxes_list)):
        box = boxes_list[i]
        if box[0] != new_box[0]:
            continue
        iou = bb_intersection_over_union(box[2:], new_box[2:])
        if iou > best_iou:
            best_index = i
            best_iou = iou

    return best_index, best_iou


def weighted_boxes_fusion(boxes_list,
                          labels_list,
                          scores_list,
                          image_shape,
                          weights=None,
                          iou_thr=0.55,
                          skip_box_thr=0.0,
                          conf_type='avg',
                          allows_overflow=False):
    '''
    :param boxes_list: list of boxes predictions from each model, each box is 4 numbers.
    It has 3 dimensions (models_number, model_preds, 4)
    Order of boxes: x1, y1, x2, y2. We expect float normalized coordinates [0; 1]
    :param scores_list: list of scores for each model
    :param labels_list: list of labels for each model
    :param weights: list of weights for each model. Default: None, which means weight == 1 for each model
    :param iou_thr: IoU value for boxes to be a match
    :param skip_box_thr: exclude boxes with score lower than this variable
    :param conf_type: how to calculate confidence in weighted boxes. 'avg': average value, 'max': maximum value
    :param allows_overflow: false if we want confidence score not exceed 1.0

    :return: boxes: boxes coordinates (Order of boxes: x1, y1, x2, y2).
    :return: scores: confidence scores
    :return: labels: boxes labels
    '''

    if weights is None:
        weights = np.ones(len(boxes_list))
    if len(weights) != len(boxes_list):
        print(
            'Warning: incorrect number of weights {}. Must be: {}. Set weights equal to 1.'
            .format(len(weights), len(boxes_list)))
        weights = np.ones(len(boxes_list))
    weights = np.array(weights)

    if conf_type not in ['avg', 'max']:
        print('Unknown conf_type: {}. Must be "avg" or "max"'.format(conf_type))
        exit()

    image_shape = np.array(image_shape, dtype='float32')
    filtered_boxes = prefilter_boxes(boxes_list, scores_list, labels_list,
                                     image_shape, weights, skip_box_thr)
    if len(filtered_boxes) == 0:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,))

    overall_boxes = []
    for label in filtered_boxes:
        boxes = filtered_boxes[label]
        new_boxes = []
        weighted_boxes = []

        # Clusterize boxes
        for j in range(0, len(boxes)):
            index, best_iou = find_matching_box(weighted_boxes, boxes[j],
                                                iou_thr)
            if index != -1:
                new_boxes[index].append(boxes[j])
                weighted_boxes[index] = get_weighted_box(
                    new_boxes[index], conf_type)
            else:
                new_boxes.append([boxes[j].copy()])
                weighted_boxes.append(boxes[j].copy())

        # Rescale confidence based on number of models and boxes
        for i in range(len(new_boxes)):
            if not allows_overflow:
                weighted_boxes[i][1] = weighted_boxes[i][1] * min(
                    weights.sum(), len(new_boxes[i])) / weights.sum()
            else:
                weighted_boxes[i][1] = weighted_boxes[i][1] * len(
                    new_boxes[i]) / weights.sum()
        overall_boxes.append(np.array(weighted_boxes))

    overall_boxes = np.concatenate(overall_boxes, axis=0)
    overall_boxes = overall_boxes[overall_boxes[:, 1].argsort()[::-1]]
    boxes = overall_boxes[:, 2:]
    scores = overall_boxes[:, 1]
    labels = overall_boxes[:, 0]

    # convert boxes back to (x,y,w,h)
    boxes[..., 2:] = boxes[..., 2:] - boxes[..., :2]
    # Scale boxes back to original image shape.
    image_wh = image_shape[..., ::-1]
    boxes[..., :2] *= image_wh
    boxes[..., 2:] *= image_wh

    return [boxes], [labels], [scores]


def yolo_decode(prediction,
                anchors,
                num_classes,
                input_dims,
                scale_x_y=None,
                use_softmax=False):
    '''Decode final layer features to bounding box parameters.'''
    batch_size = np.shape(prediction)[0]
    num_anchors = len(anchors)

    grid_size = np.shape(prediction)[1:3]
    #check if stride on height & width are same
    assert input_dims[0] // grid_size[0] == input_dims[1] // grid_size[
        1], 'model stride mismatch.'
    stride = input_dims[0] // grid_size[0]

    prediction = np.reshape(
        prediction, (batch_size, grid_size[0] * grid_size[1] * num_anchors,
                     num_classes + 5))

    ################################
    # generate x_y_offset grid map
    grid_y = np.arange(grid_size[0])
    grid_x = np.arange(grid_size[1])
    x_offset, y_offset = np.meshgrid(grid_x, grid_y)

    x_offset = np.reshape(x_offset, (-1, 1))
    y_offset = np.reshape(y_offset, (-1, 1))

    x_y_offset = np.concatenate((x_offset, y_offset), axis=1)
    x_y_offset = np.tile(x_y_offset, (1, num_anchors))
    x_y_offset = np.reshape(x_y_offset, (-1, 2))
    x_y_offset = np.expand_dims(x_y_offset, 0)

    ################################

    # Log space transform of the height and width
    anchors = np.tile(anchors, (grid_size[0] * grid_size[1], 1))
    anchors = np.expand_dims(anchors, 0)

    if scale_x_y:
        box_xy_tmp = expit(
            prediction[..., :2]) * scale_x_y - (scale_x_y - 1) / 2
        box_xy = (box_xy_tmp + x_y_offset) / np.array(grid_size)[::-1]
    else:
        box_xy = (
            expit(prediction[..., :2]) + x_y_offset) / np.array(grid_size)[::-1]
    box_wh = (
        np.exp(prediction[..., 2:4]) * anchors) / np.array(input_dims)[::-1]

    # Sigmoid objectness scores
    objectness = expit(prediction[..., 4])  # p_o (objectness score)
    objectness = np.expand_dims(
        objectness, -1)  # To make the same number of values for axis 0 and 1

    if use_softmax:
        # Softmax class scores
        class_scores = softmax(prediction[..., 5:], axis=-1)
    else:
        # Sigmoid class scores
        class_scores = expit(prediction[..., 5:])

    return np.concatenate([box_xy, box_wh, objectness, class_scores], axis=2)


def yolo_correct_boxes(predictions, img_shape, model_image_size):
    '''rescale predicition boxes back to original image shape'''
    box_xy = predictions[..., :2]
    box_wh = predictions[..., 2:4]
    objectness = np.expand_dims(predictions[..., 4], -1)
    class_scores = predictions[..., 5:]

    # model_image_size & image_shape should be (height, width) format
    model_image_size = np.array(model_image_size, dtype='float32')
    image_shape = np.array(img_shape, dtype='float32')
    height, width = image_shape

    new_shape = np.round(image_shape * np.min(model_image_size / image_shape))
    offset = (model_image_size - new_shape) / 2. / model_image_size
    scale = model_image_size / new_shape
    # reverse offset/scale to match (w,h) order
    offset = offset[..., ::-1]
    scale = scale[..., ::-1]

    box_xy = (box_xy - offset) * scale
    box_wh *= scale

    # Convert centoids to top left coordinates
    box_xy -= box_wh / 2

    # Scale boxes back to original image shape.
    image_wh = image_shape[..., ::-1]
    box_xy *= image_wh
    box_wh *= image_wh

    return np.concatenate([box_xy, box_wh, objectness, class_scores], axis=2)


def yolo_handle_predictions(predictions,
                            image_shape,
                            max_boxes=100,
                            confidence=0.1,
                            iou_threshold=0.4,
                            use_wbf=False):
    boxes = predictions[:, :, :4]
    box_confidences = np.expand_dims(predictions[:, :, 4], -1)
    box_class_probs = predictions[:, :, 5:]

    box_scores = box_confidences * box_class_probs
    box_classes = np.argmax(box_scores, axis=-1)
    box_class_scores = np.max(box_scores, axis=-1)
    pos = np.where(box_class_scores >= confidence)

    boxes = boxes[pos]
    classes = box_classes[pos]
    scores = box_class_scores[pos]

    if use_wbf:
        # use Weighted-Boxes-Fusion for boxes postprocess
        n_boxes, n_classes, n_scores = weighted_boxes_fusion(
            [boxes], [classes], [scores],
            image_shape,
            weights=None,
            iou_thr=iou_threshold)
    else:
        # Boxes, Classes and Scores returned from NMS
        n_boxes, n_classes, n_scores = nms_boxes(
            boxes,
            classes,
            scores,
            iou_threshold,
            confidence=confidence,
            use_diou=True,
            is_soft=False)

    if n_boxes:
        boxes = np.concatenate(n_boxes)
        classes = np.concatenate(n_classes).astype('int32')
        scores = np.concatenate(n_scores)
        boxes, classes, scores = filter_boxes(boxes, classes, scores, max_boxes)

        return boxes, classes, scores

    else:
        return [], [], []


def filter_boxes(boxes, classes, scores, max_boxes):
    '''
    Sort the prediction boxes according to score
    and only pick top "max_boxes" ones
    '''
    # sort result according to scores
    sorted_indices = np.argsort(scores)
    sorted_indices = sorted_indices[::-1]
    nboxes = boxes[sorted_indices]
    nclasses = classes[sorted_indices]
    nscores = scores[sorted_indices]

    # only pick max_boxes
    nboxes = nboxes[:max_boxes]
    nclasses = nclasses[:max_boxes]
    nscores = nscores[:max_boxes]

    return nboxes, nclasses, nscores


def box_iou(boxes):
    """
    Calculate iou on box array

    Parameters
    ----------
    boxes: bbox numpy array, shape=(N, 4), xywh
           x,y are top left coordinates

    Returns
    -------
    iou: numpy array, shape=(N-1,)
         IoU value of boxes[:-1] with boxes[-1]
    """
    # get box coordinate and area
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]
    areas = w * h

    # check IoU
    inter_xmin = np.maximum(x[:-1], x[-1])
    inter_ymin = np.maximum(y[:-1], y[-1])
    inter_xmax = np.minimum(x[:-1] + w[:-1], x[-1] + w[-1])
    inter_ymax = np.minimum(y[:-1] + h[:-1], y[-1] + h[-1])

    inter_w = np.maximum(0.0, inter_xmax - inter_xmin + 1)
    inter_h = np.maximum(0.0, inter_ymax - inter_ymin + 1)

    inter = inter_w * inter_h
    iou = inter / (areas[:-1] + areas[-1] - inter)
    return iou


def box_diou(boxes):
    """
    Calculate diou on box array
    Reference Paper:
        "Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression"
        https://arxiv.org/abs/1911.08287

    Parameters
    ----------
    boxes: bbox numpy array, shape=(N, 4), xywh
           x,y are top left coordinates

    Returns
    -------
    diou: numpy array, shape=(N-1,)
         IoU value of boxes[:-1] with boxes[-1]
    """
    # get box coordinate and area
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]
    areas = w * h

    # check IoU
    inter_xmin = np.maximum(x[:-1], x[-1])
    inter_ymin = np.maximum(y[:-1], y[-1])
    inter_xmax = np.minimum(x[:-1] + w[:-1], x[-1] + w[-1])
    inter_ymax = np.minimum(y[:-1] + h[:-1], y[-1] + h[-1])

    inter_w = np.maximum(0.0, inter_xmax - inter_xmin + 1)
    inter_h = np.maximum(0.0, inter_ymax - inter_ymin + 1)

    inter = inter_w * inter_h
    iou = inter / (areas[:-1] + areas[-1] - inter)

    # box center distance
    x_center = x + w / 2
    y_center = y + h / 2
    center_distance = np.power(x_center[:-1] - x_center[-1], 2) + np.power(
        y_center[:-1] - y_center[-1], 2)

    # get enclosed area
    enclose_xmin = np.minimum(x[:-1], x[-1])
    enclose_ymin = np.minimum(y[:-1], y[-1])
    enclose_xmax = np.maximum(x[:-1] + w[:-1], x[-1] + w[-1])
    enclose_ymax = np.maximum(x[:-1] + w[:-1], x[-1] + w[-1])
    enclose_w = np.maximum(0.0, enclose_xmax - enclose_xmin + 1)
    enclose_h = np.maximum(0.0, enclose_ymax - enclose_ymin + 1)
    # get enclosed diagonal distance
    enclose_diagonal = np.power(enclose_w, 2) + np.power(enclose_h, 2)
    # calculate DIoU, add epsilon in denominator to avoid dividing by 0
    diou = iou - 1.0 * (center_distance) / (
        enclose_diagonal + np.finfo(float).eps)

    return diou


def nms_boxes(boxes,
              classes,
              scores,
              iou_threshold,
              confidence=0.1,
              use_diou=True,
              is_soft=False,
              use_exp=False,
              sigma=0.5):
    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        # handle data for one class
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]

        # make a data copy to avoid breaking
        # during nms operation
        b_nms = copy.deepcopy(b)
        c_nms = copy.deepcopy(c)
        s_nms = copy.deepcopy(s)

        while len(s_nms) > 0:
            # pick the max box and store, here
            # we also use copy to persist result
            i = np.argmax(s_nms, axis=-1)
            nboxes.append(copy.deepcopy(b_nms[i]))
            nclasses.append(copy.deepcopy(c_nms[i]))
            nscores.append(copy.deepcopy(s_nms[i]))

            # swap the max line and last line
            b_nms[[i, -1], :] = b_nms[[-1, i], :]
            c_nms[[i, -1]] = c_nms[[-1, i]]
            s_nms[[i, -1]] = s_nms[[-1, i]]

            if use_diou:
                iou = box_diou(b_nms)
            else:
                iou = box_iou(b_nms)

            # drop the last line since it has been record
            b_nms = b_nms[:-1]
            c_nms = c_nms[:-1]
            s_nms = s_nms[:-1]

            if is_soft:
                # Soft-NMS
                if use_exp:
                    # score refresh formula:
                    # score = score * exp(-(iou^2)/sigma)
                    s_nms = s_nms * np.exp(-(iou * iou) / sigma)
                else:
                    # score refresh formula:
                    # score = score * (1 - iou) if iou > threshold
                    depress_mask = np.where(iou > iou_threshold)[0]
                    s_nms[depress_mask] = s_nms[depress_mask] * (
                        1 - iou[depress_mask])
                keep_mask = np.where(s_nms >= confidence)[0]
            else:
                # normal Hard-NMS
                keep_mask = np.where(iou <= iou_threshold)[0]

            # keep needed box for next loop
            b_nms = b_nms[keep_mask]
            c_nms = c_nms[keep_mask]
            s_nms = s_nms[keep_mask]

    # reformat result for output
    nboxes = [np.array(nboxes)]
    nclasses = [np.array(nclasses)]
    nscores = [np.array(nscores)]
    return nboxes, nclasses, nscores


def yolo_adjust_boxes(boxes, img_shape):
    '''
    change box format from (x,y,w,h) top left coordinate to
    (xmin,ymin,xmax,ymax) format
    '''
    if boxes is None or len(boxes) == 0:
        return []

    image_shape = np.array(img_shape, dtype='float32')
    height, width = image_shape

    adjusted_boxes = []
    for box in boxes:
        x, y, w, h = box

        xmin = x
        ymin = y
        xmax = x + w
        ymax = y + h

        ymin = max(0, np.floor(ymin + 0.5).astype('int32'))
        xmin = max(0, np.floor(xmin + 0.5).astype('int32'))
        ymax = min(height, np.floor(ymax + 0.5).astype('int32'))
        xmax = min(width, np.floor(xmax + 0.5).astype('int32'))
        adjusted_boxes.append([xmin, ymin, xmax, ymax])

    return np.array(adjusted_boxes, dtype=np.int32)


def yolo3_decode(predictions,
                 anchors,
                 num_classes,
                 input_dims,
                 elim_grid_sense=False):
    """
    YOLOv3 Head to process predictions from YOLOv3 models

    :param num_classes: Total number of classes
    :param anchors: YOLO style anchor list for bounding box assignment
    :param input_dims: Input dimensions of the image
    :param predictions: A list of three tensors with shape (N, 19, 19, 255), (N, 38, 38, 255) and (N, 76, 76, 255)
    :return: A tensor with the shape (N, num_boxes, 85)
    """
    assert len(predictions) == len(
        anchors) // 3, 'anchor numbers does not match prediction.'

    if len(predictions) == 3:  # assume 3 set of predictions is YOLOv3
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        scale_x_y = [1.05, 1.1, 1.2] if elim_grid_sense else [None, None, None]
    elif len(predictions) == 2:  # 2 set of predictions is YOLOv3-tiny
        anchor_mask = [[3, 4, 5], [0, 1, 2]]
        scale_x_y = [1.05, 1.05] if elim_grid_sense else [None, None]
    else:
        raise ValueError('Unsupported prediction length: {}'.format(
            len(predictions)))

    results = []
    for i, prediction in enumerate(predictions):
        results.append(
            yolo_decode(
                prediction,
                anchors[anchor_mask[i]],
                num_classes,
                input_dims,
                scale_x_y=scale_x_y[i],
                use_softmax=False))

    return np.concatenate(results, axis=1)


def yolo3_postprocess_np(yolo_outputs,
                         image_shape,
                         anchors,
                         num_classes,
                         model_image_size,
                         max_boxes=100,
                         confidence=0.1,
                         iou_threshold=0.4,
                         elim_grid_sense=False):
    predictions = yolo3_decode(
        yolo_outputs,
        anchors,
        num_classes,
        input_dims=model_image_size,
        elim_grid_sense=elim_grid_sense)
    predictions = yolo_correct_boxes(predictions, image_shape, model_image_size)

    boxes, classes, scores = yolo_handle_predictions(
        predictions,
        image_shape,
        max_boxes=max_boxes,
        confidence=confidence,
        iou_threshold=iou_threshold)

    boxes = yolo_adjust_boxes(boxes, image_shape)

    return boxes, classes, scores
