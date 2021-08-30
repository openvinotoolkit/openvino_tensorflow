from functools import wraps, reduce
import os
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Concatenate, LeakyReLU, UpSampling2D
from tensorflow.keras.layers import Add, ZeroPadding2D, Input, BatchNormalization
from tensorflow.keras.regularizers import l2


def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


def make_spp_last_layers(x,
                         num_filters,
                         out_filters,
                         predict_filters=None,
                         predict_id='1'):
    '''6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer'''
    x = compose(
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
        DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)))(x)

    x = Spp_Conv2D_BN_Leaky(x, num_filters)

    x = compose(
        DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)))(x)

    if predict_filters is None:
        predict_filters = num_filters * 2
    y = compose(
        DarknetConv2D_BN_Leaky(predict_filters, (3, 3)),
        DarknetConv2D(out_filters, (1, 1),
                      name='predict_conv_' + predict_id))(x)
    return x, y


def make_last_layers(x,
                     num_filters,
                     out_filters,
                     predict_filters=None,
                     predict_id='1'):
    '''6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer'''
    x = compose(
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
        DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
        DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)))(x)

    if predict_filters is None:
        predict_filters = num_filters * 2
    y = compose(
        DarknetConv2D_BN_Leaky(predict_filters, (3, 3)),
        DarknetConv2D(out_filters, (1, 1),
                      name='predict_conv_' + predict_id))(x)
    return x, y


def yolo3_predictions(feature_maps,
                      feature_channel_nums,
                      num_anchors,
                      num_classes,
                      use_spp=False):
    f1, f2, f3 = feature_maps
    f1_channel_num, f2_channel_num, f3_channel_num = feature_channel_nums

    #feature map 1 head & output (13x13 for 416 input)
    if use_spp:
        x, y1 = make_spp_last_layers(
            f1,
            f1_channel_num // 2,
            num_anchors * (num_classes + 5),
            predict_id='1')
    else:
        x, y1 = make_last_layers(
            f1,
            f1_channel_num // 2,
            num_anchors * (num_classes + 5),
            predict_id='1')

    #upsample fpn merge for feature map 1 & 2
    x = compose(
        DarknetConv2D_BN_Leaky(f2_channel_num // 2, (1, 1)), UpSampling2D(2))(x)
    x = Concatenate()([x, f2])

    #feature map 2 head & output (26x26 for 416 input)
    x, y2 = make_last_layers(
        x, f2_channel_num // 2, num_anchors * (num_classes + 5), predict_id='2')

    #upsample fpn merge for feature map 2 & 3
    x = compose(
        DarknetConv2D_BN_Leaky(f3_channel_num // 2, (1, 1)), UpSampling2D(2))(x)
    x = Concatenate()([x, f3])

    #feature map 3 head & output (52x52 for 416 input)
    x, y3 = make_last_layers(
        x, f3_channel_num // 2, num_anchors * (num_classes + 5), predict_id='3')

    return y1, y2, y3


@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Conv2D."""
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides') == (
        2, 2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)


def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs), BatchNormalization(),
        LeakyReLU(alpha=0.1))


def resblock_body(x, num_filters, num_blocks):
    '''A series of resblocks starting with a downsampling Convolution2D'''
    # Darknet uses left and top padding instead of 'same' mode
    x = ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3, 3), strides=(2, 2))(x)
    for i in range(num_blocks):
        y = compose(
            DarknetConv2D_BN_Leaky(num_filters // 2, (1, 1)),
            DarknetConv2D_BN_Leaky(num_filters, (3, 3)))(x)
        x = Add()([x, y])
    return x


def darknet53_body(x):
    '''Darknet53 body having 52 Convolution2D layers'''
    x = DarknetConv2D_BN_Leaky(32, (3, 3))(x)
    x = resblock_body(x, 64, 1)
    x = resblock_body(x, 128, 2)
    x = resblock_body(x, 256, 8)
    x = resblock_body(x, 512, 8)
    x = resblock_body(x, 1024, 4)
    return x


def yolo3_body(inputs, num_anchors, num_classes, weights_path=None):
    """Create YOLO_V3 model CNN body in Keras."""
    darknet = Model(inputs, darknet53_body(inputs))
    if weights_path is not None:
        darknet.load_weights(weights_path, by_name=True)
        print('Load weights {}.'.format(weights_path))
    # f1: 13 x 13 x 1024
    f1 = darknet.output
    # f2: 26 x 26 x 512
    f2 = darknet.layers[152].output
    # f3: 52 x 52 x 256
    f3 = darknet.layers[92].output
    f1_channel_num = 1024
    f2_channel_num = 512
    f3_channel_num = 256

    y1, y2, y3 = yolo3_predictions(
        (f1, f2, f3), (f1_channel_num, f2_channel_num, f3_channel_num),
        num_anchors, num_classes)

    return Model(inputs, [y1, y2, y3])


def get_yolo3_model(weights_path,
                    model_type,
                    num_feature_layers,
                    num_anchors,
                    num_classes,
                    input_tensor=None,
                    input_shape=None,
                    model_pruning=False,
                    pruning_end_step=10000):
    #prepare input tensor
    yolo3_model_map = {'yolo3_darknet': [yolo3_body, 185, weights_path]}
    if input_shape:
        input_tensor = Input(shape=input_shape, name='image_input')

    if input_tensor is None:
        input_tensor = Input(shape=(None, None, 3), name='image_input')

    if num_feature_layers == 3:
        if model_type in yolo3_model_map:
            model_function = yolo3_model_map[model_type][0]
            backbone_len = yolo3_model_map[model_type][1]
            weights_path = yolo3_model_map[model_type][2]

            if weights_path:
                model_body = model_function(
                    input_tensor,
                    num_anchors // 3,
                    num_classes,
                    weights_path=weights_path)
            else:
                model_body = model_function(input_tensor, num_anchors // 3,
                                            num_classes)
        else:
            raise ValueError('This model type is not supported now')
    else:
        raise ValueError('model type mismatch anchors')

    if model_pruning:
        model_body = get_pruning_model(
            model_body, begin_step=0, end_step=pruning_end_step)

    return model_body, backbone_len


def load_model(weights_path, num_anchors, num_classes, image_size):
    '''to generate the bounding boxes'''
    weights_path = os.path.expanduser(weights_path)
    assert weights_path.endswith(
        '.h5'), 'Keras model or weights must be a .h5 file.'

    # Load model, or construct model and load weights.

    #YOLOv3 model has 9 anchors and 3 feature layers
    #so we can calculate feature layers number to get model type
    num_feature_layers = num_anchors // 3

    try:
        yolo_model, _ = get_yolo3_model(
            weights_path,
            'yolo3_darknet',
            num_feature_layers,
            num_anchors,
            num_classes,
            input_shape=image_size + (3,),
            model_pruning=False)
        print("model loaded")
        yolo_model.load_weights(
            weights_path)  # make sure model, anchors and classes match
        # yolo_model.summary()
    except Exception as e:
        print(repr(e))
        assert yolo_model.layers[-1].output_shape[-1] == \
            num_anchors/len(yolo_model.output) * (num_classes + 5), \
            'Mismatch between model and given anchor and class sizes'
    print('{} model, anchors, and classes loaded.'.format(weights_path))
    return yolo_model
