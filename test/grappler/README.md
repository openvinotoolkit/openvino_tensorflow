  
  ### To use the ngraph-optimizer, you will need to apply the corresponding patch to the model you want to run
  
| Model        | Patch File      | Notes|
|:------------:|:---------------:|:-------:|
| Densenet |  benchmarks_cnn_ng_optimizer.patch|
|Inception-v3|benchmarks_cnn_ng_optimizer.patch|
|Inception-v4| benchmarks_cnn_ng_optimizer.patch|
|Resnet50-v1| benchmarks_cnn_ng_optimizer.patch|
|Resnet50-v2| benchmarks_cnn_ng_optimizer.patch|
|Mobilinet-v2|benchmarks_cnn_ng_optimizer.patch|
|Vgg16| benchmarks_cnn_ng_optimizer.patch|
|a3c| a3c_ng_optimizer.patch|
|ssd-mobilenet-v1 |infer_detections_ng_optimizer.patch|
|fasterRCNN |infer_detections_ng_optimizer.patch|
|mask_rcnn |infer_detections_ng_optimizer.patch|
|RFCN |infer_detections_ng_optimizer.patch|
|DCGAN |dcgan_inference_bench_ng_optimizer.patch|
|DRAW |draw_inf_ng_optimizer.patch|
|Inception-resnet-v2 |eval_image_classifier_ng_optimizer.patch|
|mobilenet-v1 |eval_image_classifier_ng_optimizer.patch|
|ssd-vgg16 |inference_model_ng_optimizer.patch|
|UNET |unet_ng_optimizer.patch|
|Wavenet |fastgen_ng_optimizer.patch|
|YOLO |build_ng_optimizer.patch|
|NCF |ncf_main_ng_optimizer.patch|
|Wide and Deep |wide_deep_ng_optimizer.patch|
|GNMT|misc_utils_ng_optimizer.patch| 
|Squeezenet| train_squeezenet_ng_optimizer.patch| Does not run currently - Gives InvalidArgumentError: Default MaxPoolingOp only supports NHWC by default|
|Transformer - LT| misc_utils_ng_optimizer.patch| Does not run currently - Segmentation Fault |
