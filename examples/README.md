# How to use ngraph

Simple! Just add `import ngraph_bridge` after [building](https://github.com/NervanaSystems/ngraph-tf/blob/master/README.md) it

The simplest hello-world example can be found in [```axpy.py```](https://github.com/NervanaSystems/ngraph-tf/blob/master/examples/axpy.py). For real world examples checkout the instructions below to run tf_cnn_benchmarks and models from Tensorflow Hub



# tf_cnn_benchmarks: High performance benchmarks

tf_cnn_benchmarks contains implementations of several popular convolutional
models, and is designed to be as fast as possible. tf_cnn_benchmarks supports
both running on a single machine or running in distributed mode across multiple
hosts. See the [High-Performance models
guide](https://www.tensorflow.org/performance/performance_models) for more
information.

These models utilize many of the strategies in the [TensorFlow Performance
Guide](https://www.tensorflow.org/performance/performance_guide). Benchmark
results can be found [here](https://www.tensorflow.org/performance/benchmarks).

These models are designed for performance. For models that have clean and
easy-to-read implementations, see the [TensorFlow Official
Models](https://github.com/tensorflow/models/tree/master/official).

## Running examples from tf_cnn_benchmarks

### Use the following instructions

    git clone https://github.com/tensorflow/benchmarks.git
    git checkout 4c7b09ad87bbfc4b1f89650bcee40b3fc5e7dfed
    cd benchmarks/scripts/tf_cnn_benchmarks/

Next enable nGraph by editing the `convnet_builder.py` by adding `import ngraph_bridge` right after
the `import tensorflow` line.

### Train for a few iterations:

    KMP_BLOCKTIME=0  OMP_NUM_THREADS=56 KMP_AFFINITY=granularity=fine,compact,1,0 \
        python tf_cnn_benchmarks.py --data_format NCHW \
        --num_inter_threads 2 --train_dir=./modelsavepath/ \
        --model=resnet50 --num_batches 10 --batch_size=128

### Evaluate the model (Inference pass):

    KMP_BLOCKTIME=0  OMP_NUM_THREADS=56 KMP_AFFINITY=granularity=fine,compact,1,0 \
        python tf_cnn_benchmarks.py --data_format NCHW \
        --num_inter_threads 1 --train_dir=$(pwd)/modelsavepath \
        --eval --model=resnet50 --batch_size=128```


### Tips
* Use ```--data_format NCHW``` to get better performance. Avoid ```NHWC``` if possible.
* Change the batch_size to 128 for batch inference performance and batch_size=1 for real time inference
* Change the `--model` flag to test for different topologies. The (by no means non-exhaustive) list of models known to run as of now are:
1. `vgg11`
2. `vgg16`
3. `vgg19`
4. `lenet`
5. `googlenet`
6. `overfeat`
7. `alexnet`
8. `trivial`
9. `inception3`
10. `inception4`
11. `resnet50`
12. `resnet50_v1.5`
13. `resnet50_v2`
14. `resnet101`
15. `resnet101_v2`
16. `resnet152`
17. `resnet152_v2`
18. `mobilenet`


Please feel free to run more models and let us know if you run across any issues.
* A more involved example of the run command
KMP_BLOCKTIME=0 OMP_NUM_THREADS=28  KMP_AFFINITY=granularity=fine,proclist=[0-27] python tf_cnn_benchmarks.py --model=resnet50  --eval --num_inter_threads=1 --batch_size=128  --train_dir /nfs/fm/disks/aipg_trained_dataset/ngraph_tensorflow/partially_trained/resnet50 --data_format NCHW --num_epochs=1 --data_name=imagenet --data_dir /mnt/data/TF_ImageNet_latest/ --datasets_use_prefetch=False 


# Tensorflow Hub:
Tensorflow Hub models should also work. For example, you can try out network retraining by following instructions from [here](https://www.tensorflow.org/hub/tutorials/image_retraining) on, lets say, inceptionv3

# Keras models:
Keras (with Tensorflow backend) too should also work out of the box with ngraph, once one adds ```import ngraph_bridge``` to the script. [Here](https://github.com/NervanaSystems/ngraph-tf/blob/master/examples/keras_sample.py) is an example.