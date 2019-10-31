# Bridge examples

Simple! Just add `import ngraph_bridge` after [building] it.

The simplest `hello-world` example can be found in [axpy.py]. For other 
real-world examples, see the instructions below to run `tf_cnn_benchmarks` and 
models from TensorFlow Hub:

# `tf_cnn_benchmarks`: High-performance benchmarks

`tf_cnn_benchmarks` contains implementations of several popular convolutional
models, and is designed to be as fast as possible. `tf_cnn_benchmarks` supports
running either on a single machine or running in *distributed mode* across multiple
hosts. See the [High-Performance Models Guide] for more information.

These models utilize many of the strategies in the [TensorFlow Performance
Guide].

These models are designed for performance. For models that have clean and
easy-to-read implementations, see the [TensorFlow Official Models].

## Running examples from `tf_cnn_benchmarks`

### Use the following instructions

    git clone https://github.com/tensorflow/benchmarks.git
    git checkout 4c7b09ad87bbfc4b1f89650bcee40b3fc5e7dfed
    cd benchmarks/scripts/tf_cnn_benchmarks/

Next, enable nGraph by editing the `convnet_builder.py` and adding 
`import ngraph_bridge` right after the `import tensorflow` line.

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

* Use ```--data_format NCHW``` to get better performance. Avoid ```NHWC``` if 
  possible.
* Change the batch_size to `128` for batch inference performance and `batch_size=1` 
  for real-time inference
* Change the `--model` flag to test for different topologies. The (by no means 
  exhaustive) list of models known to run can be found on the nGraph [validated workloads]
  page

Please feel free to run more models and let us know if you run across any issues.

* A more involved example of the run command

    KMP_BLOCKTIME=0 OMP_NUM_THREADS=28  KMP_AFFINITY=granularity=fine,proclist=[0-27] python tf_cnn_benchmarks.py --model=resnet50  --eval --num_inter_threads=1 --batch_size=128  --train_dir /nfs/fm/disks/aipg_trained_dataset/ngraph_tensorflow/partially_trained/resnet50 --data_format NCHW --num_epochs=1 --data_name=imagenet --data_dir /mnt/data/TF_ImageNet_latest/ --datasets_use_prefetch=False 

# TensorFlow Hub:

TensorFlow Hub models should also work. For example, you can try out [retraining] 
tutorials on, `inceptionv3`.

# Keras models:

Keras (with Tensorflow backend) should also work out-of-the-box with nGraph, 
once one adds ```import ngraph_bridge``` to the script. [Example](https://github.com/tensorflow/ngraph-bridge/blob/master/examples/keras_sample.py)



[building]:https://github.com/tensorflow/ngraph-bridge/blob/master/README.md
[axpy.py]:https://github.com/tensorflow/ngraph-bridge/blob/master/examples/axpy.py
[High-Performance Models Guide]:https://www.tensorflow.org/performance/performance_models 
[TensorFlow Performance Guide]: https://www.tensorflow.org/performance/performance_guide
[TensorFlow Official Models]:https://github.com/tensorflow/models/tree/master/official
[validated workloads]: http://ngraph.nervanasys.com/docs/latest/frameworks/validated/list.html
[retraining]:https://www.tensorflow.org/hub/tutorials/image_retraining