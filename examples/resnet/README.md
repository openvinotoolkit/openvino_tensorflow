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

## Getting Started

Benchmark scripts to run CNN based models (inference) on NGRAPH

First run training for a few iterations to store the model checkpoints:

`KMP_BLOCKTIME=0  OMP_NUM_THREADS=56 KMP_AFFINITY=granularity=fine,compact,1,0 python tf_cnn_benchmarks.py --batch_size=128
    --model=resnet50 --num_inter_threads 2 --train_dir=/path/to/save --num_batches 10`

Use this command to get the inference numbers:

`KMP_BLOCKTIME=0  OMP_NUM_THREADS=56 KMP_AFFINITY=granularity=fine,compact,1,0 python tf_cnn_benchmarks.py --batch_size=1 --model=resnet50
    --num_inter_threads 1 --train_dir=/path/to/loadd_model --eval`

Note that `--log_dir` during training and `--train_dir` during  inference must be the same directory.

Change the batch_size to 128 for batch inference performance and batch_size=1 for real time inference

Change the `--model` flag to test for different topologies.

The models supported from the POR list is as follows:

1. `--model=resnet50`
2. `--model=inception3`
3. `--model=inception4`
4. ` --model=mobilenet`


`--select_device` argument can also be used to compare with a reference run.
Supported options are `NGRAPH`, `CPU` and `XLA_CPU`

run command example:
KMP_BLOCKTIME=0 OMP_NUM_THREADS=28  KMP_AFFINITY=granularity=fine,proclist=[0-27] python tf_cnn_benchmarks.py --model=resnet50  --eval --num_inter_threads=1 --batch_size=128  --train_dir /nfs/fm/disks/aipg_trained_dataset/ngraph_tensorflow/partially_trained/resnet50 --data_format NCHW --select_device NGRAPH  --num_epochs=1 --data_name=imagenet --data_dir /mnt/data/TF_ImageNet_latest/ --datasets_use_prefetch=False --select_device NGRAPH
