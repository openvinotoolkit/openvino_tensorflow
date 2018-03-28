This version of ResNet includes slightly modified scripts from the `benchmarks` branch on the [`tensorflow/models`](https://github.com/tensorflow/models) repo.
The original model can be found at [`tensorflow/models/official/resnet`](https://github.com/tensorflow/models/tree/master/official/resnet).

We have made the following modifications:
- The model was placed on NGRAPH using `with tf.device`
- Training accuracy computation was placed on CPU using `with tf.device`
- `tf.train.piecewise_constant` function was placed on CPU
- Replaced 
```
 with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(loss, global_step)
```
with
`train_op = tf.group(optimizer.minimize(loss, global_step), update_ops)`
in `cifar10_main.py` and `imagenet_main.py`

# Training

To train ResNet with CIFAR10, run the following command for best performance (replace `56` with the number of physical CPU cores on your system):

`OMP_NUM_THREADS=56 KMP_AFFINITY='granularity=fine,scatter' python cifar10_main.py 
--data_dir /path/to/dataset --model_dir /path/to/models --batch_size 128 --resnet_size 20 --select_device NGRAPH`

Use `--select_device` to choose the device to execute on. The default is `NGRAPH`. Other options include `XLA_CPU` and `CPU`.


To train ResNet with the ImageNet-1K dataset, run the following command for best performance (replace `56` with the number of physical CPU cores on your system):

`OMP_NUM_THREADS=56 KMP_AFFINITY='granularity=fine,scatter' python imagenet_main.py 
--data_dir /path/to/dataset --model_dir /path/to/models --train_batch_size 128 --resnet_size 50 --select_device NGRAPH`

Here, the `--data_dir` flag specifies the path to the pre-processed dataset files. It is assumed that the dataset has already been pre-processed according to the instructions in the `tensorflow/models` repo.
