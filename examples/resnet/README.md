This version of Resnet includes slightly modified scripts from the `benchmarks` branch on the `tensorflow/models` repo. 
The original model can be found at `tensorflow/models/oficial/resnet`


It has the following modifications at a high level:
- The model was placed on NGRAPH using `with tf.device`
- Train accuracy computation was placed on CPU
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

To train Resnet with CIFAR10, please run the following command for best performance:

`OMP_NUM_THREADS=56 KMP_AFFINITY='granularity=fine,scatter' python cifar10_main.py 
--data_dir /path/to/dataset --model_dir /path/to/models --batch_size 128 --resnet_size 20 --select_device NGRAPH`

Use `--select_device` to choose the device to execute on. Defaults to `NGRAPH`,  other options include `XLA_CPU` and `CPU` 



To train Resnet with I1k, please run the following command for best performance:

`OMP_NUM_THREADS=56 KMP_AFFINITY='granularity=fine,scatter' python imagenet_main.py 
--data_dir /path/to/dataset --model_dir /path/to/models --train_batch_size 128 --resnet_size 50 --select_device NGRAPH`

`--data_dir`  is the path to the pre-processed files. The assumption here is that the dataset was already pre-processed following instructions in the `tensorflow/models` repo.
