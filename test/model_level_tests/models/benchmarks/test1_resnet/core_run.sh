cd scripts/tf_cnn_benchmarks/
KMP_BLOCKTIME=1  OMP_NUM_THREADS=56 KMP_AFFINITY=granularity=fine,compact,1,0 python tf_cnn_benchmarks.py \
--model=resnet50 --num_inter_threads=2 --batch_size=32 --data_format NCHW --data_name=imagenet --datasets_use_prefetch=False \
--print_training_accuracy=True --num_batches=10 --display_every=1 --init_learning_rate=0.001
