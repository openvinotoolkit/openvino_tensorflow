cd scripts/tf_cnn_benchmarks/
OMP_NUM_THREADS=28 KMP_AFFINITY=granularity=fine,compact,1,0 \
    python tf_cnn_benchmarks.py --model=inception4 \
    --print_training_accuracy=True --num_inter_threads=2 --batch_size=16 --data_format NCHW \
    --train_dir=inception4_train  --num_batches 10