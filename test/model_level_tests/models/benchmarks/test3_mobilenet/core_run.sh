cd scripts/tf_cnn_benchmarks/
OMP_NUM_THREADS=28 KMP_AFFINITY=granularity=fine,compact,1,0 \
    python tf_cnn_benchmarks.py --model=mobilenet \
    --num_inter_threads=2 --batch_size=1 --data_format NCHW \
    --num_batches 200 --train_dir=mobilenet_train