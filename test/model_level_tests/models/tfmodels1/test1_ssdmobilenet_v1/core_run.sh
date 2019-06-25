cd research


export PYTHONPATH=`pwd`:`pwd`/slim
KMP_BLOCKTIME=1  OMP_NUM_THREADS=28 KMP_AFFINITY=granularity=fine,compact,1,0  \
python3 object_detection/model_main.py \
--pipeline_config_path=object_detection/samples/configs/ssd_mobilenet_v1_coco.config \
--model_dir=ssd_mobilenet_v1_train \
--num_train_steps=1 \
--logtostderr \
--sample_1_of_n_eval_examples=1