export BERT_BASE_DIR=/nfs/site/disks/aipg_trained_dataset/ngraph_tensorflow/fully_trained/bert/multi_cased_L-12_H-768_A-12
export GLUE_DIR=/mnt/data/bert_tf_records/

OMP_NUM_THREADS=28 \
python run_pretraining.py \
  --input_file=/mnt/data/bert_tf_records/bert_data.tfrecord* \
  --output_dir=/tmp/pretraining_output \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --train_batch_size=4 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=20 \
  --num_warmup_steps=10 \
  --learning_rate=2e-5