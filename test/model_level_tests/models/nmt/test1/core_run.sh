
if [ -e "/tmp/mnt/nmt_model" ];then rm -rf "/tmp/mnt/nmt_model" ; fi  

OMP_NUM_THREADS=28 KMP_AFFINITY=granularity=fine,compact,1,0 \
python -m nmt.nmt --src=vi --tgt=en --vocab_prefix=/dataset/nmt_data/vocab \
  --train_prefix=/dataset/nmt_data/train --dev_prefix=/dataset/nmt_data/tst2012 \
  --test_prefix=/dataset/nmt_data/tst2013 --out_dir=/tmp/mnt/nmt_model \
  --num_train_steps=1  --steps_per_stats=100 --num_layers=1 \
  --dropout=0.2 --metrics=bleu
