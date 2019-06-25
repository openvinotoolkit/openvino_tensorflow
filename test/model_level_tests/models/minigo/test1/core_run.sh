
if [ -e "minigo_train" ];then rm -rf "minigo_train" ; fi 

OMP_NUM_THREADS=28 BOARD_SIZE=19 python3 train.py \
  outputs/data/selfplay/* \
  --work_dir=minigo_train \
  --export_path=outputs/models/000001-first_generation
