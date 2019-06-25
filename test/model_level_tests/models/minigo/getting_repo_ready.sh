pip3 install -r requirements.txt
pip install -U google.cloud
pip install -U google-cloud-storage

#BOOTSTRAP: This command initializes your working directory for the trainer and a random model. 
#This random model is also exported to --model-save-path so that selfplay can immediately 
#start playing with this random model.

if [ -e "outputs/models/" ];then rm -rf "outputs/models/" && mkdir outputs/models; fi 

export MODEL_NAME=000000-bootstrap
BOARD_SIZE=19 python3 bootstrap.py \
  --work_dir=minigo_train \
  --export_path=outputs/models/$MODEL_NAME

#SELF-PLAY This command starts self-playing, outputting its raw game data as tf.
#Examples as well as in SGF form in the directories.

BOARD_SIZE=19 python3 selfplay.py \
  --load_file=outputs/models/$MODEL_NAME \
  --num_readouts 10 \
  --verbose 1 \
  --selfplay_dir=outputs/data/selfplay \
  --holdout_dir=outputs/data/holdout \
  --sgf_dir=outputs/sgf



