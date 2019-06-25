export PYTHONPATH="${PYTHONPATH}:`pwd`"

cd official/transformer

# Export variables
PARAM_SET=big
DATA_DIR=/mnt/data/transformer/data
MODEL_DIR=/mnt/data/transformer/checkpoints/model_$PARAM_SET
VOCAB_FILE=$DATA_DIR/vocab.ende.32768

python translate.py --model_dir=$MODEL_DIR --vocab_file=$VOCAB_FILE \
    --param_set=$PARAM_SET --text="hello world"