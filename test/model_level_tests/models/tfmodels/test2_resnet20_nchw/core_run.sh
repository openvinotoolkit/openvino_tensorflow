export PYTHONPATH="${PYTHONPATH}:`pwd`"

cd official/resnet
OMP_NUM_THREADS=28 KMP_AFFINITY=granularity=fine,compact,1,0  python cifar10_main.py --resnet_size=20 --batch_size=32 --resnet_version=1 --use_synthetic_data --clean -md ./tmp1

