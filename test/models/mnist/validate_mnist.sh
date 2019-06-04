#!/bin/bash

mnist_training_example="../../../examples/mnist/mnist_deep_simplified.py"
echo $mnist_training_example

echo "**************************************************************************"
echo "Run MNIST ADAM OPTIMIZER on NGRAPH"
echo "**************************************************************************"
python $mnist_training_example --train_loop_count=20 --make_deterministic

echo "**************************************************************************"
echo "Run MNIST ADAM OPTIMIZER on TF"
echo "**************************************************************************"
NGRAPH_TF_DISABLE=1 python $mnist_training_example --train_loop_count=20 --make_deterministic

echo "**************************************************************************"
echo "Run MNIST GRADIENT DESCENT on NGRAPH"
echo "**************************************************************************"

python $mnist_training_example --train_loop_count=20 --make_deterministic --optimizer sgd

echo "**************************************************************************"
echo "Run MNIST GRADIENT DESCENT on TF"
echo "**************************************************************************"

NGRAPH_TF_DISABLE=1 python $mnist_training_example --train_loop_count=20 --make_deterministic --optimizer sgd
