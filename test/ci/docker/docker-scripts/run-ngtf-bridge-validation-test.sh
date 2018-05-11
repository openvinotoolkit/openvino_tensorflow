#!  /bin/bash

# This script is designed to be called from within a docker container.
# It is installed into a docker image.  It will not run outside the container.

# Use either the environment variable TF_NG_MODEL_DATASET or the first parameter
# as the model and dataset to run.  First parameter takes priority.
# Syntax is "model-dataset".


if [ ! -z "${TF_NG_MODEL_DATASET}" ] ; then
    model_dataset="${TF_NG_MODEL_DATASET}"
fi
if [ ! -z "${1}" ] ; then
    model_dataset="${1}"
fi
if [ -z "${model_dataset}" ] ; then
    ( >&2 echo "SYNTAX ERROR: First and only parameter should be model-dataset." )
    ( >&2 echo "Supported model-dataset combinations are:")
    ( >&2 echo "    mlp-mnist  resnet20-cifar10")
    exit 1
fi

# NOTE: environment variable TF_NG_ITERATIONS is also supported below


set -e  # Make sure we exit on any command that returns non-zero
set -u  # No unset variables
set -o pipefail # Make sure cmds in pipe that are non-zero also fail immediately


# ===== Function Defitions ====================================================


setup_tf_and_ngraph_plugin() {

    # ----- Pre-Wheel-Install Sanity Checks -----------------------------------

    if [ ! -f "${TF_WHEEL}" ] ; then
        ( >&2 echo "TensorFlow wheel not found at ${TF_WHEEL}" )
        exit 1
    fi

    # ------ Install TF-Wheel and Activate Virtual Environment -----------------

    xtime="$(date)"
    echo  ' '
    echo  "===== Installing nGraph-TensorFlow Wheel and Activating Virtual Environment at ${xtime} ====="
    echo  ' '

    cd "${HOME}"

    # Make sure the bash shell prompt variables are set, as virtualenv crashes
    # if PS2 is not set.
    PS1='prompt> '
    PS2='prompt-more> '

    virtualenv --system-site-packages -p /usr/bin/python2 venv-vtest
    source venv-vtest/bin/activate
    echo "Using virtual-environment at /home/dockuser/venv-vtest"

    echo "Python being used is:"
    which python

    # sudo -E pip install "${TF_WHEEL}"
    pip install "${TF_WHEEL}"

    # ------ Patch TF Install To Include nGraph-Plugin  ------------------------

    xtime="$(date)"
    echo  ' '
    echo  "===== Installing nGraph-Plugin into TF Installation at ${xtime} ====="
    echo  ' '

    tf_loc=`python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())'`
    if [ -z "${tf_loc}" ] ; then
        ( >&2 echo "TensorFlow wheel failed to install" )
        exit 1
    fi
    echo "Tensorflow installation location is: ${tf_loc}"

    cd "${tf_loc}"
    tar xvzf "${HOME}/bridge/plugins_dist.tgz"

    export LD_LIBRARY_PATH="${tf_loc}/plugins/ngraph/lib"
    echo "LD_LIBRARY_PATH is ${LD_LIBRARY_PATH}"

    echo ' '
    echo "Contents of ${tf_loc} are:"
    ls -l "${tf_loc}"

    echo ' '
    echo "Contents of plugins dir are:"
    ls -lR "${tf_loc}/plugins"

    # ----- Pre-Wheel-Install Sanity Checks ------------------------------------

    xtime="$(date)"
    echo  ' '
    echo  "===== Run Additional Sanity Check for Plugins at ${xtime} ====="
    echo  ' '

    if [ ! -f "$LD_LIBRARY_PATH/libngraph.so" ] ; then
        ( >&2 echo "FATAL ERROR: libngraph.so not found in LD_LIBRARY_PATH [$LD_LIBRARY_PATH]" )
        exit 1
    fi

    if [ ! -f "$LD_LIBRARY_PATH/libmkldnn.so" ] ; then
        ( >&2 echo "FATAL ERROR: libmkldnn.so not found in LD_LIBRARY_PATH [$LD_LIBRARY_PATH]" )
        exit 1
    fi

    cd "${HOME}/bridge"
    python test/install_test.py

}  # setup_tf_and_ngraph_plugin()


setup_tf_mkldnn() {

    # ----- Pre-Wheel-Install Sanity Checks -----------------------------------

    if [ ! -f "${TF_WHEEL_MKLDNN}" ] ; then
        ( >&2 echo "TensorFlow wheel not found at ${TF_WHEEL_MKLDNN}" )
        exit 1
    fi

    # ------ Install TF-Wheel and Activate Virtual Environment -----------------

    xtime="$(date)"
    echo  ' '
    echo  "===== Installing TensorFlow-MKLDNN Wheel and Activating Virtual Environment at ${xtime} ====="
    echo  ' '

    cd "${HOME}"

    # Make sure the bash shell prompt variables are set, as virtualenv crashes
    # if PS2 is not set.
    PS1='prompt> '
    PS2='prompt-more> '

    virtualenv --system-site-packages -p /usr/bin/python2 venv-vtest
    source venv-vtest/bin/activate
    echo "Using virtual-environment at /home/dockuser/venv-vtest"

    echo "Python being used is:"
    which python

    # sudo -E pip install "${TF_WHEEL_MKLDNN}"
    pip install "${TF_WHEEL_MKLDNN}"

    # ----- Pre-Wheel-Install Sanity Checks ------------------------------------

    xtime="$(date)"
    echo  ' '
    echo  "===== Run Sanity Check for TensorFlow-MKLDNN at ${xtime} ====="
    echo  ' '

    # One cannot import tensorflow when in the top-level of the tensorflow
    # source directory, so let's use /tmp
    cd /tmp
    python -c 'import tensorflow as tf;  hello = tf.constant("Hello world!"); sess = tf.Session(); print(sess.run(hello))'

}  # setup_tf_mkldnn()


setup_MNIST_dataset() {

    cd "${HOME}/bridge/test/ci"

    xtime="$(date)"
    echo  ' '
    echo  "===== Locating MNIST Dataset for Daily Validation at ${xtime} ====="
    echo  ' '

    # Obtain a local copy of the dataset used by the Tensorflow's
    # MNIST scripts.  The MNIST scripts theoretically have the ability
    # to download these data files themselves, but that code appears
    # unable to deal with certain firewall / proxy situations.  The
    # `download-mnist-data.sh` script called below has no such
    # problem.
    #
    # Note that this environment variable is read by several scripts
    # exercised during our CI testing.  This variable ultimately is
    # used as the '--data_dir ...' argument for our MNIST scripts
    # during integration testing.
    if [ -d '/home/dockuser/dataset/mnist' ] ; then
        dataDir='/home/dockuser/dataset/mnist'
    elif [ -d '/dataset/mnist' ] ; then
        dataDir='/dataset/mnist'
    else
        dataDir='/tmp/tensorflow/mnist/input_data'
        mkdir -p "$dataDir"
        ../tensorflow/compiler/plugin/ngraph/tests/integration-tests/util/download-mnist-data.sh "$dataDir"
        echo ' '
    fi
    echo " Listing of files in directory '${dataDir}':"
    ls -l "${dataDir}"
    echo ' '

}  # setup_MNIST()      


setup_CIFAR10_dataset() {

    cd "${HOME}/bridge/test/ci"

    xtime="$(date)"
    echo  ' '
    echo  "===== Locating CIFAR10 Dataset for Daily Validation at ${xtime} ====="
    echo  ' '

    # Copy cifar10 data from /dataset to /tmp directory (in Docker container).
    # The resnet20 script unpacks the cifar10 data in the same directory, and we
    # do not want that to happen in /dataset
    dataDir='/tmp/cifar10_input_data'
    if [ -f '/home/dockuser/dataset/cifar-10-binary.tar.gz' ] ; then
        mkdir /tmp/cifar10_input_data
        cp -v /home/dockuser/dataset/cifar-10-binary.tar.gz /tmp/cifar10_input_data
        (cd /tmp/cifar10_input_data; tar xvzf cifar-10-binary.tar.gz)
    else
        ( >&2 echo "FATAL ERROR: /home/dockeruser/dataset/cifar-10-binary.tar.gz not found" )
        exit 1
    fi
    echo " Listing of files in directory '${dataDir}':"
    ls -l "${dataDir}"
    echo ' '

}  # setup_CIFAR10_dataset()


run_MLP_MNIST() {

    cd "${HOME}/bridge/test/ci"

    xtime="$(date)"
    echo  ' '
    echo  "===== Running Tensorflow Daily Validation on CPU-Backend at ${xtime} ====="
    echo  ' '
    echo  "===== PWD is $PWD ====="
    # In MLP pytest script, OMP_NUM_THREADS and KMP_AFFINITY are explicitly
    # set only for the nGraph run.  Thus, they are not set here.
    # Test parameters
    export TEST_MLP_MNIST_DATA_DIR="${dataDir}"
    export TEST_MLP_MNIST_LOG_DIR="${HOME}/bridge"
    export TEST_MLP_MNIST_ITERATIONS="${TF_NG_ITERATIONS:-}"
    if [ -z "${TEST_MLP_MNIST_ITERATIONS}" ] ; then
        export TEST_MLP_MNIST_ITERATIONS=100000  # Default is 100,000 iterations
    fi
    # Run the test
    pytest -s ./test_mlp_mnist_cpu_daily_validation.py --junit-xml=../../validation_tests_mlp_mnist_cpu.xml --junit-prefix=daily_validation_mlp_mnist_cpu
    echo "===== Daily Validation CPU-Backend Pipeline Exited with $? ====="

}  # run_MLP_MNIST()


run_convnet_MNIST() {

    cd "${HOME}/bridge/test/ci"

    xtime="$(date)"
    echo  ' '
    echo  "===== Running Tensorflow Daily Validation on CPU-Backend at ${xtime} ====="
    echo  ' '
    echo  "===== PWD is $PWD ====="
    # Test parameters
    export TEST_CONVNET_MNIST_DATA_DIR="${dataDir}"
    export TEST_CONVNET_MNIST_LOG_DIR="${HOME}/bridge"
    export TEST_CONVNET_MNIST_ITER="${TF_NG_ITERATIONS:-}"
    if [ -z "${TEST_CONVNET_MNIST_ITER}" ] ; then
        export TEST_CONVNET_MNIST_ITER=1000  # Default is 1000 epochs
    fi
    # Run the test

    pytest -s test_convnet_mnist_cpu_daily_validation.py --junit-xml=../../validation_tests_convnet_mnist_cpu.xml --junit-prefix=daily_validation_convnet_mnist_cpu
    echo "===== Daily Validation CPU-Backend Pipeline Exited with $? ====="

}  # run_convnet_MNIST()


run_convnet_MNIST_ngraph_reference() {

    cd "${HOME}/bridge/test/ci"

    xtime="$(date)"
    echo  ' '
    echo  "===== Running Tensorflow Daily Validation on CPU-Backend at ${xtime} ====="
    echo  ' '
    echo  "===== PWD is $PWD ====="
    # Test parameters
    export TEST_CONVNET_MNIST_DATA_DIR="${dataDir}"
    export TEST_CONVNET_MNIST_LOG_DIR="${HOME}/bridge"
    export TEST_CONVNET_MNIST_ITER="${TF_NG_ITERATIONS:-}"
    if [ -z "${TEST_CONVNET_MNIST_ITER}" ] ; then
        export TEST_CONVNET_MNIST_ITER=1000  # Default is 1000 epochs
    fi
    # Run the test

    pytest -s test_convnet_mnist_reference_cpu_validation.py --junit-xml=../../validation_tests_convnet_mnist_reference_cpu.xml --junit-prefix=daily_validation_convnet_mnist_reference_cpu
    echo "===== Daily Validation CPU-Backend Pipeline Exited with $? ====="

}  # run_convnet_MNIST_ngraph()


run_convnet_MNIST_ngraph() {

    cd "${HOME}/bridge/test/ci"

    xtime="$(date)"
    echo  ' '
    echo  "===== Running Tensorflow Daily Validation on CPU-Backend at ${xtime} ====="
    echo  ' '
    echo  "===== PWD is $PWD ====="
    # Test parameters
    export TEST_CONVNET_MNIST_DATA_DIR="${dataDir}"
    export TEST_CONVNET_MNIST_LOG_DIR="${HOME}/bridge"
    export TEST_CONVNET_MNIST_ITER="${TF_NG_ITERATIONS:-}"
    if [ -z "${TEST_CONVNET_MNIST_ITER}" ] ; then
        export TEST_CONVNET_MNIST_ITER=1000  # Default is 1000 epochs
    fi
    # Make sure the comparison env. var. is not present
    unset TEST_CONVNET_ITER_COMPARE_TO
    # Run the test

    pytest -s test_convnet_mnist_ngraph_cpu_validation.py --junit-xml=../../validation_tests_convnet_mnist_ngraph_cpu.xml --junit-prefix=daily_validation_convnet_mnist_ngraph_cpu
    echo "===== Daily Validation CPU-Backend Pipeline Exited with $? ====="

}  # run_convnet_MNIST_ngraph()


run_convnet_MNIST_ngraph_compare() {

    cd "${HOME}/bridge/test/ci"

    xtime="$(date)"
    echo  ' '
    echo  "===== Running Tensorflow Daily Validation on CPU-Backend at ${xtime} ====="
    echo  ' '
    echo  "===== PWD is $PWD ====="
    # Test parameters
    export TEST_CONVNET_MNIST_DATA_DIR="${dataDir}"
    export TEST_CONVNET_MNIST_LOG_DIR="${HOME}/bridge"
    export TEST_CONVNET_MNIST_ITER="${TF_NG_ITERATIONS:-}"
    export TEST_CONVNET_MNIST_COMPARE_TO="${TF_NG_COMPARE_TO:-}"
    if [ -z "${TEST_CONVNET_MNIST_ITER}" ] ; then
        export TEST_CONVNET_MNIST_ITER=1000  # Default is 1000 epochs
    fi
    # Make sure the comparison env. var. is not present
    unset TEST_CONVNET_ITER_COMPARE_TO
    # Run the test

    pytest -s test_convnet_mnist_ngraph_cpu_validation.py --junit-xml=../../validation_tests_convnet_mnist_ngraph_cpu.xml --junit-prefix=daily_validation_convnet_mnist_ngraph_cpu
    echo "===== Daily Validation CPU-Backend Pipeline Exited with $? ====="

}  # run_convnet_MNIST_ngraph_compare()


run_resnet20_CIFAR10() {

    cd "${HOME}/bridge/test/ci"

    xtime="$(date)"
    echo  ' '
    echo  "===== Running Tensorflow Daily Validation on CPU-Backend at ${xtime} ====="
    echo  ' '
    echo  "===== PWD is $PWD ====="
    # In Resnet20 pytest script, OMP_NUM_THREADS and KMP_AFFINITY are explicitly
    # set only for the nGraph run.  Thus, they are not set here.
    # Test parameters
    export TEST_RESNET20_CIFAR10_DATA_DIR="${dataDir}"
    export TEST_RESNET20_CIFAR10_LOG_DIR="${HOME}/bridge"
    export TEST_RESNET20_CIFAR10_EPOCHS="${TF_NG_EPOCHS:-}"
    if [ -z "${TEST_RESNET20_CIFAR10_EPOCHS}" ] ; then
        export TEST_RESNET20_CIFAR10_EPOCHS=250  # Default is 250 epochs
    fi
    # Run the test
    pytest -s ./test_resnet20_cifar10_cpu_daily_validation.py --junit-xml=../../validation_tests_resnet20_cifar10_cpu.xml --junit-prefix=daily_validation_resnet20_cifar10_cpu
    echo "===== Daily Validation CPU-Backend Pipeline Exited with $? ====="

}  # run_resnet20_CIFAR10()


run_resnet20_CIFAR10_reference() {

    cd "${HOME}/bridge/test/ci"

    xtime="$(date)"
    echo  ' '
    echo  "===== Running Tensorflow Daily Validation on TF-MKLDNN CPU-Backend at ${xtime} ====="
    echo  ' '
    echo  "===== PWD is $PWD ====="
    # In Resnet20 pytest script, OMP_NUM_THREADS and KMP_AFFINITY are explicitly
    # set only for the nGraph run.  Thus, they are not set here.
    # Test parameters
    export TEST_RESNET20_CIFAR10_DATA_DIR="${dataDir}"
    export TEST_RESNET20_CIFAR10_LOG_DIR="${HOME}/bridge"
    export TEST_RESNET20_CIFAR10_EPOCHS="${TF_NG_EPOCHS:-}"
    if [ -z "${TEST_RESNET20_CIFAR10_EPOCHS}" ] ; then
        export TEST_RESNET20_CIFAR10_EPOCHS=250  # Default is 250 epochs
    fi
    # Run the test

    pytest -s test_resnet20_cifar10_reference_cpu_validation.py --junit-xml=../../validation_tests_resnet20_cifar10_reference_cpu.xml --junit-prefix=daily_validation_resnet20_cifar10_reference_cpu
    echo "===== Daily Validation CPU-Backend Pipeline Exited with $? ====="

}  # run_resnet20_CIFAR10_reference()


run_resnet20_CIFAR10_ngraph() {

    cd "${HOME}/bridge/test/ci"

    xtime="$(date)"
    echo  ' '
    echo  "===== Running Tensorflow Daily Validation on CPU-Backend at ${xtime} ====="
    echo  ' '
    echo  "===== PWD is $PWD ====="
    # In Resnet20 pytest script, OMP_NUM_THREADS and KMP_AFFINITY are explicitly
    # set only for the nGraph run.  Thus, they are not set here.
    # Test parameters
    export TEST_RESNET20_CIFAR10_DATA_DIR="${dataDir}"
    export TEST_RESNET20_CIFAR10_LOG_DIR="${HOME}/bridge"
    export TEST_RESNET20_CIFAR10_EPOCHS="${TF_NG_EPOCHS:-}"
    if [ -z "${TEST_RESNET20_CIFAR10_EPOCHS}" ] ; then
        export TEST_RESNET20_CIFAR10_EPOCHS=250  # Default is 250 epochs
    fi
    # Make sure the comparison env. var. is not present
    unset TEST_RESNET20_CIFAR10_COMPARE_TO
    # Run the test

    pytest -s test_resnet20_cifar10_ngraph_cpu_validation.py --junit-xml=../../validation_tests_resnet20_cifar10_ngraph_cpu.xml --junit-prefix=daily_validation_resnet20_cifar10_ngraph_cpu
    echo "===== Daily Validation CPU-Backend Pipeline Exited with $? ====="

}  # run_resnet20_CIFAR10_ngraph()


run_resnet20_CIFAR10_ngraph_compare() {

    cd "${HOME}/bridge/test/ci"

    xtime="$(date)"
    echo  ' '
    echo  "===== Running Tensorflow Daily Validation on CPU-Backend at ${xtime} ====="
    echo  ' '
    echo  "===== PWD is $PWD ====="
    # In Resnet20 pytest script, OMP_NUM_THREADS and KMP_AFFINITY are explicitly
    # set only for the nGraph run.  Thus, they are not set here.
    # Test parameters
    export TEST_RESNET20_CIFAR10_DATA_DIR="${dataDir}"
    export TEST_RESNET20_CIFAR10_LOG_DIR="${HOME}/bridge"
    export TEST_RESNET20_CIFAR10_EPOCHS="${TF_NG_EPOCHS:-}"
    export TEST_RESNET20_CIFAR10_COMPARE_TO="${TF_NG_COMPARE_TO:-}"
    if [ -z "${TEST_RESNET20_CIFAR10_EPOCHS}" ] ; then
        export TEST_RESNET20_CIFAR10_EPOCHS=250  # Default is 250 epochs
    fi
    # Run the test

    pytest -s test_resnet20_cifar10_ngraph_cpu_validation.py --junit-xml=../../validation_tests_resnet20_cifar10_ngraph_cpu.xml --junit-prefix=daily_validation_resnet20_cifar10_ngraph_cpu
    echo "===== Daily Validation CPU-Backend Pipeline Exited with $? ====="

}  # run_resnet20_CIFAR10_ngraph_compare()


# ===== Main ==================================================================

# For now we simply test ng-tf for python 2.  Later, python 3 builds will
# be added.
export PYTHON_VERSION_NUMBER=2
export PYTHON_BIN_PATH="/usr/bin/python$PYTHON_VERSION_NUMBER"

# This path is dependent on where host dir-tree is mounted into docker run
# See script docker-run-ngtf-bridge-validation-test.sh
# HOME is expected to be /home/dockuser.  See script run-as-user.sh, which
# sets this up.
cd "$HOME/bridge"

export TF_WHEEL="$HOME/bridge/tensorflow-1.6.0-cp27-cp27mu-linux_x86_64.whl"
export TF_WHEEL_MKLDNN="$HOME/bridge/tensorflow-mkldnn-1.6.0-cp27-cp27mu-linux_x86_64.whl"

echo "In $(basename ${0}):"
echo "  model_dataset=${model_dataset}"
echo "  HOME=${HOME}"
echo "  PYTHON_VERSION_NUMBER=${PYTHON_VERSION_NUMBER}"
echo "  PYTHON_BIN_PATH=${PYTHON_BIN_PATH}"
echo "  TF_WHEEL=${TF_WHEEL}"
echo "  TF_WHEEL=${TF_WHEEL_MKLDNN}"

# ----- Install Dataset and Run Pytest Script ----------------------------------

# test/ci is located in the mounted ngraph-tensorflow-bridge cloned repo
cd "${HOME}/bridge/test/ci"

# Case switch to run data setup and pytest script, for each
# model + dataset that is supported

case "${model_dataset}" in

# nGraph-only run, but compares to provide results (in JSON format)
# Produces an ngraph results JSON file.
#
convnet-mnist-ngraph)
    setup_tf_and_ngraph_plugin
    setup_MNIST_dataset
    run_convnet_MNIST_ngraph
    ;;
resnet20-cifar10-ngraph)
    setup_tf_and_ngraph_plugin
    setup_CIFAR10_dataset
    run_resnet20_CIFAR10_ngraph
    ;;

# nGraph+comparison
convnet-mnist-ngraph-compare)
    setup_tf_and_ngraph_plugin
    setup_MNIST_dataset
    run_convnet_MNIST_ngraph_compare
    ;;
resnet20-cifar10-ngraph-compare)
    setup_tf_and_ngraph_plugin
    setup_CIFAR10_dataset
    run_resnet20_CIFAR10_ngraph_compare
    ;;

# Reference-only run, with tf-mkldnn.  Produces a reference results JSON file.
#
convnet-mnist-ngraph-ref)
    setup_tf_and_ngraph_plugin
    setup_MNIST_dataset
    run_convnet_MNIST_ngraph_reference
    ;;
resnet20-cifar10-ref)  # nGraph+comparison: Resnet20 with CIFAR10 dataset
    setup_tf_mkldnn
    setup_CIFAR10_dataset
    run_resnet20_CIFAR10_reference
    exit 1
    ;;

# Reference + nGraph runs.
#
# NOTE: Reference runs done with the SAME tensorflow wheel (with ngraph-plugin)
# as ngraph runs
#
mlp-mnist)  # Ref+nGraph: Multi-Layer Perceptron (MLP) with MNIST dataset
    setup_tf_and_ngraph_plugin
    setup_MNIST_dataset
    run_MLP_MNIST
    ;;
convnet-mnist)
    setup_tf_and_ngraph_plugin
    setup_MNIST_dataset
    run_convnet_MNIST
    ;;
resnet20-cifar10)  # Ref+nGraph: Resnet20 with CIFAR10 dataset
    setup_tf_and_ngraph_plugin
    setup_CIFAR10_dataset
    run_resnet20_CIFAR10
    ;;

# Unrecognized model_dataset code
*)
    ( >&2 echo "FATAL ERROR: ${model_dataset} is not supported in this script")
    exit 1
    ;;
esac

xtime="$(date)"
echo  ' '
echo  "===== Deactivating the Virtual Environment at ${xtime} ====="
echo  ' '

deactivate

xtime="$(date)"
echo ' '
echo "===== Completed NGraph-Tensorflow-Bridge Validation Test for ${model_dataset} at ${xtime} ====="
echo ' '

exit 0
