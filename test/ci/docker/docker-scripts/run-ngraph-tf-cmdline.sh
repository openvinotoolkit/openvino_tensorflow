#!  /bin/bash

# ==============================================================================
#  Copyright 2018 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ==============================================================================

# This script is designed to be called from within a docker container.
# It is installed into a docker image.  It will not run outside the container.

if [ -z "${CMDLINE}" ] ; then
    ( >&2 echo "CMDLINE not set when run-ngtf-bridge-validation-cmdline.sh called")
    exit
fi

# NG_TF_DATASET can be an empty string, or not set, but make sure it has a value
if [ -z "${NG_TF_DATASET}" ] ; then  NG_TF_DATASET='' ; fi

# NG_TF_LOG_ID can be an empty string, or not set, but make sure it has a value
if [ -z "${NG_TF_LOG_ID}" ] ; then  NG_TF_LOG_ID='' ; fi

# NG_TF_PY_VERSION can be an empty string or not, but make sure it has a value
if [ -z "${NG_TF_PY_VERSION}" ] ; then  NG_TF_PY_VERSION='' ; fi

# NG_TF_WHEEL_NGRAPH can be an empty string or not, but make sure it has a value

if [ -z "${NG_TF_WHEEL_NGRAPH}" ] ; then  NG_TF_WHEEL_NGRAPH='' ; fi

# NG_TF_WHEEL_TF can be an empty string or not, but make sure it has a value
if [ -z "${NG_TF_WHEEL_TF}" ] ; then  NG_TF_WHEEL_TF='' ; fi

# NG_TF_WHEEL_TF_REF can be an empty string or not, but make sure it has a value
if [ -z "${NG_TF_WHEEL_TF_REF}" ] ; then  NG_TF_WHEEL_TF_REF='' ; fi

# NG_TF_RUN_TYPE cannot be an empty string, so make "ngraph" the default
if [ -z "${NG_TF_RUN_TYPE}" ] ; then  NG_TF_RUN_TYPE='ngraph' ; fi

set -e  # Make sure we exit on any command that returns non-zero
set -u  # No unset variables
set -o pipefail # Make sure cmds in pipe that are non-zero also fail immediately


# ===== Function Defitions ====================================================


setup_tf_and_ngraph_wheels() {

    # ----- Pre-Wheel-Install Sanity Checks -----------------------------------

    if [ ! -f "${TF_WHEEL}" ] ; then
        ( >&2 echo "TensorFlow wheel not found at ${TF_WHEEL}" )
        exit 1
    fi

    if [ ! -f "${NGTF_WHEEL}" ] ; then
        ( >&2 echo "ngraph wheel not found at ${NGTF_WHEEL}" )
        exit 1
    fi

    # ------ Install nGraph and TF Wheels and Activate Virtual Environment -----------------

    xtime="$(date)"
    echo  ' '
    echo  "===== Installing nGraph and TensorFlow Wheels and Activating Virtual Environment at ${xtime} ====="
    echo  ' '

    cd "${HOME}"

    # Make sure the bash shell prompt variables are set, as virtualenv crashes
    # if PS2 is not set.
    PS1='prompt> '
    PS2='prompt-more> '

    virtualenv --system-site-packages -p "${PYTHON_BIN_PATH}" venv-vtest
    source venv-vtest/bin/activate
    echo "Using virtual-environment at /home/dockuser/venv-vtest"

    echo "Python being used is:"
    which python

    pip install "${TF_WHEEL}"

    pip install "${NGTF_WHEEL}"

    # ----- Pre-Wheel-Install Sanity Checks ------------------------------------

    xtime="$(date)"
    echo  ' '
    echo  "===== Run Additional Sanity Check for Plugins at ${xtime} ====="
    echo  ' '

    # TODO: Remove when appropriate
    # if [ ! -f "${NGTF_DIST}/libngraph_device.so" ] ; then
    #     ( >&2 echo "FATAL ERROR: libngraph_device.so not found in ${NGTF_DIST}" )
    #     exit 1
    # fi
    # 
    # if [ ! -f "${NGTF_DIST}/ngraph_dist/lib/libngraph.so" ] ; then
    #     ( >&2 echo "FATAL ERROR: libngraph.so not found in ${NGTF_DIST}/ngraph_dist/lib" )
    #     exit 1
    # fi
    # 
    # if [ ! -f "${NGTF_DIST}/ngraph_dist/lib/libmkldnn.so" ] ; then
    #     ( >&2 echo "FATAL ERROR: libmkldnn.so not found in ${NGTF_DIST}/ngraph_dist/lib" )
    #     exit 1
    # fi

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

    virtualenv --system-site-packages -p "${PYTHON_BIN_PATH}" venv-vtest
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


setup_no_wheel() {

    # ------ Activate Virtual Environment -----------------

    xtime="$(date)"
    echo  ' '
    echo  "===== Activating Virtual Environment at ${xtime} ====="
    echo  ' '

    cd "${HOME}"

    # Make sure the bash shell prompt variables are set, as virtualenv crashes
    # if PS2 is not set.
    PS1='prompt> '
    PS2='prompt-more> '

    virtualenv --system-site-packages -p "${PYTHON_BIN_PATH}" venv-vtest
    source venv-vtest/bin/activate
    echo "Using virtual-environment at /home/dockuser/venv-vtest"

    echo "Python being used is:"
    which python

}  # setup_no_wheel()


setup_MNIST_dataset() {

    cd "${HOME}/bridge/test/ci"

    xtime="$(date)"
    echo  ' '
    echo  "===== Locating MNIST Dataset for Model Run at ${xtime} ====="
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
    if [ -d '/dataset/mnist' ] ; then
        dataDir='/dataset/mnist'
    else
        ( >&2 echo "FATAL ERROR: /dataset/mnist not found" )
        exit 1
    fi
    echo " Listing of files in directory '${dataDir}':"
    ls -l "${dataDir}"
    echo ' '

}  # setup_MNIST()      


setup_CIFAR10_dataset() {

    cd "${HOME}/bridge/test/ci"

    xtime="$(date)"
    echo  ' '
    echo  "===== Locating CIFAR10 Dataset for Model Run at ${xtime} ====="
    echo  ' '

    # Copy cifar10 data from /dataset to /tmp directory (in Docker container).
    # The resnet20 script unpacks the cifar10 data in the same directory, and we
    # do not want that to happen in /dataset
    dataDir='/tmp/cifar10'
    if [ -f '/dataset/cifar-10-binary.tar.gz' ] ; then
        mkdir "${dataDir}"
        cp -v /dataset/cifar-10-binary.tar.gz "${dataDir}"
        (cd "${dataDir}"; tar xvzf cifar-10-binary.tar.gz)
    else
        ( >&2 echo "FATAL ERROR: /dataset/cifar-10-binary.tar.gz not found" )
        exit 1
    fi
    echo " Listing of files in directory '${dataDir}':"
    ls -l "${dataDir}"
    echo ' '

}  # setup_CIFAR10_dataset()


# ===== Main ==================================================================

# Default is Python 2, but can override with NG_TF_PY_VERSION env. variable
export PYTHON_VERSION_NUMBER="${NG_TF_PY_VERSION}"
if [ -z "${PYTHON_VERSION_NUMBER}" ] ; then
    PYTHON_VERSION_NUMBER=2
fi
export PYTHON_BIN_PATH="/usr/bin/python${PYTHON_VERSION_NUMBER}"

# This path is dependent on where host dir-tree is mounted into docker run
# See script docker-run-ngtf-bridge-validation-test.sh
# HOME is expected to be /home/dockuser.  See script run-as-user.sh, which
# sets this up.
cd "$HOME/bridge"

if [ -z "${NG_TF_WHEEL_NGRAPH}" ] ; then
    NGTF_WHEEL="$HOME/bridge/ngraph-0.0.0-py2.py3-none-linux_x86_64.whl"
else
    NGTF_WHEEL="$HOME/bridge/${NG_TF_WHEEL_NGRAPH}"
fi

if [ -z "${NG_TF_WHEEL_TF}" ] ; then
    TF_WHEEL="$HOME/bridge/tensorflow-1.9.0-cp27-cp27mu-linux_x86_64.whl"
else
    TF_WHEEL="$HOME/bridge/${NG_TF_WHEEL_TF}"
fi

if [ -z "${NG_TF_WHEEL_TF_REF}" ] ; then
    TF_WHEEL_MKLDNN="$HOME/bridge/tensorflow-mkldnn-1.9.0-cp27-cp27mu-linux_x86_64.whl"
else
    TF_WHEEL_MKLDNN="$HOME/bridge/${NG_TF_WHEEL_TF_REF}"
fi

echo "In $(basename ${0}):"
echo "  CMDLINE=[${CMDLINE}]"
echo "  NG_TFDATASET=${NG_TF_DATASET}"
echo "  NG_TF_LOG_ID=${NG_TF_LOG_ID}"
echo "  HOST_HOSTNAME=${HOST_HOSTNAME}"
echo "  HOSTNAME=${HOSTNAME}"
echo "  HOME=${HOME}"
echo "  PYTHON_VERSION_NUMBER=${PYTHON_VERSION_NUMBER}"
echo "  PYTHON_BIN_PATH=${PYTHON_BIN_PATH}"
echo "  NGTF_WHEEL=${NGTF_WHEEL}"
echo "  TF_WHEEL=${TF_WHEEL}"
echo "  TF_WHEEL_MKLDNN=${TF_WHEEL_MKLDNN}"

# ----- Set defaults -----------------------------------------------------------

if [ -z "${NG_TF_RUN_TYPE}" ] ; then
    export NG_TF_RUN_TYPE='ngraph'  # Default to using ngraph for run
fi

# ----- Report on Optional Mounted Directories ---------------------------------

if [ -d "${HOME}/ngraph-models" ] ; then
    echo "NG_TF_MODELS_REPO mounted at ${HOME}/ngraph-models"
else
    echo "NG_TF_MODELS_REPO was not specified, and thus is not mounted"
fi

if [ -d /aipg_trained_dataset ] ; then
    echo "NG_TF_TRAINED mounted at /aipg_trained_dataset"
else
    echo "NG_TF_TRAINED was not specified, and thus is not mounted"
fi

# ----- Install Dataset and Run Pytest Script ----------------------------------

# test/ci is located in the mounted ngraph-tensorflow-bridge cloned repo
cd "${HOME}/bridge/test/ci"

# Install either the tf+ngraph or reference wheels
case "${NG_TF_RUN_TYPE}" in
    # Run with a tensorflow wheel and ngraph wheel
    ngraph-tf)
        setup_tf_and_ngraph_wheels
        ;;
    # Run with a generic tensorflow with wheel "tensorflow-mkldnn*.whl"
    reference)
        setup_tf_mkldnn
        ;;
    # Run with ngraph built into the tensorflow wheel
    tf-with-ngraph)
        # FUTURE: support this mode!
        ( >&2 echo "ERROR: run-type 'tf-with-ngraph' is not yet supported")
        ;;
    # Special run-type where no wheels are installed
    no-wheel)
        setup_no_wheel
        ;;
    *)
        ( >&2 echo "FATAL ERROR: run-type ${NG_TF_RUN_TYPE} is not supported in this script")
        exit 1
        ;;
esac

# If a dataset is requested, then set it up
case "${NG_TF_DATASET}" in
    aipg_trained_dataset)
        # Results can be accessed in /aipg_trained_dataset
        if [ ! -d "/aipg_trained_dataset" ] ; then
            ( >&2 echo "FATAL ERROR: aipg_trained_datase selected, but /aipg_trained_dataset is not mounted" )
            exit 1
        fi
        ;;
    mnist)
        setup_MNIST_dataset    # Results can be accessed in /dataset/mnist
        ;;
    cifar10)
        setup_CIFAR10_dataset  # Results can be accessed in /tmp/cifar10
        ;;
    '')
        ( >&2 echo "NG_TF_DATASET not specified, so no dataset preparation has been done")
        ;;
    *)
        ( >&2 echo "FATAL ERROR: dataset ${NG_TF_DATASET} is not supported in this script")
        exit 1
        ;;
esac

xtime="$(date)"
echo ' '
echo "===== Running NGraph-TF Model Run(s) on CPU-Backend at ${xtime} ====="

cd "${HOME}/bridge"

if [ -z "${NG_TF_LOG_ID}" ] ; then
     logfile="${PWD}/log_run_cmdline.txt"
else
     logfile="${PWD}/log_${NG_TF_LOG_ID}.txt"
fi

echo ' '
echo "Running command: [${CMDLINE}]"
echo "Saving output to file: ${logfile}"

echo "Command: ${CMDLINE}" > "${logfile}"
eval "${CMDLINE}" 2>&1 | tee -a "${logfile}"

xtime="$(date)"
echo  ' '
echo  "===== Deactivating the Virtual Environment at ${xtime} ====="
echo  ' '

deactivate

xtime="$(date)"
echo ' '
echo "===== Completed NGraph-TF Model Run(s) for [${CMDLINE}] at ${xtime} ====="
echo ' '

exit 0
