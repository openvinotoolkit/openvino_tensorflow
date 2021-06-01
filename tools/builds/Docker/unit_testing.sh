#Setting Up Variables
export OV_DIR=/opt/intel/openvino_2021.3.394
export OVTF_DIR=/opt/intel/openvino_tensorflow
export TF_DIR=/home/tf_m1 
export NGRAPH_TF_UTEST_RTOL=0.0001

#Activating Environment and Setting Up
bazel version
cd ${OVTF_DIR}
if [[ $(lsb_release -rs) == "18.04" ]]; 
then
    python3 build_tf.py --output_dir=${TF_DIR} --cxx11_abi_version=1
    python3 build_ovtf.py --use_tensorflow_from_location=${TF_DIR} --use_openvino_from_location=${OV_DIR} --disable_packaging_openvino_libs --cxx11_abi_version=1
else
    python3.8 build_tf.py --output_dir=${TF_DIR} --cxx11_abi_version=1
    python3.8 build_ovtf.py --use_tensorflow_from_location=${TF_DIR} --use_openvino_from_location=${OV_DIR} --disable_packaging_openvino_libs --cxx11_abi_version=1
fi
pip install keras
${OV_DIR}/install_dependencies/install_NEO_OCL_driver.sh -y
${OV_DIR}/install_dependencies/install_NCS_udev_rules.sh
source ${OV_DIR}/bin/setupvars.sh
source ${OVTF_DIR}/build_cmake/venv-tf-py3/bin/activate

#Example Testing
cd ${OVTF_DIR}/examples/data
wget https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz && tar -xzvf inception_v3_2016_08_28_frozen.pb.tar.gz
cd ${OVTF_DIR} && python3 ./examples/classification_sample.py
exit_code=$?
echo "Exit Result for Example Testing is ${exit_code}"

#tf_ov C++ Tests
function tf_ov_cpp {
    cd ${OVTF_DIR} && PYTHONPATH=`pwd` python3 test/ci/azure/test_runner.py \
    --artifacts ${OVTF_DIR}/build_cmake/artifacts/ --test_cpp
    exit_code=$?
    echo "Exit Result for tf_ov C++ Tests is ${exit_code}"
}

#C++ Inference Example
function cpp_inference {
    cd ${OVTF_DIR}/test/ci/azure/
    bash run_inception_v3.sh ${OVTF_DIR}/build_cmake/artifacts
    exit_code=$?
    echo "Exit Result for C++ Inference Example is ${exit_code}"
}

#OVTF Python Tests ${OPENVINO_TF_BACKEND}
function ovtf_python {
    cd ${OVTF_DIR} && PYTHONPATH=`pwd`:`pwd`/tools:`pwd`/examples python3 test/ci/azure/test_runner.py \
    --artifacts ${OVTF_DIR}/build_cmake/artifacts --test_python
    exit_code=$?
    echo "Exit Result for Python Tests OPENVINO_TF_BACKEND is ${exit_code}"
}

#TF Python Tests ${OPENVINO_TF_BACKEND}
function tf_python {
    cd ${OVTF_DIR} && PYTHONPATH=`pwd` python3 test/ci/azure/test_runner.py \
    --artifacts ${OVTF_DIR}/build_cmake/artifacts --test_tf_python
    exit_code=$?
    echo "Exit Result for TF Python Tests OPENVINO_TF_BACKEND is ${exit_code}"
}

for processor in CPU GPU MYRIAD
do
    
    export OPENVINO_TF_BACKEND=$processor
    echo "****************************************************${OPENVINO_TF_BACKEND}************************************************************************"
    tf_ov_cpp
    cpp_inference
    ovtf_python
    tf_python   
done
 