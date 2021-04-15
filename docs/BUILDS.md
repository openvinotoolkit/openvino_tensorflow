# Build From Source


## OpenVINO integration with TensorFlow

### Basic

1. Pulls compatible prebuilt TF package from PyPi, clones and builds OpenVINO from source. 

        python3 build_ovtf.py

2. Pulls compatible prebuilt TF package from PyPi. Uses OpenVINO binary.

        python3 build_ovtf.py –use_openvino_from_location=/opt/intel/openvino_2021.3.394/ --cxx11_abi_version=1
    

### Advanced

1. Pulls and builds TF and OpenVINO from source 

        python3 build_ovtf.py --build_tf_from_source

2. Pulls and builds TF from Source. Uses OpenVINO binary. 

        python3 build_ovtf.py –build_tf_from_source –use_openvino_from_location=/opt/intel/openvino_2021.3.394/ --cxx11_abi_version=1

3. Uses pre-built TF from the given location (built with build_tf.py). Pulls and builds OpenVINO from source. Use this if you need to build OpenVINO-TensorFlow frequently without building TF from source everytime.

        python3 build_ovtf.py –use_tensorflow_from_location=/path/to/tensorflow/build/

4. Uses prebuilt TF from the given location. Uses OpenVINO binary. This is only compatible with ABI1 built TF.

        python3 build_ovtf.py –use_tensorflow_from_location=/path/to/tensorflow/build/  –use_openvino_from_location=/opt/intel/openvino_2021/ --cxx11_abi_version=1

## TensorFlow

TensorFlow can be built from source using `build_tf.py`. The build artifacts can be found under ${PATH_TO_TF_BUILD}/artifacts/

- Set your build path
        export PATH_TO_TF_BUILD=/path/to/tensorflow/build/

- For all available build options

        python3 build_tf.py -h

- Builds TF with CXX11_ABI=0.

        python3 build_tf.py --output_dir=${PATH_TO_TF_BUILD} --cxx11_abi_version=0

- Builds TF with CXX11_ABI=1

        python3 build_tf.py --output_dir=${PATH_TO_TF_BUILD} --cxx11_abi_version=1

- To build a desired TF version

        python3 build_tf.py --output_dir=${PATH_TO_TF_BUILD} --tf_version=r2.x

## OpenVINO

OpenVINO can be built from source independently using `build_ov.py`