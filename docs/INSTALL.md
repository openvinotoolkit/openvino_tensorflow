# 1. <a name='Pre-BuiltPackages'></a>Installation using Pre-Built Packages

**OpenVINO™ integration with TensorFlow** has two releases: one built with CXX11_ABI=0 and another built with CXX11_ABI=1.

Since TensorFlow packages available in [PyPi](https://pypi.org) are built with CXX11_ABI=0 and OpenVINO™ release packages are built with CXX11_ABI=1, binary releases of these packages **cannot be installed together**. Based on your needs, you can choose one of the two available methods:

- **OpenVINO™ integration with TensorFlow** PyPi release alongside PyPi TensorFlow
  (CXX11_ABI=0, no OpenVINO™ installation required, no VAD-M support)

- **OpenVINO™ integration with TensorFlow** package released in Github alongside the Intel® Distribution of OpenVINO™ Toolkit
  (CXX11_ABI=1, needs a custom TensorFlow package, enables VAD-M support)  
<br/>  

## <a name='Prebuiltpackagessummary'></a>Pre-built packages summary
  
|TensorFlow Pip Package| **OpenVINO™ integration with TensorFlow** Pip Package|Supported OpenVINO™ Flavor|Supported Hardware Backends|Comments|
| -----------------|-----------------------------------|----------------------------|---------------------------|----------------|
|tensorflow| openvino-tensorflow|OpenVINO™ built from source|CPU,GPU,MYRIAD|**OpenVINO™** libraries are built from source and included in the wheel package|
|tensorflow-abi1| openvino-tensorflow-abi1|Dynamically links to OpenVINO™ binary release|CPU,GPU,MYRIAD,VAD-M|**OpenVINO™ integration with TensorFlow** libraries are dynamically linked to OpenVINO™ binaries|
<br/>  

##  1.1. <a name='InstallOpenVINOintegrationwithTensorFlowalongsidePyPiTensorFlow'></a>Install **OpenVINO™ integration with TensorFlow** alongside PyPi TensorFlow

This **OpenVINO™ integration with TensorFlow** package includes pre-built libraries of OpenVINO™ version 2021.4.1. The users do not have to install OpenVINO™ separately. This package supports Intel<sup>®</sup> CPUs, Intel<sup>®</sup> integrated GPUs, and Intel<sup>®</sup> Movidius™ Vision Processing Units (VPUs).


        pip3 install pip==21.0.1
        pip3 install tensorflow==2.5.1
        pip3 install -U openvino-tensorflow
<br/> 

##  1.2. <a name='InstallOpenVINOintegrationwithTensorFlowalongsidetheIntelDistributionofOpenVINOToolkit'></a>Install **OpenVINO™ integration with TensorFlow** alongside the Intel® Distribution of OpenVINO™ Toolkit

This **OpenVINO™ integration with TensorFlow** package is currently compatible with OpenVINO™ version 2021.4.1. This package supports Intel<sup>®</sup> CPUs, Intel<sup>®</sup> integrated GPUs, Intel<sup>®</sup> Movidius™ Vision Processing Units (VPUs), and Intel<sup>®</sup> Vision Accelerator Design with Movidius™ (VAD-M).

You can build [TensorFlow from source](https://github.com/openvinotoolkit/openvino_tensorflow/blob/master/docs/BUILD.md#tensorflow) with -D_GLIBCXX_USE_CXX11_ABI=1 or follow the instructions below to install the appropriate version:

1. Ensure the following versions are being used for pip and numpy:

        pip3 install pip==21.0.1
        pip3 install numpy==1.20.2

2. Install `TensorFlow`. Based on your Python version, use the appropriate package below:

        pip3.6 install https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v1.0.0/tensorflow_abi1-2.5.1-cp36-cp36m-manylinux2010_x86_64.whl

        or

        pip3.7 install https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v1.0.0/tensorflow_abi1-2.5.1-cp37-cp37m-manylinux2010_x86_64.whl

        or

        pip3.8 install https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v1.0.0/tensorflow_abi1-2.5.1-cp38-cp38-manylinux2010_x86_64.whl

3. Download & install Intel® Distribution of OpenVINO™ Toolkit 2021.4.1 release along with its dependencies from ([https://software.intel.com/en-us/openvino-toolkit/download](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit/download.html)).

4. Initialize the OpenVINO™ environment by running the `setupvars.sh` located in <code>\<openvino\_install\_directory\>\/bin</code> using the command below:

        source setupvars.sh

5. Install `openvino-tensorflow`. Based on your Python version, choose the appropriate package below:

        pip3.6 install https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v1.0.0/openvino_tensorflow_abi1-1.0.0-cp36-cp36m-linux_x86_64.whl

        or

        pip3.7 install https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v1.0.0/openvino_tensorflow_abi1-1.0.0-cp37-cp37m-linux_x86_64.whl

        or

        pip3.8 install https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v1.0.0/openvino_tensorflow_abi1-1.0.0-cp38-cp38-linux_x86_64.whl


