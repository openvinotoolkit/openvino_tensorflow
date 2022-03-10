# <a name='Pre-BuiltPackages'></a>Installation using Pre-Built Packages

**OpenVINO™ integration with TensorFlow** is released for Linux, macOS, and Windows. You can choose one of the following methods based on your requirements.


## Linux

  **OpenVINO™ integration with TensorFlow** on Linux is released in two different versions: one built with CXX11_ABI=0 and the other built with CXX11_ABI=1.

  Since TensorFlow packages available in [PyPi](https://pypi.org) are built with CXX11_ABI=0 and OpenVINO™ release packages are built with CXX11_ABI=1, binary releases of these packages **cannot be installed together**.

  - [**OpenVINO™ integration with TensorFlow** PyPi release alongside PyPi TensorFlow](#InstallOpenVINOintegrationwithTensorFlowalongsidePyPiTensorFlow)
  * Includes pre-built libraries of OpenVINO™ version 2022.1. The users do not have to install OpenVINO™ separately 
  * Supports Intel<sup>®</sup> CPUs, Intel<sup>®</sup> integrated GPUs, and Intel<sup>®</sup> Movidius™ Vision Processing Units (VPUs). No VAD-M support
  * Build with CXX11_ABI=0  

  <br/>  

  - [**OpenVINO™ integration with TensorFlow** package released in Github alongside the Intel® Distribution of OpenVINO™ Toolkit](#InstallOpenVINOintegrationwithTensorFlowalongsidetheIntelDistributionofOpenVINOToolkit)
  * Compatible with OpenVINO™ version 2022.1
  * Supports Intel<sup>®</sup> CPUs, Intel<sup>®</sup> integrated GPUs, Intel<sup>®</sup> Movidius™ Vision Processing Units (VPUs),    and Intel<sup>®</sup> Vision Accelerator Design with Movidius™ (VAD-M)
  * Build with CXX11_ABI=1
  * Needs a custom TensorFlow ABI1 package, which is available in Github release  

<br/>  

## macOS

  - [**OpenVINO™ integration with TensorFlow** PyPi release alongside PyPi TensorFlow](#InstallOpenVINOintegrationwithTensorFlowalongsidePyPiTensorFlow)
  * Includes pre-built libraries of OpenVINO™ version 2022.1. The users do not have to install OpenVINO™ separately 
  * Supports Intel<sup>®</sup> CPUs, Intel<sup>®</sup> integrated GPUs, and Intel<sup>®</sup> Movidius™ Vision Processing Units (VPUs). No VAD-M support

<br/>  

## Windows

  - [**OpenVINO™ integration with TensorFlow** PyPi release alongside TensorFlow released in Github](#InstallOpenVINOintegrationwithTensorFlowalongsideTensorFlow)
  * Includes pre-built libraries of OpenVINO™ version 2022.1. The users do not have to install OpenVINO™ separately 
  * Supports Intel<sup>®</sup> CPUs, Intel<sup>®</sup> integrated GPUs, and Intel<sup>®</sup> Movidius™ Vision Processing Units (VPUs). No VAD-M support
  * TensorFlow wheel for Windows from PyPi does't have all the API symbols enabled which are required for **OpenVINO™ integration with TensorFlow**. User needs to install the  TensorFlow wheel from the assets of the Github release page.
  
<br/>  

## <a name='Prebuiltpackagessummary'></a>Pre-built packages summary
  
|TensorFlow Pip Package| **OpenVINO™ integration with TensorFlow** Pip Package|Supported OpenVINO™ Flavor|Supported Hardwares|Comments|
| -----------------|-----------------------------------|----------------------------|---------------------------|----------------|
|tensorflow| openvino-tensorflow|OpenVINO™ built from source|CPU,iGPU,MYRIAD|**OpenVINO™** libraries are built from source and included in the wheel package|
|tensorflow-abi1| openvino-tensorflow-abi1|Dynamically links to OpenVINO™ binary release|CPU,iGPU,MYRIAD,VAD-M|**OpenVINO™ integration with TensorFlow** libraries are dynamically linked to **OpenVINO™** binaries|
<br/>  

##  1.1. <a name='InstallOpenVINOintegrationwithTensorFlowalongsidePyPiTensorFlow'></a>Install **OpenVINO™ integration with TensorFlow** alongside PyPi TensorFlow (Works on Linux, macOS)

        pip3 install -U pip
        pip3 install tensorflow==2.8.0
        pip3 install openvino-tensorflow==2.0.0
<br/> 

##  1.2. <a name='InstallOpenVINOintegrationwithTensorFlowalongsideTensorFlow'></a>Install **OpenVINO™ integration with TensorFlow** alongside TensorFlow released on Github (Works on Windows)

        pip3.9 install -U pip
        pip3.9 install https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v2.0.0.dev20220224/tensorflow-2.8.0-cp39-cp39-win_amd64.whl
        pip3.9 install openvino-tensorflow==2.0.0
<br/> 

##  1.3. <a name='InstallOpenVINOintegrationwithTensorFlowalongsidetheIntelDistributionofOpenVINOToolkit'></a>Install **OpenVINO™ integration with TensorFlow** alongside the Intel® Distribution of OpenVINO™ Toolkit (Works on Linux)

1. Ensure the following versions are being used for pip and numpy:

        pip3 install -U pip
        pip3 install numpy==1.20.2

2. Install `TensorFlow` based on your Python version. You can build [TensorFlow from source](https://github.com/openvinotoolkit/openvino_tensorflow/blob/master/docs/BUILD.md#tensorflow) with -D_GLIBCXX_USE_CXX11_ABI=1  or follow the insructions below to use the appropriate package:

        pip3.7 install https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v2.0.0.dev20220224/tensorflow_abi1-2.8.0-cp37-cp37m-manylinux_2_12_x86_64.manylinux2010_x86_64.whl

        or

        pip3.8 install https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v2.0.0.dev20220224/tensorflow_abi1-2.8.0-cp38-cp38m-manylinux_2_12_x86_64.manylinux2010_x86_64.whl

        or

        pip3.9 install https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v2.0.0.dev20220224/tensorflow_abi1-2.8.0-cp39-cp39m-manylinux_2_12_x86_64.manylinux2010_x86_64.whl

3. Download & install Intel® Distribution of OpenVINO™ Toolkit 2022.1 release along with its dependencies from ([https://software.intel.com/en-us/openvino-toolkit/download](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit/download.html)).

4. Initialize the OpenVINO™ environment by running the `setupvars.sh` located in <code>\<openvino\_install\_directory\>\/bin</code> using the command below:

        source setupvars.sh

5. Install `openvino-tensorflow`. Based on your Python version, choose the appropriate package below:

        pip3.7 install https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v1.1.0/openvino_tensorflow_abi1-1.1.0-cp37-cp37m-linux_x86_64.whl

        or

        pip3.8 install https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v1.1.0/openvino_tensorflow_abi1-1.1.0-cp38-cp38-linux_x86_64.whl

        or

        pip3.9 install https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v1.1.0/openvino_tensorflow_abi1-1.1.0-cp39-cp39-linux_x86_64.whl


