<p>English | <a href="./INSTALL_cn.md">简体中文</a></p>

# <a name='Pre-BuiltPackages'></a>Installation using Pre-Built Packages

**OpenVINO™ integration with TensorFlow** is released for Linux, macOS, and Windows. You can choose one of the following methods based on your requirements.


## Linux

  ### Install **OpenVINO™ integration with TensorFlow** PyPi release
  * Includes pre-built libraries of OpenVINO™ version 2022.1.0. The users do not have to install OpenVINO™ separately 
  * Supports Intel<sup>®</sup> CPUs, Intel<sup>®</sup> integrated GPUs, and Intel<sup>®</sup> Movidius™ Vision Processing Units (VPUs). No VAD-M support 

        pip3 install -U pip
        pip3 install tensorflow==2.9.1
        pip3 install openvino-tensorflow==2.1.0
    The openvino-tensorflow PyPi package is cross-compatible with PATCH versions of TensorFlow. For example, openvino-tensorflow wheel for TF 2.9.1 would work with any future PATCH versions like TF 2.9.2, and 2.9.3
  <br/>  
 
  ### Install **OpenVINO™ integration with TensorFlow** PyPi release alongside the Intel® Distribution of OpenVINO™ Toolkit for VAD-M Support
  * Compatible with OpenVINO™ version 2022.1.0
  * Supports Intel<sup>®</sup> Vision Accelerator Design with Movidius™ (VAD-M), it also supports Intel<sup>®</sup> CPUs, Intel<sup>®</sup> integrated GPUs and Intel<sup>®</sup> Movidius™ Vision Processing Units (VPUs)
  * To use it:
    1. Install tensorflow and openvino-tensorflow packages from PyPi as explained in the section above
    2. Download & install Intel® Distribution of OpenVINO™ Toolkit 2022.1.0 release along with its dependencies from ([https://software.intel.com/en-us/openvino-toolkit/download](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit/download.html)).
    3. Initialize the OpenVINO™ environment by running the `setupvars.sh` located in <code>\<openvino\_install\_directory\>\/bin</code> using the command below. This step needs to be executed in the same environment which is used for the TensorFlow model inference using openvino-tensorflow.

        source setupvars.sh  
      
  
## macOS

  Install **OpenVINO™ integration with TensorFlow** PyPi release
  * Includes pre-built libraries of OpenVINO™ version 2022.1.0. The users do not have to install OpenVINO™ separately 
  * Supports Intel<sup>®</sup> CPUs, Intel<sup>®</sup>, and Intel<sup>®</sup> Movidius™ Vision Processing Units (VPUs). No VAD-M support

        pip3 install -U pip
        pip3 install tensorflow==2.9.1
        pip3 install openvino-tensorflow==2.1.0


## Windows

  Install **OpenVINO™ integration with TensorFlow** PyPi release alongside TensorFlow released in Github
  * TensorFlow wheel for Windows from PyPi does't have all the API symbols enabled which are required for **OpenVINO™ integration with TensorFlow**. User needs to install the TensorFlow wheel from the assets of the Github release page
  * Includes pre-built libraries of OpenVINO™ version 2022.1.0. The users do not have to install OpenVINO™ separately 
  * Supports Intel<sup>®</sup> CPUs, Intel<sup>®</sup> integrated GPUs, and Intel<sup>®</sup> Movidius™ Vision Processing Units (VPUs). No VAD-M support

        pip3.9 install -U pip
        pip3.9 install https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v2.1.0/tensorflow-2.9.1-cp39-cp39-win_amd64.whl
        pip3.9 install openvino-tensorflow==2.1.0
  
