# Instructions to accelerate TensorFlow models on Azure with the **OpenVINO™ integration with TensorFlow**

-	Create a Virtual Machine – choose the  Ubuntu Server 20.10 – Gen 2 mage

<p align="center">
 <img src="images/Azure_image_1.png">
</p>

-	Pick an instance  like HC44rs. The bigger the instance the higher the performance 

<p align="center">
<img src="images/Azure_image_2.png">
 </p>

-	Then download the *.pem files for keys on your laptop. 
chmod 400 the *.pem key.  

-	Get the public IP address of your instance 

<p align="center">
<img src="images/Azure_image_3.png">
</p>

-	Login to the Azure instance using SSH  
  ```bash
  $ ssh -i *.pem <IP-addr-of-your-instance>  
  $ scp -i *.pem  <source-file> <IP-addr-of-your-instance>:/tmp
  ```

  SSH and SCP with the Azure instance should be working (Note: It worked seamlessly on our team Intel provided Azure account, but developers might need to configure networking to enable this.) 
  ```bash
  azureuser@tf-u27:~$ python3 --version
  Python 3.8.6
  ```
- Install **OpenVINO™ integration with TensorFlow**
```bash
$ sudo apt-get update

$ sudo apt install python3-pip 

$ sudo pip3 install -U tensorflow==2.5.0

$ sudo pip3 install -U openvino-tensorflow
```

- Verify the installation 
```bash
azureuser@tf-u26:~$ python3 -c "import tensorflow as tf; print('TensorFlow version: ',tf.__version__); import openvino_tensorflow; print(openvino_tensorflow.__version__)" 
TensorFlow version:  2.5.0
OpenVINO integration with TensorFlow version: b'0.5.0'
OpenVINO version used for this build: b'2021.4'
TensorFlow version used for this build: v2.5.0
CXX11_ABI flag used for this build: 0
OpenVINO integration with TensorFlow built with Grappler: False
```


- Clone the repo to run the example
```bash
$ git clone --recursive https://github.com/openvinotoolkit/openvino_tensorflow.git
```

Or just download the entire gitrepo as a zip/tar file to your local directory and scp it to the azure instance and unzip it in your home directory  
You may need to install unzip package to extract the content of the zipped repo
```bash
$ sudo apt install unzip  
```
```bash
azureuser@tf-u26:~$ ls
openvino_tensorflow-master  openvino_tensorflow-master.zip

azureuser@tf-u26:~$ cd openvino_tensorflow-master/

azureuser@tf-u27:~/openvino_tensorflow-master$ curl -L "https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz" | tar -C ./examples/data -xz

  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100 84.5M  100 84.5M    0     0  41.4M      0  0:00:02  0:00:02 --:--:-- 41.4M

azureuser@tf-u27:~/openvino_tensorflow-master$ python3 examples/classification_sample.py 

2021-03-16 18:49:47.205522: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory

2021-03-16 18:49:47.205553: E tensorflow/stream_executor/cuda/cuda_driver.cc:313] failed call to cuInit: UNKNOWN ERROR (303)

2021-03-16 18:49:47.205574: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (tf-u27): /proc/driver/nvidia/version does not exist

2021-03-16 18:49:47.205753: I tensorflow/core/platform/cpu_feature_guard.cc:143] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA

2021-03-16 18:49:47.214779: I tensorflow/core/platform/profile_utils/cpu_utils.cc:102] CPU Frequency: 2693670000 Hz

2021-03-16 18:49:47.219130: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x7f881c000b60 initialized for platform Host (this does not guarantee that XLA will be used). Devices:

2021-03-16 18:49:47.219153: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version

Available Backends:  
CPU  
Inference time in ms: 8.149624  
military uniform 0.8343049  
mortarboard 0.021869553  
academic gown 0.010358133  
pickelhaube 0.008008199  
bulletproof vest 0.0053509558  
```
