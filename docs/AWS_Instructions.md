[TO BE VALIDATED BY ENGINEERING]

Here are the instructions to accelerate TensorFlow models on AWS with the Intel® OpenVINO(TM) add-on for TensorFlow  


1.	Launch the Deep Learning AMI EC2 instance Ubuntu 18.04 Version 41.0

<p align="center">
  <img src="https://github.com/openvinotoolkit/openvino_tensorflow/blob/arindam-doc-changes-3-17-2011/images/AWS_image_1.png" >
</p>


2.	Choose one of the C5 instances –these are optimized for inference. The larger the instance , the faster the inference. 

<p align="center">
  <img src="https://github.com/openvinotoolkit/openvino_tensorflow/blob/arindam-doc-changes-3-17-2011/images/AWS_Image_2.png" >
</p>


3.	Then download the *.pem files for keys on your laptop. 
chmod 400 the *.pem key.  

4.	Get the public IP address of your instance 

<p align="center">
  <img src="https://github.com/openvinotoolkit/openvino_tensorflow/blob/arindam-doc-changes-3-17-2011/images/AWS_image_3.png" >
</p>


5.	Wait for the instance to finish initializing and be fully running and functional 

6.	ssh -i *.pem <IP-addr-of-your-instance>
a.	scp -i *.pem  <source-file> <IP-addr-of-your-instance>:/tmp

SSH and SCP with the AWS instance should be working (Note: It worked seamlessly on our team Intel provided AWS account, but developers might need to configure networking to enable this.)

7.	git clone  https://github.com/openvinotoolkit/openvino_tensorflow.git

OR just download zip of the whole repo to your local laptop and then scp it to the AWS instance (you might need to scp to :/tmp for permission reasons, and then ssh to the instance then copy the zip to your home directory)

8.	ubuntu@ip-10-0-0-123:~$python3 -m venv myenv

9.	ubuntu@ip-10-0-0-123:~$ source myenv/bin/activate

10.	(myenv) ubuntu@ip-10-0-0-123:~$ pip install --upgrade pip

11.	(myenv) ubuntu@ip-10-0-0-123:~$ pip install tensorflow==2.2.2

12.	(myenv) ubuntu@ip-10-0-0-123:~$ pip  install -U --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ openvino-tensorflow-addon-abi0

13.	(myenv) ubuntu@ip-10-0-0-123:~$ python3 -c "import tensorflow as tf; print('TensorFlow version: ',tf.__version__); import openvino_tensorflow; print(openvino_tensorflow.__version__)" 

TensorFlow version:  2.2.2
openvino tensorflow add-on version: 0.5.0
nGraph version used for this build: b'0.0.0+a8a6e8f'
TensorFlow version used for this build: v2.2.2-1-g876c0a59768
CXX11_ABI flag used for this build: 0
openvino tensorflow add-on built with Grappler: False

14.	(myenv) ubuntu@ip-10-0-0-123:~$ cd openvino_tensorflow-master/

15.	(myenv) ubuntu@ip-10-0-0-123:~/openvino_tensorflow-master$ curl -L "https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz" | tar -C ./examples/data -xz  

% Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100 84.5M  100 84.5M    0     0  29.0M      0  0:00:02  0:00:02 --:--:-- 29.0M


16.	(myenv) ubuntu@ip-10-0-0-123:~/openvino_tensorflow-master$ python3 examples/classification_sample.py 

2021-03-16 23:38:39.107565: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcuda.so.1

2021-03-16 23:38:39.609060: E tensorflow/stream_executor/cuda/cuda_driver.cc:313] failed call to cuInit: CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected

2021-03-16 23:38:39.609158: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (ip-10-0-0-123): /proc/driver/nvidia/version does not exist

2021-03-16 23:38:39.610135: I tensorflow/core/platform/cpu_feature_guard.cc:143] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA

2021-03-16 23:38:39.624320: I tensorflow/core/platform/profile_utils/cpu_utils.cc:102] CPU Frequency: 2999995000 Hz

2021-03-16 23:38:39.627801: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x7f827c000b20 initialized for platform Host (this does not guarantee that XLA will be used). Devices:

2021-03-16 23:38:39.627822: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
Available Backends:

CPU

Inference time in ms: 7.504463

military uniform 0.8343049

mortarboard 0.021869553

academic gown 0.010358133

pickelhaube 0.008008199

bulletproof vest 0.0053509558

