import os
import subprocess


build_options={
    "ubuntu_pip_yes":["pip3.6 install https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v0.5.0/tensorflow_abi1-2.4.1-cp36-cp36m-manylinux2010_x86_64.whl","pip3.6 install https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v0.5.0/openvino_tensorflow_abi1-0.5.0-cp36-cp36m-linux_x86_64.whl"],
    "ubuntu_pip_no":["pip3 install -U pip==21.0.1","pip3 install -U tensorflow==2.4.1","pip3 install openvino-tensorflow"],
    "ubuntu_source_yes":["git clone https://github.com/openvinotoolkit/openvino_tensorflow.git /opt/intel/openvino_tensorflow","cd /opt/intel/openvino_tensorflow && git submodule init","cd /opt/intel/openvino_tensorflow  && git submodule update --recursive","cd /opt/intel/openvino_tensorflow && python3 build_ovtf.py --use_openvino_from_location=/opt/intel/openvino_2021.3.394/ --cxx11_abi_version=1"],
    "ubuntu_source_no":["git clone https://github.com/openvinotoolkit/openvino_tensorflow.git /opt/intel/openvino_tensorflow","cd /opt/intel/openvino_tensorflow && git submodule init","cd /opt/intel/openvino_tensorflow  && git submodule update --recursive","cd /opt/intel/openvino_tensorflow && python3 build_ovtf.py"],
    "tensorflow_ubuntu_pip_no":["pip3 install tensorflow","pip3 install openvino-tensorflow"]
    }
success_build={}
for bo in build_options:
    if os.path.exists(bo):
        os.remove(bo)
    base = open("Dockerfile", "r")
    temp = open(bo,"w")
    for line in base:
        temp.write(line)
    for line in build_options[bo]:
        temp.write("RUN "+line+"\n")
    temp.close()

# If proxy settings are detected in the environment, make sure they are
# included on the docker-build command-line.  This mirrors a similar system
# in the Makefile.
DOCKER_HTTP_PROXY=''
DOCKER_HTTPS_PROXY=''
DOCKER_FTP_PROXY=''
DOCKER_NO_PROXY=''

if os.environ.__contains__('http_proxy'):
    DOCKER_HTTP_PROXY="--build-arg http_proxy="+os.environ['http_proxy']

if os.environ.__contains__('https_proxy'):
    DOCKER_HTTPS_PROXY="--build-arg https_proxy="+os.environ['https_proxy']

if os.environ.__contains__('ftp_proxy'):
    DOCKER_FTP_PROXY="--build-arg ftp_proxy="+os.environ['ftp_proxy']

if os.environ.__contains__('no_proxy'):
    DOCKER_NO_PROXY="--build-arg no_proxy="+os.environ['no_proxy']

for bo in build_options:
    command = ("docker build --pull --rm %s %s %s %s -t %s:latest -f %s  ." %(DOCKER_HTTP_PROXY,DOCKER_HTTPS_PROXY,DOCKER_NO_PROXY,DOCKER_FTP_PROXY,bo,bo))
    print(command)
    output=subprocess.run(command,shell=True,stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if os.path.exists(bo+".log"):
        os.remove(bo+".log")
    logging = open(bo+".log","w")
    logging.write("**************STDOUT**********************\n")
    logging.write(output.stdout.decode('UTF-8'))
    logging.write("**************STDERR**********************\n")
    logging.write(output.stderr.decode('UTF-8'))
    logging.close()
    success = "Success" if output.returncode==0 else "Failure"
    success_build[bo]=success

if os.path.exists("overallstatus.txt"):
    os.remove("overallstatus.txt")
over_all_status = open("overallstatus.txt","w")
over_all_status.write(str(success_build))
over_all_status.close()





