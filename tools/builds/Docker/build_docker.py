import os
import subprocess
from bs4 import BeautifulSoup


#Creates Docker Files for each base image and for each combination
base_docker_files=os.listdir("./docker_files")
print(base_docker_files)
success_build={}
success_test={}
for bdf in base_docker_files:
    Soup = BeautifulSoup(open('build_options.xml','r'),'lxml')
    for bo in Soup.find_all(['item']):
        if os.path.exists(bdf+"_"+bo.find(['name']).text):
            os.remove(bdf+"_"+bo.find(['name']).text)
        # Use the name of approprite docker file
        #eg for ubuntu20 use Dockerfile.ubuntu20.04_base    
        base = open("./docker_files/"+bdf, "r")
        temp = open(bdf+"_"+bo.find(['name']).text,"w")
        for line in base:
            temp.write(line)
        for line in bo.find_all(['command']):
            temp.write("RUN "+line.text+"\n")
        temp.close()
        base.close()

# # # If proxy settings are detected in the environment, make sure they are
# # # included on the docker-build command-line.  This mirrors a similar system
# # # in the Makefile.
DOCKER_HTTP_PROXY=''
DOCKER_HTTPS_PROXY=''
DOCKER_FTP_PROXY=''
DOCKER_NO_PROXY=''
DOCKER_RUN_HTTP_PROXY=''
DOCKER_RUN_HTTPS_PROXY=''
DOCKER_RUN_FTP_PROXY=''
DOCKER_RUN_NO_PROXY=''

if os.environ.__contains__('http_proxy'):
    DOCKER_HTTP_PROXY="--build-arg http_proxy="+os.environ['http_proxy']
    DOCKER_RUN_HTTP_PROXY = "--env http_proxy="+os.environ['http_proxy']

if os.environ.__contains__('https_proxy'):
    DOCKER_HTTPS_PROXY="--build-arg https_proxy="+os.environ['https_proxy']
    DOCKER_RUN_HTTPS_PROXY="--env https_proxy="+os.environ['https_proxy']

if os.environ.__contains__('ftp_proxy'):
    DOCKER_FTP_PROXY="--build-arg ftp_proxy="+os.environ['ftp_proxy']
    DOCKER_RUN_FTP_PROXY="--env ftp_proxy="+os.environ['ftp_proxy']

if os.environ.__contains__('no_proxy'):
    DOCKER_NO_PROXY="--build-arg no_proxy="+os.environ['no_proxy']
    DOCKER_RUN_NO_PROXY="--env no_proxy="+os.environ['no_proxy']

#Run docker files
for bdf in base_docker_files:
    for bo in Soup.find_all(['item']):
        command = ("docker build --rm %s %s %s %s -t %s:latest -f %s  ." %(DOCKER_HTTP_PROXY,DOCKER_HTTPS_PROXY,DOCKER_NO_PROXY,DOCKER_FTP_PROXY,bdf+"_"+bo.find(['name']).text,bdf+"_"+bo.find(['name']).text))
        #print(command)
        output=subprocess.run(command,shell=True,stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if os.path.exists(bdf+"_"+bo.find(['name']).text+".log"):
            os.remove(bdf+"_"+bo.find(['name']).text+".log")
        logging = open(bdf+"_"+bo.find(['name']).text+".log","w")
        logging.write("**************STDOUT**********************\n")
        logging.write(output.stdout.decode('UTF-8'))
        logging.write("**************STDERR**********************\n")
        logging.write(output.stderr.decode('UTF-8'))
        logging.close()
        success = "Success" if output.returncode==0 else "Failure"
        success_build[bdf+"_"+bo.find(['name']).text]=success
        name_of_the_image = bdf+"_"+bo.find(['name']).text+":latest"
        if success=="Success" and 'source' in name_of_the_image:
            command_run_test = ''
            if 'no' in bdf+"_"+bo.find(['name']).text:
                command_run_test = ('docker run  %s %s %s %s %s   /bin/bash -c "cd /opt/intel/openvino_tensorflow/examples/data && wget https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz && tar -xzvf inception_v3_2016_08_28_frozen.pb.tar.gz && source /opt/intel/openvino_tensorflow/build_cmake/venv-tf-py3/bin/activate && cd /opt/intel/openvino_tensorflow && python3 ./examples/classification_sample.py"'%(DOCKER_RUN_FTP_PROXY,DOCKER_RUN_HTTP_PROXY,DOCKER_RUN_HTTPS_PROXY,DOCKER_RUN_NO_PROXY,name_of_the_image))
            else :
                command_run_test = ('docker run  %s %s %s %s %s   /bin/bash -c "source /opt/intel/openvino_2021.3.394/bin/setupvars.sh && cd /opt/intel/openvino_tensorflow/examples/data && wget https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz && tar -xzvf inception_v3_2016_08_28_frozen.pb.tar.gz && source /opt/intel/openvino_tensorflow/build_cmake/venv-tf-py3/bin/activate && cd /opt/intel/openvino_tensorflow && python3 ./examples/classification_sample.py"'%(DOCKER_RUN_FTP_PROXY,DOCKER_RUN_HTTP_PROXY,DOCKER_RUN_HTTPS_PROXY,DOCKER_RUN_NO_PROXY,name_of_the_image))
            #print(command_run_test)
            run_test = subprocess.run(command_run_test,shell=True,stdout=subprocess.PIPE, stderr=subprocess.PIPE)#,stdout=subprocess.PIPE, stderr=subprocess.PIPE
            logging = open(bdf+"_"+bo.find(['name']).text+"_test.log","w")
            logging.write("**************STDOUT**********************\n")
            logging.write(run_test.stdout.decode('UTF-8'))
            logging.write("**************STDERR**********************\n")
            logging.write(run_test.stderr.decode('UTF-8'))
            logging.close()
            success_test[bdf+"_"+bo.find(['name']).text]="Success" if run_test.returncode==0 else "Failure"

if os.path.exists("overallstatus.txt"):
    os.remove("overallstatus.txt")
over_all_status = open("overallstatus.txt","w")
over_all_status.write("***************Builds**************\n")
for key in success_build:
    over_all_status.write(str(key)+":"+str(success_build[key])+"\n")
over_all_status.write("***************Test**************\n")
for key in success_test:
    over_all_status.write(str(key)+":"+str(success_test[key])+"\n")
over_all_status.close()





