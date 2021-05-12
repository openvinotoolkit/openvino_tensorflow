import os
import subprocess
from bs4 import BeautifulSoup


#Creates Docker Files for each base image and for each combination
base_docker_files=os.listdir("./docker_files")
print(base_docker_files)
success_build={}
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

if os.environ.__contains__('http_proxy'):
    DOCKER_HTTP_PROXY="--build-arg http_proxy="+os.environ['http_proxy']

if os.environ.__contains__('https_proxy'):
    DOCKER_HTTPS_PROXY="--build-arg https_proxy="+os.environ['https_proxy']

if os.environ.__contains__('ftp_proxy'):
    DOCKER_FTP_PROXY="--build-arg ftp_proxy="+os.environ['ftp_proxy']

if os.environ.__contains__('no_proxy'):
    DOCKER_NO_PROXY="--build-arg no_proxy="+os.environ['no_proxy']

#Run docker files
for bdf in base_docker_files:
    for bo in Soup.find_all(['item']):
        command = ("docker build --rm %s %s %s %s -t %s:latest -f %s  ." %(DOCKER_HTTP_PROXY,DOCKER_HTTPS_PROXY,DOCKER_NO_PROXY,DOCKER_FTP_PROXY,bdf+"_"+bo.find(['name']).text,bdf+"_"+bo.find(['name']).text))
        print(command)
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

if os.path.exists("overallstatus.txt"):
    os.remove("overallstatus.txt")
over_all_status = open("overallstatus.txt","w")
over_all_status.write(str(success_build))
over_all_status.close()





