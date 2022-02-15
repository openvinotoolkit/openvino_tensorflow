Build the docker image using the Dockerfile.ubuntu using the command below:

docker build --build-arg BASE_IMAGE=ubuntu:<18.04/20.04> --build-arg VERSION=<github branch> -t <image name>:<image tag> . -f <path to Dockerfile.ubuntu>

It will take 2 - 6 hours of time, depending on your build type.

Once the image is ready, please execute the command below to start the container.
docker run -p 8888:8888 <image name>:<image tag> 
