### Build and run nGraph in Docker

To run nGraph in Docker, choose one of two ways to create your container:
  1. Use the [`docker_build_and_install_ngtf.sh`](docker_build_and_install_ngtf.sh) script to do a multi-stage build and run nGraph for Ubuntu 18.04 in a single command. 
     This will perform all of the build steps automatically in an intermediate container and provide a final image without all the tools needed to build Tensorflow and nGraph. 
  2. Use `Dockerfile.ubuntu.18.04` by itself to set up a build environment that you can use to then manually build Tensorflow, nGraph, and the bridge in a virtualenv. 

##### Method 1:

- Clone the `ngraph-bridge` repo:
  
        git clone https://github.com/tensorflow/ngraph-bridge.git
  
- Navigate into the `tools` directory and run the installation script:
  
        cd ngraph-bridge/tools
        . docker_build_and_install_ngtf.sh

  If you want to use build options such as `--use_prebuilt_tensorflow` or `--use_grappler_optimizer`, set an input argument when running the installation script.

        . docker_build_and_install_ngtf.sh '--use_prebuilt_tensorflow --use_grappler_optimizer'

  For more information about build options, see [here](/build_ngtf.py).
  There may be some build options not supported with this method, so if your customized build is failing, **Method 2** is recommended. 
  
- When the multi-stage docker build is complete, you will be able to run a container with Tensorflow and nGraph using the `ngraph-bridge:ngtf` image:

        docker run -it --name ngtf ngraph-bridge:ngtf
        
  Note: If running behind a proxy, you will need to set `-e http_proxy=<http_proxy>` and `-e https_proxy=<https_proxy>` variables in order to run the test script.

- After running the container, you can perform an inference test by running: 

        python examples/infer_image.py
  
##### Method 2:

- Clone the `ngraph-bridge` repo:

        git clone https://github.com/tensorflow/ngraph-bridge.git

- Navigate into the `tools` directory and build the dockerfile:

        cd ngraph-bridge/tools
        docker build -t ngraph-bridge:devel -f=Dockerfile.ubuntu18.04 .

- Navigate up one level and run the image with the ngraph-bridge project mounted to `/workspace`:  

        cd ..
        docker run -it -v ${PWD}:/workspace -w /workspace --name ngtf ngraph-bridge:devel

- Follow the instructions in [Build an nGraph bridge](/README.md#build-an-ngraph-bridge) to execute `python3 build_ngtf.py`.
  You do not need to clone the repo inside the container because it is already mounted to `/workspace`.
  The mounted volume allows you to access the build artifacts (`whl` files) outside the container if you wish to do so.
  
- After the build completes, you will be able to use the virtualenv located at `/workspace/build_cmake/venv-tf-py3` and run a test.

        source build_cmake/venv-tf-py3/bin/activate
        python examples/infer_image.py