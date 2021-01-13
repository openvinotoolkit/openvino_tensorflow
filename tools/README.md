# TOOLS

## Build and run nGraph-TF bridge in Docker

To run nGraph in Docker, choose one of two ways to create your container:
  1. Use [`docker-build.sh`](docker-build.sh) to build two docker images with tags:
    - `devel-<version>` that serves as a development image which includes the prerequisites and the source
    - `<release>` that includes an installation of the bridge and itsruntime dependencies
  2. Use `Dockerfile.ubuntu.18.04` by itself to set up a build environment that you can use to then manually build Tensorflow, nGraph, and the bridge in a virtualenv. 

##### Method 1:

- Clone the `ngraph-bridge` repo:
  
        git clone https://github.com/tensorflow/ngraph-bridge.git
  
- Navigate into the `tools` directory and run the installation script:
  
        cd ngraph-bridge/tools
        ./docker-build.sh

  If you want to use build options such as `--use_prebuilt_tensorflow` or `--use_grappler_optimizer`, set an input argument when running the installation script.

        ./docker-build.sh '--use_prebuilt_tensorflow --disable_cpp_api'

  For more information about build options, see [here](/build_ngtf.py).
  There may be some build options not supported with this method, so if your customized build is failing, **Method 2** is recommended. 
  
- When the multi-stage docker build is complete, you will be able to run a container with Tensorflow and nGraph using the `ngraph-bridge` image. For instance:

        docker run -it ngraph-bridge:0.23
        
  Note: If running behind a proxy, you will need to set `-e http_proxy=<http_proxy>` and `-e https_proxy=<https_proxy>` variables in order to run the test script.

- After running the container, you can perform an inference test by running: 

        python examples/infer_image.py
  
##### Method 2:

- Clone the `ngraph-bridge` repo:

        git clone https://github.com/tensorflow/ngraph-bridge.git

- Navigate into the `tools` directory and build the devel image:

        cd ngraph-bridge/tools
        docker build -t ngraph-bridge:devel -f=Dockerfile.ubuntu18.04.devel .

- Build the install image:

        docker build --build-arg base_image=ngraph-bridge:devel -t ngraph-bridge:master -f=Dockerfile.ubuntu18.04.install .
  
- After the build completes, you will be able to run applications using the container:

        python examples/infer_image.py
