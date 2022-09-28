<h1 align="center">📚 OpenVINO™ integration with TensorFlow Notebooks</h1>

Welcome to our collection of ready-to-run Jupyter notebooks that helps you to quickly try out **OpenVINO™ integration with TensorFlow**. The notebooks carry out popular deep learning tasks like Image Classification, and Object Detection in TensorFlow and demonstrate it to developers how to leverage our simple two-liner API for optimized deep learning inference, without ever leaving the TensorFlow and Python ecosystem.

The notebooks can be run on an **Intel CPU running a supported version of the Ubuntu OS (currently 18.04 or 20.04)**. We recommend a Python virtual environment to start the Jupyter server.

### 1. Install Python, Git, and GPU drivers (optional)

You may need to install some additional libraries on Ubuntu Linux. These steps work on a clean install of Ubuntu Desktop 20.04, and should also work on Ubuntu 18.04 and 20.10, and on Ubuntu Server.

	sudo apt-get update
	sudo apt-get upgrade
	sudo apt-get install python3-venv build-essential python3-dev git-all

If you have a CPU with an Intel Integrated Graphics Card, you can install the [Intel Graphics Compute Runtime](https://github.com/intel/compute-runtime) to enable inference on this device. The command for Ubuntu 20.04 is:

Note: Only execute this command if you do not yet have OpenCL drivers installed.

	sudo apt-get install intel-opencl-icd

### 2. Create and activate the virtual environment

First, let's clone this repo to get access to the notebooks

	git clone https://github.com/openvinotoolkit/openvino_tensorflow
	cd openvino_tensorflow

Now, create a Python virtual environment and activate it

	python3 -m venv openvino_tensorflow_env
	source openvino_tensorflow_env/bin/activate

### 3. Launch the Notebooks!

Let's install JupyterLab

	python3 -m pip install jupyterlab

To launch a single notebook, like the TFHub Object Detection notebook

	jupyter notebook examples/notebooks/OpenVINO_TensorFlow_TFHub_Object_Detection.ipynb

## (Optional) Run these notebooks on Docker

Alternatively, if you want to skip a local setup and want a stable runtime consider our [docker instructions for runtime images](https://github.com/openvinotoolkit/openvino_tensorflow/tree/master/docker). The images tagged `latest` start a Jupyter server by default.