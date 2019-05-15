# A simple makefile that builds the hello_tf example

# Location of the ngraph-tf directory
ARTIFACTS_DIR=../../build_cmake/artifacts
TF_PY_DIR=../../build_cmake/venv-tf-py3/lib/python3.5/site-packages/tensorflow
TF_SRC_DIR=../../build_cmake/tensorflow

define GET_TF_CXX_ABI
import tensorflow as tf
print (" ".join(str(tf.__cxx11_abi_flag__)))
endef

ifneq ("$(wildcard /etc/redhat-release)","")
LIB_DIR := lib64
else
LIB_DIR := lib
endif

# Set the CXX_ABI_FLAG based on what is used to build TensorFlow that is being used
CXX_ABI_FLAG := $(shell python3 -c '$(GET_TF_CXX_ABI)')
$(info CXX_ABI_FLAG $(CXX_ABI_FLAG))

CXXFLAGS=-std=c++11 -Wall -pedantic -g -o0\
	-I $(ARTIFACTS_DIR)/include \
	-I $(TF_PY_DIR)/include/ \
	-I $(TF_SRC_DIR)/ \
	-I $(TF_SRC_DIR)/bazel-genfiles \
	-D_GLIBCXX_USE_CXX11_ABI=$(CXX_ABI_FLAG)

ODIR=obj
INCDIR = include
LD_FLAGS=\
	-L $(ARTIFACTS_DIR)/$(LIB_DIR) \
	-L $(ARTIFACTS_DIR)/tensorflow 

BIN = hello_tf

LIBS=\
	-ltensorflow_framework \
	-ltensorflow_cc \
	-lngraph \
	-lngraph_bridge \
	-lpthread


SRC=$(wildcard *.cpp)
OBJ=$(SRC:%.cpp=%.o)

all: $(OBJ)
	$(CXX) $^  $(LD_FLAGS)  $(LIBS) -o $(BIN)

%.o: %.c
	$(CXX) $@ -c $<

.PHONY: clean

clean:
	rm -f *.o
	rm $(BIN)
	