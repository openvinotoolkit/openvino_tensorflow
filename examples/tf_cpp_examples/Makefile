# A simple makefile that builds the hello_tf example

# Location of the ngraph-tf directory
NGRAPH_TF_DIR=../../
# Set the CXX_ABI_FLAG to 0 if you are using Clang or gcc 4.8
# Also ensure that TensorFlow is built with the same compiler version. 
CXX_ABI_FLAG=1

CXXFLAGS=-std=c++11 -Wall -pedantic \
	-I $(NGRAPH_TF_DIR)/build/artifacts/include \
	-I $(NGRAPH_TF_DIR)/build/tensorflow/ \
	-I $(NGRAPH_TF_DIR)/build/tensorflow/bazel-genfiles \
	-I $(NGRAPH_TF_DIR)/build/venv-tf-py3/lib/python3.5/site-packages/tensorflow/include \
	-I $(NGRAPH_TF_DIR)/logging \
	-D_GLIBCXX_USE_CXX11_ABI=$(CXX_ABI_FLAG)

ODIR=obj
INCDIR = include
LD_FLAGS=\
	-L $(NGRAPH_TF_DIR)/build/artifacts/lib \
	-L $(NGRAPH_TF_DIR)/build/artifacts/tensorflow 

BIN = hello_tf

LIBS=\
	-ltensorflow_framework \
	-ltensorflow_cc \
	-lngraph \
	-lngraph_bridge 


SRC=$(wildcard *.cpp)
OBJ=$(SRC:%.cpp=%.o)

all: $(OBJ)
	$(CXX) $^  $(LD_FLAGS)  $(LIBS) -o $(BIN)

%.o: %.c
	$(CXX)  $@ -c $<

.PHONY: clean

clean:
	rm -f *.o
	rm $(BIN)
	
