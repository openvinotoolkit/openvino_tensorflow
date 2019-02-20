# Build nGraph TensorFlow bridge using bazel

This directory contains scripts necessary to build the nGraph TensorFlow bridge using `bazel`. 

:warning: This is experimental and will change over time. 

## Prerequisites

Please ensure that bazel and Python is installed on your system and you are able to build TensorFlow from source (though not needed for building the bridge). Please see the [build preperation] for details.

## Build C++ library

Go to the ngraph-tf directory and execute these commands to build the C++ library for nGraph-TensorFlow bridge:

        ./configure_bazel.sh
        bazel build libngraph_bridge.so
        bazel build @ngraph//:libinterpreter_backend.so   

This will produce the following binary files:

```
    bazel-bin/libngraph_bridge.so
    bazel-bin/external/ngraph/libinterpreter_backend.so
```

### How to use the C++ library

The C++ library `libngraph_bridge.so` can be used with a TensorFlow C++ application as described in the examples/tf_cpp_examples ([TensorFlow C++ example]) directory. Basic steps are the following:

1. Get a copy of the TensorFlow C++ library by building one. Use [Option 2] to build all the necessary libraries as described in the [TensorFlow C++ example]  

2. Replace the `libngraph_bridge.so` built by this bazel script in the `build/artifacts/lib<64>` directory.

3. Run `make` to relink and run the example as described in the [TensorFlow C++ example] document.

**Note** Currently only the INTERPRETER backend may be built using bazel. However other backends built using cmake system is fully binary compatible with the bridge built with bazel based build system as described here.

## Build the Python wheel

Coming up soon. For now please use the cmake based build system described in the [main README]

[build preperation]: ../README.md#prepare-the-build-environment
[Option 2]: ../README.md#option-2-build-ngraph-bridge-from-source
[TensorFlow C++ example]: ../examples/tf_cpp_examples/README.md
[main README]: ../README.md
