# nGraph As a Device

This is an experimental project to connect nGraph with TensorFlow as a TensorFlow device (such as a CPU or a GPU)
Currently nGraph is added an optimizer for the CPU in which computation nodes are replaced by one or more nGraph 
encapsulatr nodes. However, transferring the tensors between nGraph device and the host CPU involves several 
layers of book keeping.

In this experimental project, nGraph will be registered as NGRAPH device and will take dvantages of the 
TensorFlow placer. 

Earlier versionb of the nGraph bridge used the device registration mechanism. Those changes
are available in the following commit and prior:

Commit hash: 7cf40f3e3519bb4e507ab3791e46cfa6b891f9b8
