# Compare model output between two different backends

This model_test tool will run the model inference seperately on any two specified backends(e.g. Backends: Tensorflow (native), nGraph-CPU, nGraph-GPU etc) in json file and the outputs from each backend should match given the same inputs. It can be used as a debugging tool for layer by layer comparison, and also a verification that nGraph produces the same output as Tensorflow. 

# Required files to use the tool:
* A json file: Provide model specific parameters. Look at the example ```mnist_cnn.json```. You can start with the ```template.json``` and modify it to match your model
* A tensorflow model file: Either .pb or .pbtxt file

## To prepare the required json file:
* Specify the ```reference_backend``` and ```testing_backend```. For tensolrflow on CPU, use 'CPU' and for nGraph, use 'NGRAPH_[desired backend name]' (e.g. Use 'NGAPH_CPU' for nGraph on CPU)
* You will need the names of the input/output tensors of the model. Currently we are supporting
multiple input tensors and output tensors. Put the input tensor names as a list in the ```input_tensor_name``` field of the json file, and the output tensor name as a list in the ```output_tensor_name``` field of the json file. If no outputs are specified in the ```output_tensor_name```, then it will compare all output tensors
* You will need the input dimensions for all the input tensors provided. Put the dimensions information as a list in the ```input_dimension``` field of the json file, and the corresponding order of ```input_tensor_name``` list should match the ```input_dimension``` list. Therfore, the length of ```input_tensor_name``` list should match the length of ```input_dimension``` list
* Specify the the location of the graph file in the ```graph_location``` of the json file
* Specify the ```batch_size``` field in the json file to the desired batch size for inference
* Specify the tolerance between the TF and NGraph outputs at ```l1_norm_threshold```, ```l2_norm_threshold``` and ```inf_norm_threshold``` in the json file 
* Specify the ```random_val_range``` used to generate the input within 0 to random_val_range

# To run the model test tool:
	python verify_model.py --json_file="/path/to/your/json/file"

# Result Metrics
The model_test tool will run the model inference and compare the outputs in terms of L1, L2 and Inf norm. If the corresponding norm is smaller than the matching tolerance specified in the json file, then the test passes. Otherwise, the test fails. Each output tensors will be saved as .npy file in a folder named as '[reference_backend_name]-[testing_backend_name]'(e.g. CPU-NGRAPH_CPU). In case of test failure, feel free to report the problem at the ngraph-tf github issue section.
