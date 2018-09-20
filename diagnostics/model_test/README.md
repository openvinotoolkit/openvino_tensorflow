# Compare model output between Tensorflow and NGraph

### This model_test tool will run the model inference seperately on TF and NGraph, and the desired output from TF and NGraph should match given the same inputs. It can be used as a debugging tool, and also a verification that NGraph produces the same output as Tensorflow. 

# Required files to use the tool:
* A json file: Provide model specific parameters. Look at the example ```mnist_cnn.json```. You can start with the ```template.json``` and modify it to match your model
* A tensorflow frozen graph: A model frozen graph(.pb file) with trained weights and model architecture

## To prepare the required json file:"
* You will need the names of the input/output tensors of the model. Currently we are supporting
multiple input tensors and one output tensor. Put the input tensor names as a list in the ```input_tensor_name``` field of the json file, and the output tensor name as a string in the ```output_tensor_name``` field of the json file
* You will need the input dimensions for all the input tensors provided. Put the dimensions information as a list in the ```input_dimension``` field of the json file, and the corresponding order of ```input_tensor_name``` list should match the ```input_dimension``` list. Therfore, the length of ```input_tensor_name``` list should match the length of ```input_dimension``` list
* Specify the the location of the frozen graph in the ```frozen_graph_location``` of the json file
* Specify the ```batch_size``` field in the json file to the desired batch size for inference
* Specify the tolerance between the TF and NGraph outputs at ```l1_norm_threshold```, ```l2_norm_threshold``` and ```inf_norm_threshold``` in the json file 

# To run the model test tool:
	python verify_model.py --json_file="/path/to/your/json/file"

# Result Metrics
### The model_test tool will run the model inference and compare the outputs from TF and NGraph in terms of L1, L2 and Inf norm. If the corresponding norm is smaller than the matching tolerance specified in the json file, then the test passes. Otherwise, the test failes. In the situation of test failure, feel free to report the problem at the ngraph-tf github issue section.