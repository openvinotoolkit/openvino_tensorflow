# Verify model accuracy by running the models using ngraph
This model_accuracy tool will run inference for Image Recognition models - Inception_v4, ResNet50_v1, Mobilenet_v1 using the official tensorflow models repo https://github.com/tensorflow/models
After running the models using ngraph, the tool validates the accuracy by comparing with the known accuracy from published papers.

# Required setup to use the tool:
ngraph_bridge should be installed to be able to use this tool. 
Refer to Option#2 here on how to build and install - https://github.com/NervanaSystems/ngraph-tf/blob/master/README.md


# To run the model accuracy tool(example):
	python verify_inference_model.py --model_name Inception_v4

# Limitations:
The tool can run only one network at a time now, will be extended to run multiple models at once and validate accuracy of the same 
