#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename T>
class NGraphOp : public OpKernel {
 public:
  explicit NGraphOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    std::cout << "NGraphOp::Compute" << std::endl;
    // // Grab the input tensor
    // const Tensor& input_tensor = context->input(0);

    // // Create an output tensor
    // Tensor* output_tensor = NULL;
    // OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
    //                                                  &output_tensor));

    // // Do the computation.
    // OP_REQUIRES(context, input_tensor.NumElements() <= tensorflow::kint32max,
    //             errors::InvalidArgument("Too many elements in tensor"));
    // ExampleFunctor<Device, T>()(
    //     context->eigen_device<Device>(),
    //     static_cast<int>(input_tensor.NumElements()),
    //     input_tensor.flat<T>().data(),
    //     output_tensor->flat<T>().data());
  }
};

class NGraphNoOp : public OpKernel {
 public:
  explicit NGraphNoOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    std::cout << "Step: " << context->step_id()
              << " Op: " << context->op_kernel().name() << std::endl;
    // // Grab the input tensor
    // const Tensor& input_tensor = context->input(0);

    // // Create an output tensor
    // Tensor* output_tensor = NULL;
    // OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
    //                                                  &output_tensor));

    // // Do the computation.
    // OP_REQUIRES(context, input_tensor.NumElements() <= tensorflow::kint32max,
    //             errors::InvalidArgument("Too many elements in tensor"));
    // ExampleFunctor<Device, T>()(
    //     context->eigen_device<Device>(),
    //     static_cast<int>(input_tensor.NumElements()),
    //     input_tensor.flat<T>().data(),
    //     output_tensor->flat<T>().data());
  }
};

// This form allows you to specify a list of types as the constraint.
REGISTER_KERNEL_BUILDER(
    Name("Add").Device("NGRAPH_CPU").TypeConstraint("T", {DT_FLOAT}),
    NGraphOp<float>);

REGISTER_KERNEL_BUILDER(Name("NoOp").Device("NGRAPH_CPU"), NGraphNoOp);
REGISTER_KERNEL_BUILDER(Name("Placeholder").Device("NGRAPH_CPU"), NGraphNoOp);
REGISTER_KERNEL_BUILDER(Name("_Recv").Device("NGRAPH_CPU"), NGraphNoOp);
REGISTER_KERNEL_BUILDER(Name("_Send").Device("NGRAPH_CPU"), NGraphNoOp);
