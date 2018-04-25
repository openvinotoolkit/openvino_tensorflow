#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/strings/strcat.h"

#include "tensorflow/core/platform/default/logging.h"

#include "ngraph_utils.h"

//
// NOTE: Check out the Optimizer registration phase
// The macro is: REGISTER_OPTIMIZATION
//
// This is one way to get the graph and do the necessary analysis, fusion etc
// Check out:
// /Users/avijitch/Projects/ngraph-tensorflow/tensorflow/compiler/jit/build_xla_launch_ops_pass.cc
// to see how XLALaunchOps are used. Can we do something similar?

const char* const DEVICE_NGRAPH = "NGRAPH_CPU";

using namespace tensorflow;

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename T>
class NGraphOp : public OpKernel {
 public:
  explicit NGraphOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    auto node_def = ctx->def();
    VLOG(0) << "NGraphOp::ctor(): Node: " << node_def.name()
            << " Op: " << node_def.op();
    // SummarizeOp(ctx, std::cout);
  }

  void Compute(OpKernelContext* ctx) override {
    VLOG(0) << "NGraphOp::Compute() Step: " << ctx->step_id()
            << " Op: " << ctx->op_kernel().name();
    // std::cout << "Step: " << ctx->step_id()
    //           << " Op: " << ctx->op_kernel().name() << std::endl;
    // std::cout << "NGraphOp::Compute" << std::endl;
  }
};

class NGraphNoOp : public OpKernel {
 public:
  explicit NGraphNoOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    auto node_def = ctx->def();
    VLOG(0) << "NGraphNoOp::ctor(): Node: " << node_def.name()
            << " Op: " << node_def.op();
    // SummarizeOp(ctx, std::cout);
  }

  void Compute(OpKernelContext* ctx) override {
    VLOG(0) << "NGraphNoOp::Compute() Step: " << ctx->step_id()
            << " Op: " << ctx->op_kernel().name();
    // std::cout << "Step: " << ctx->step_id()
    //           << " Op: " << ctx->op_kernel().name() << std::endl;
  }
};

class NgPlaceholderOp : public OpKernel {
 public:
  explicit NgPlaceholderOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    auto node_def = ctx->def();
    VLOG(0) << "NgPlaceholderOp::ctor(): Node: " << node_def.name()
            << " Op: " << node_def.op();
    // SummarizeOp(ctx, std::cout);
  }

  void Compute(OpKernelContext* ctx) override {
    VLOG(0) << "NGraphNoOp::Compute() Step: " << ctx->step_id()
            << " Op: " << ctx->op_kernel().name();
    // std::cout << "NgPlaceholderOp: Step: " << ctx->step_id()
    //           << " Op: " << ctx->op_kernel().name() << std::endl;
  }
};

class NGraphConstantOp : public OpKernel {
 public:
  explicit NGraphConstantOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    auto node_def = ctx->def();
    VLOG(0) << "NgPlaceholderOp::ctor(): Node: " << node_def.name()
            << " Op: " << node_def.op();
    // SummarizeNodeDef(def());
  }
  void Compute(OpKernelContext* ctx) override {
    VLOG(0) << "NGraphNoOp::Compute() Step: " << ctx->step_id()
            << " Op: " << ctx->op_kernel().name();
  }
  ~NGraphConstantOp() override {}

 private:
};

#if 0
static string GetRendezvousKeyPrefix(const string& send_device,
                                     const string& recv_device,
                                     const uint64 send_device_incarnation,
                                     const string& tensor_name) {
  return strings::StrCat(send_device, ";",
                         strings::FpToString(send_device_incarnation), ";",
                         recv_device, ";", tensor_name);
}

static void GetRendezvousKey(const string& key_prefix,
                             const FrameAndIter& frame_iter, string* key) {
  key->clear();
  strings::StrAppend(key, key_prefix, ";", frame_iter.frame_id, ":",
                     frame_iter.iter_id);
}

static FrameAndIter GetFrameAndIter(OpKernelContext* ctx,
                                    bool hostmem_sendrecv) {
  if (hostmem_sendrecv && ctx->call_frame() != nullptr) {
    // Host memory send/recv pairs are added by
    // common_runtime/memory_types.cc.  When the pair of nodes are
    // added inside a function, we need to use the function call frame
    // to formulate the unique rendezvous key.
    return FrameAndIter(reinterpret_cast<uint64>(ctx->call_frame()), 0);
  } else {
    return ctx->frame_iter();
  }
}

class NGraphRecv : public AsyncOpKernel {
 public:
  explicit NGraphRecv(OpKernelConstruction* ctx) : AsyncOpKernel(ctx) {
    std::cout << SummarizeNodeDef(def()) << std::endl;
  }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    std::cout << "NGraphRecv: Step: " << ctx->step_id()
              << " Op: " << ctx->op_kernel().name() << std::endl;
  }

 private:
  string key_prefix_;
  Rendezvous::ParsedKey parsed_key_;
  bool hostmem_sendrecv_;
};

class NGraphSend : public OpKernel {
 public:
  explicit NGraphSend(OpKernelConstruction* ctx) : OpKernel(ctx) {
    std::cout << SummarizeNodeDef(def()) << std::endl;

    //--- from TF
    string send_device;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("send_device", &send_device));
    string recv_device;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("recv_device", &recv_device));
    uint64 send_device_incarnation;
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("send_device_incarnation",
                          reinterpret_cast<int64*>(&send_device_incarnation)));
    string tensor_name;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("tensor_name", &tensor_name));
    key_prefix_ = GetRendezvousKeyPrefix(send_device, recv_device,
                                         send_device_incarnation, tensor_name);
    // The vast majority of Send nodes are outside any loop context, so
    // proactively cache the rendezvous key for the top-level.
    GetRendezvousKey(key_prefix_, {0, 0}, &parsed_key_.buf_);
    OP_REQUIRES_OK(ctx, Rendezvous::ParseKey(parsed_key_.buf_, &parsed_key_));
    if (!ctx->GetAttr("_hostmem_sendrecv", &hostmem_sendrecv_).ok()) {
      hostmem_sendrecv_ = false;
    }
  }

  void Compute(OpKernelContext* ctx) override {
    std::cout << "NGraphSend: Step: " << ctx->step_id()
              << " Op: " << ctx->op_kernel().name() << std::endl;
    // - TF
    OP_REQUIRES(
        ctx, ctx->rendezvous() != nullptr,
        errors::Internal("Op kernel context needs to provide a rendezvous."));

    // The device context may be passed between the Send/Recv
    // boundary, so that the device context used to produce the Tensor
    // is used when performing the copy on the recv side (which may be
    // a different device).
    Rendezvous::Args args;
    args.device_context = ctx->op_device_context();
    args.alloc_attrs = ctx->input_alloc_attr(0);

    FrameAndIter frame_iter = GetFrameAndIter(ctx, hostmem_sendrecv_);
    if (frame_iter == FrameAndIter(0, 0)) {
      // Use the cached rendezvous key.
      VLOG(2) << "Send " << parsed_key_.buf_;
      ctx->SetStatus(ctx->rendezvous()->Send(parsed_key_, args, ctx->input(0),
                                             ctx->is_input_dead()));
      return;
    } else {
      Rendezvous::ParsedKey in_loop_parsed;
      GetRendezvousKey(key_prefix_, frame_iter, &in_loop_parsed.buf_);
      VLOG(2) << "Send " << in_loop_parsed.buf_;
      OP_REQUIRES_OK(
          ctx, Rendezvous::ParseKey(in_loop_parsed.buf_, &in_loop_parsed));

      ctx->SetStatus(ctx->rendezvous()->Send(
          in_loop_parsed, args, ctx->input(0), ctx->is_input_dead()));
      return;
    }
  }

 private:
  string key_prefix_;
  Rendezvous::ParsedKey parsed_key_;
  bool hostmem_sendrecv_;
};
#endif

// This form allows you to specify a list of types as the constraint.
REGISTER_KERNEL_BUILDER(
    Name("Add").Device(DEVICE_NGRAPH).TypeConstraint("T", {DT_FLOAT}),
    NGraphOp<float>);
REGISTER_KERNEL_BUILDER(
    Name("Sub").Device(DEVICE_NGRAPH).TypeConstraint("T", {DT_FLOAT}),
    NGraphOp<float>);
REGISTER_KERNEL_BUILDER(
    Name("Mul").Device(DEVICE_NGRAPH).TypeConstraint("T", {DT_FLOAT}),
    NGraphOp<float>);

REGISTER_KERNEL_BUILDER(
    Name("Assign").Device(DEVICE_NGRAPH).TypeConstraint("T", {DT_FLOAT}),
    NGraphOp<float>);
REGISTER_KERNEL_BUILDER(
    Name("ApplyAdam").Device(DEVICE_NGRAPH).TypeConstraint("T", {DT_FLOAT}),
    NGraphOp<float>);

// REGISTER_KERNEL_BUILDER(
//     Name("Const").Device(DEVICE_NGRAPH).TypeConstraint("T", {DT_FLOAT}),
//     NGraphOp<float>);ß
// REGISTER_KERNEL_BUILDER(
//     Name("Const").Device(DEVICE_NGRAPH).TypeConstraint("T", {DT_STRING}),
//     NGraphOp<std::string>);

REGISTER_KERNEL_BUILDER(
    Name("Fill").Device(DEVICE_NGRAPH).TypeConstraint("T", {DT_FLOAT}),
    NGraphOp<float>);

REGISTER_KERNEL_BUILDER(Name("NoOp").Device(DEVICE_NGRAPH), NGraphNoOp);
REGISTER_KERNEL_BUILDER(Name("Placeholder").Device(DEVICE_NGRAPH),
                        NgPlaceholderOp);
REGISTER_KERNEL_BUILDER(Name("PlaceholderV2").Device(DEVICE_NGRAPH),
                        NgPlaceholderOp);
// REGISTER_KERNEL_BUILDER(Name("_Recv").Device(DEVICE_NGRAPH), NGraphNoOp);
// REGISTER_KERNEL_BUILDER(Name("_Send").Device(DEVICE_NGRAPH), NGraphNoOp);

REGISTER_KERNEL_BUILDER(Name("Const").Device(DEVICE_NGRAPH), NGraphNoOp);
REGISTER_KERNEL_BUILDER(Name("IsVariableInitialized").Device(DEVICE_NGRAPH),
                        NGraphNoOp);

REGISTER_KERNEL_BUILDER(Name("Prod")
                            .Device(DEVICE_NGRAPH)
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int32>("Tidx"),
                        NGraphOp<int32>);

REGISTER_KERNEL_BUILDER(Name("ArgMax")
                            .Device(DEVICE_NGRAPH)
                            .TypeConstraint<float>("T")
                            .TypeConstraint<int32>("Tidx"),
                        NGraphOp<float>);

REGISTER_KERNEL_BUILDER(Name("Reshape")
                            .Device(DEVICE_NGRAPH)
                            .TypeConstraint<float>("T")
                            .TypeConstraint<int32>("Tshape"),
                        NGraphOp<float>);

REGISTER_KERNEL_BUILDER(Name("ConcatV2")
                            .Device(DEVICE_NGRAPH)
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int32>("Tidx"),
                        NGraphOp<int32>);

REGISTER_KERNEL_BUILDER(Name("Mean")
                            .Device(DEVICE_NGRAPH)
                            .TypeConstraint<float>("T")
                            .TypeConstraint<int32>("Tidx"),
                        NGraphOp<int32>);

REGISTER_KERNEL_BUILDER(Name("Slice")
                            .Device(DEVICE_NGRAPH)
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int32>("Index"),
                        NGraphOp<int32>);

REGISTER_KERNEL_BUILDER(
    Name("Maximum").Device(DEVICE_NGRAPH).TypeConstraint<int32>("T"),
    NGraphOp<int32>);

REGISTER_KERNEL_BUILDER(
    Name("Sub").Device(DEVICE_NGRAPH).TypeConstraint<int32>("T"),
    NGraphOp<int32>);

REGISTER_KERNEL_BUILDER(Name("Sum")
                            .Device(DEVICE_NGRAPH)
                            .TypeConstraint<float>("T")
                            .TypeConstraint<int32>("Tidx"),
                        NGraphOp<float>);

REGISTER_KERNEL_BUILDER(
    Name("Pack").Device(DEVICE_NGRAPH).TypeConstraint<int32>("T"),
    NGraphOp<int32>);

REGISTER_KERNEL_BUILDER(
    Name("RefSwitch").Device(DEVICE_NGRAPH).TypeConstraint<float>("T"),
    NGraphOp<float>);

REGISTER_KERNEL_BUILDER(
    Name("Switch").Device(DEVICE_NGRAPH).TypeConstraint<bool>("T"),
    NGraphOp<bool>);

REGISTER_KERNEL_BUILDER(
    Name("Switch").Device(DEVICE_NGRAPH).TypeConstraint<float>("T"),
    NGraphOp<float>);

REGISTER_KERNEL_BUILDER(
    Name("Merge").Device(DEVICE_NGRAPH).TypeConstraint<float>("T"),
    NGraphOp<float>);

REGISTER_KERNEL_BUILDER(
    Name("Identity").Device(DEVICE_NGRAPH).TypeConstraint<float>("T"),
    NGraphOp<float>);

REGISTER_KERNEL_BUILDER(
    Name("Identity").Device(DEVICE_NGRAPH).TypeConstraint<bool>("T"),
    NGraphOp<bool>);

REGISTER_KERNEL_BUILDER(
    Name("StopGradient").Device(DEVICE_NGRAPH).TypeConstraint<float>("T"),
    NGraphOp<float>);

REGISTER_KERNEL_BUILDER(
    Name("Shape").Device(DEVICE_NGRAPH).TypeConstraint<float>("T"),
    NGraphOp<float>);

REGISTER_KERNEL_BUILDER(
    Name("ShapeN").Device(DEVICE_NGRAPH).TypeConstraint<float>("T"),
    NGraphOp<float>);

REGISTER_KERNEL_BUILDER(
    Name("Conv2D").Device(DEVICE_NGRAPH).TypeConstraint<float>("T"),
    NGraphOp<float>);

REGISTER_KERNEL_BUILDER(Name("Conv2DBackpropFilter")
                            .Device(DEVICE_NGRAPH)
                            .TypeConstraint<float>("T"),
                        NGraphOp<float>);

REGISTER_KERNEL_BUILDER(Name("Conv2DBackpropInput")
                            .Device(DEVICE_NGRAPH)
                            .TypeConstraint<float>("T"),
                        NGraphOp<float>);

REGISTER_KERNEL_BUILDER(Name("BroadcastGradientArgs")
                            .Device(DEVICE_NGRAPH)
                            .TypeConstraint<int32>("T"),
                        NGraphOp<int32>);
REGISTER_KERNEL_BUILDER(
    Name("Relu").Device(DEVICE_NGRAPH).TypeConstraint<float>("T"),
    NGraphOp<float>);

REGISTER_KERNEL_BUILDER(
    Name("MaxPool").Device(DEVICE_NGRAPH).TypeConstraint<float>("T"),
    NGraphOp<float>);

REGISTER_KERNEL_BUILDER(
    Name("MatMul").Device(DEVICE_NGRAPH).TypeConstraint<float>("T"),
    NGraphOp<float>);

REGISTER_KERNEL_BUILDER(
    Name("Equal").Device(DEVICE_NGRAPH).TypeConstraint<int64>("T"),
    NGraphOp<int64>);

REGISTER_KERNEL_BUILDER(
    Name("Neg").Device(DEVICE_NGRAPH).TypeConstraint<float>("T"),
    NGraphOp<float>);

REGISTER_KERNEL_BUILDER(
    Name("LogSoftmax").Device(DEVICE_NGRAPH).TypeConstraint<float>("T"),
    NGraphOp<float>);

REGISTER_KERNEL_BUILDER(
    Name("ScalarSummary").Device(DEVICE_NGRAPH).TypeConstraint<float>("T"),
    NGraphOp<float>);

REGISTER_KERNEL_BUILDER(
    Name("ReluGrad").Device(DEVICE_NGRAPH).TypeConstraint<float>("T"),
    NGraphOp<float>);

REGISTER_KERNEL_BUILDER(
    Name("MaxPoolGrad").Device(DEVICE_NGRAPH).TypeConstraint<float>("T"),
    NGraphOp<float>);

REGISTER_KERNEL_BUILDER(Name("SoftmaxCrossEntropyWithLogits")
                            .Device(DEVICE_NGRAPH)
                            .TypeConstraint<float>("T"),
                        NGraphOp<float>);

REGISTER_KERNEL_BUILDER(
    Name("ZerosLike").Device(DEVICE_NGRAPH).TypeConstraint<float>("T"),
    NGraphOp<float>);

REGISTER_KERNEL_BUILDER(
    Name("FloorDiv").Device(DEVICE_NGRAPH).TypeConstraint<int32>("T"),
    NGraphOp<int32>);

REGISTER_KERNEL_BUILDER(
    Name("RealDiv").Device(DEVICE_NGRAPH).TypeConstraint<float>("T"),
    NGraphOp<float>);

REGISTER_KERNEL_BUILDER(Name("Cast")
                            .Device(DEVICE_NGRAPH)
                            .TypeConstraint<bool>("SrcT")
                            .TypeConstraint<float>("DstT"),
                        NGraphOp<float>);

REGISTER_KERNEL_BUILDER(Name("Cast")
                            .Device(DEVICE_NGRAPH)
                            .TypeConstraint<int32>("SrcT")
                            .TypeConstraint<float>("DstT"),
                        NGraphOp<float>);

REGISTER_KERNEL_BUILDER(Name("Tile")
                            .Device(DEVICE_NGRAPH)
                            .TypeConstraint<float>("T")
                            .TypeConstraint<int32>("Tmultiples"),
                        NGraphOp<float>);

REGISTER_KERNEL_BUILDER(Name("ExpandDims")
                            .Device(DEVICE_NGRAPH)
                            .TypeConstraint<float>("T")
                            .TypeConstraint<int32>("Tdim"),
                        NGraphOp<float>);

#define REGISTER_NG_KERNEL(NAME, TYPE)                                  \
  REGISTER_KERNEL_BUILDER(                                              \
      Name((NAME)).Device(DEVICE_NGRAPH).TypeConstraint<TYPE>("dtype"), \
      NGraphNoOp);
// REGISTER_NG_KERNEL("Const", float);
REGISTER_NG_KERNEL("VariableV2", float);
REGISTER_NG_KERNEL("RandomUniform", float);

REGISTER_KERNEL_BUILDER(Name("MergeSummary").Device(DEVICE_NGRAPH), NGraphNoOp);
