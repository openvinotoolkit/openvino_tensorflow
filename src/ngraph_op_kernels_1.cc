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

template <typename T>
class NGraphAddOp : public OpKernel {
 public:
  explicit NGraphAddOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    auto node_def = ctx->def();
    VLOG(0) << "NGraphAddOp::ctor(): Node: " << node_def.name()
            << " Op: " << node_def.op();
    // Verify that the tensor shapes match
    // TODO
  }

  void Compute(OpKernelContext* ctx) override {
    VLOG(0) << "NGraphAddOp::Compute() Step: " << ctx->step_id()
            << " Op: " << ctx->op_kernel().name();
    VLOG(0) << "Inputs: " << ctx->num_inputs()
            << " Outputs: " << ctx->num_outputs();
    // Get the inputs
    const tf::Tensor& input_tensor_1 = ctx->input(0);
    const tf::Tensor& input_tensor_2 = ctx->input(1);

    // DO the Math

    // Save the output
    // Create an output tensor
    tf::Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, input_tensor_1.shape(), &output_tensor));
  }
};

template <typename T>
class NGraphMulOp : public OpKernel {
 public:
  explicit NGraphMulOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    auto node_def = ctx->def();
    VLOG(0) << "NGraphMulOp::ctor(): Node: " << node_def.name()
            << " Op: " << node_def.op();
  }

  void Compute(OpKernelContext* ctx) override {
    VLOG(0) << "NGraphMulOp::Compute() Step: " << ctx->step_id()
            << " Op: " << ctx->op_kernel().name();
    VLOG(0) << "Inputs: " << ctx->num_inputs()
            << " Outputs: " << ctx->num_outputs();
    // Get the inputs
    const tf::Tensor& input_tensor_1 = ctx->input(0);
    const tf::Tensor& input_tensor_2 = ctx->input(1);

    // DO the Math

    // Save the output
    // Create an output tensor
    tf::Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, input_tensor_1.shape(), &output_tensor));
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
  }
  void Compute(OpKernelContext* ctx) override {
    VLOG(0) << "NGraphNoOp::Compute() Step: " << ctx->step_id()
            << " Op: " << ctx->op_kernel().name();
  }
  ~NGraphConstantOp() override {}

 private:
};

static string GetRendezvousKeyPrefix(const string& send_device,
                                     const string& recv_device,
                                     const uint64 send_device_incarnation,
                                     const string& tensor_name) {
  return strings::StrCat(send_device, ";",
                         strings::FpToString(send_device_incarnation), ";",
                         recv_device, ";", tensor_name);
}

static void GetRendezvousKey(const string& key_prefix,
                             const tf::FrameAndIter& frame_iter, string* key) {
  key->clear();
  strings::StrAppend(key, key_prefix, ";", frame_iter.frame_id, ":",
                     frame_iter.iter_id);
}

static tf::FrameAndIter GetFrameAndIter(OpKernelContext* ctx,
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

namespace {
tf::Rendezvous::DoneCallback make_recv_callback(
    tf::OpKernelContext* ctx, tf::AsyncOpKernel::DoneCallback done) {
  using namespace std::placeholders;
  return std::bind(
      [ctx](tf::AsyncOpKernel::DoneCallback done,
            // Begin unbound arguments.
            const tf::Status& s, const tf::Rendezvous::Args& send_args,
            const tf::Rendezvous::Args& recv_args, const tf::Tensor& val,
            bool is_dead) {
        ctx->SetStatus(s);
        if (s.ok()) {
          // 'ctx' allocates the output tensor of the expected type.
          // The runtime checks whether the tensor received here is
          // the same type.
          if (!is_dead) {
            ctx->set_output(0, val);
          }
          *ctx->is_output_dead() = is_dead;
        }
        done();
      },
      std::move(done), _1, _2, _3, _4, _5);
}
}  // namespace

//-----------------------------------------------------------------------------
class NGraphRecv : public AsyncOpKernel {
 public:
  explicit NGraphRecv(OpKernelConstruction* ctx) : AsyncOpKernel(ctx) {
    auto node_def = ctx->def();
    VLOG(0) << "NGraphRecv::ctor(): Node: " << node_def.name()
            << " Op: " << node_def.op();

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

    std::cout << "NGraphRecv: send_device: " << send_device
              << " recv_device: " << recv_device
              << " send_device_incarnation: " << send_device_incarnation
              << " tensor_name: " << tensor_name
              << " key_prefix_: " << key_prefix_ << std::endl;
    // The vast majority of Recv nodes are outside any loop context, so
    // proactively cache the rendezvous key for the top-level.
    string key;
    GetRendezvousKey(key_prefix_, {0, 0}, &key);
    std::cout << "Key: " << key << std::endl;

    OP_REQUIRES_OK(ctx, Rendezvous::ParseKey(key, &parsed_key_));

    if (!ctx->GetAttr("_hostmem_sendrecv", &hostmem_sendrecv_).ok()) {
      hostmem_sendrecv_ = false;
    }
  }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    std::cout << "NGraphRecv: Step: " << ctx->step_id()
              << " Op: " << ctx->op_kernel().name() << std::endl;
    OP_REQUIRES_ASYNC(
        ctx, ctx->rendezvous() != nullptr,
        errors::Internal("Op kernel context needs to provide a rendezvous."),
        done);

    tf::Rendezvous::Args args;
    args.device_context = ctx->op_device_context();
    args.alloc_attrs = ctx->output_alloc_attr(0);

    std::cout << "ComputeAsync: DEV-CTX: " << args.device_context << std::endl;

    tf::FrameAndIter frame_iter = GetFrameAndIter(ctx, hostmem_sendrecv_);
    if (frame_iter == FrameAndIter(0, 0)) {
      VLOG(0) << "ComputeAsync::Recv " << parsed_key_.FullKey();
      ctx->rendezvous()->RecvAsync(parsed_key_, args,
                                   make_recv_callback(ctx, std::move(done)));
    } else {
      Rendezvous::ParsedKey in_loop_parsed;
      string key;
      GetRendezvousKey(key_prefix_, frame_iter, &key);
      VLOG(0) << "ComputeAsync::Recv " << in_loop_parsed.FullKey();
      OP_REQUIRES_OK_ASYNC(ctx, Rendezvous::ParseKey(key, &in_loop_parsed),
                           done);
      ctx->rendezvous()->RecvAsync(in_loop_parsed, args,
                                   make_recv_callback(ctx, std::move(done)));
    }
  }

 private:
  string key_prefix_;
  tf::Rendezvous::ParsedKey parsed_key_;
  bool hostmem_sendrecv_;
};

class NGraphSend : public OpKernel {
 public:
  explicit NGraphSend(OpKernelConstruction* ctx) : OpKernel(ctx) {
    auto node_def = ctx->def();
    VLOG(0) << "NGraphSend::ctor(): Node: " << node_def.name()
            << " Op: " << node_def.op();
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
    string key;
    GetRendezvousKey(key_prefix_, {0, 0}, &key);
    OP_REQUIRES_OK(ctx, Rendezvous::ParseKey(key, &parsed_key_));
    if (!ctx->GetAttr("_hostmem_sendrecv", &hostmem_sendrecv_).ok()) {
      hostmem_sendrecv_ = false;
    }
  }

  void Compute(OpKernelContext* ctx) override {
    std::cout << "NGraphSend: Step: " << ctx->step_id()
              << " Op: " << ctx->op_kernel().name() << std::endl;
    OP_REQUIRES(ctx, ctx->rendezvous() != nullptr,
                tf ::errors::Internal(
                    "Op kernel context needs to provide a rendezvous."));

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
      VLOG(0) << "Send " << parsed_key_.FullKey();
      ctx->SetStatus(ctx->rendezvous()->Send(parsed_key_, args, ctx->input(0),
                                             ctx->is_input_dead()));
      return;
    } else {
      Rendezvous::ParsedKey in_loop_parsed;
      string key;
      GetRendezvousKey(key_prefix_, frame_iter, &key);
      OP_REQUIRES_OK(ctx, Rendezvous::ParseKey(key, &in_loop_parsed));
      VLOG(0) << "Send " << in_loop_parsed.FullKey();

      ctx->SetStatus(ctx->rendezvous()->Send(
          in_loop_parsed, args, ctx->input(0), ctx->is_input_dead()));
      return;
    }
  }

 private:
  string key_prefix_;
  tf::Rendezvous::ParsedKey parsed_key_;
  bool hostmem_sendrecv_;
};

// This form allows you to specify a list of types as the constraint.
REGISTER_KERNEL_BUILDER(
    Name("Add").Device(DEVICE_NGRAPH).TypeConstraint("T", {DT_FLOAT}),
    NGraphAddOp<float>);
REGISTER_KERNEL_BUILDER(
    Name("Sub").Device(DEVICE_NGRAPH).TypeConstraint("T", {DT_FLOAT}),
    NGraphOp<float>);
REGISTER_KERNEL_BUILDER(
    Name("Mul").Device(DEVICE_NGRAPH).TypeConstraint("T", {DT_FLOAT}),
    NGraphMulOp<float>);

REGISTER_KERNEL_BUILDER(Name("NoOp").Device(DEVICE_NGRAPH), NGraphNoOp);
// REGISTER_KERNEL_BUILDER(Name("Placeholder").Device(DEVICE_NGRAPH),
//                         NgPlaceholderOp);
// REGISTER_KERNEL_BUILDER(Name("PlaceholderV2").Device(DEVICE_NGRAPH),
//                         NgPlaceholderOp);
REGISTER_KERNEL_BUILDER(Name("_Recv").Device(DEVICE_NGRAPH), NGraphRecv);
REGISTER_KERNEL_BUILDER(Name("_Send").Device(DEVICE_NGRAPH), NGraphSend);
