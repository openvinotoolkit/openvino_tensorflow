/*******************************************************************************
 * Copyright 2017-2018 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/strings/strcat.h"

#include "tensorflow/core/platform/default/logging.h"

#include "ngraph_utils.h"

namespace ngraph_bridge {
extern const char* const DEVICE_NGRAPH;
}

using namespace tensorflow;

//-----------------------------------------------------------------------------
//  GetRendezvousKeyPrefix
//-----------------------------------------------------------------------------
static string GetRendezvousKeyPrefix(const string& send_device,
                                     const string& recv_device,
                                     const uint64 send_device_incarnation,
                                     const string& tensor_name) {
  return strings::StrCat(send_device, ";",
                         strings::FpToString(send_device_incarnation), ";",
                         recv_device, ";", tensor_name);
}

//-----------------------------------------------------------------------------
//  GetRendezvousKey
//-----------------------------------------------------------------------------
static void GetRendezvousKey(const string& key_prefix,
                             const tf::FrameAndIter& frame_iter, string* key) {
  key->clear();
  strings::StrAppend(key, key_prefix, ";", frame_iter.frame_id, ":",
                     frame_iter.iter_id);
}

//-----------------------------------------------------------------------------
//  GetFrameAndIter
//-----------------------------------------------------------------------------
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
//-----------------------------------------------------------------------------
//  make_recv_callback
//-----------------------------------------------------------------------------
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
//  NGraphSend
//-----------------------------------------------------------------------------
class NGraphRecv : public AsyncOpKernel {
 public:
  explicit NGraphRecv(OpKernelConstruction* ctx) : AsyncOpKernel(ctx) {
    auto node_def = ctx->def();
    NGRAPH_VLOG(4) << "NGraphRecv::ctor(): Node: " << node_def.name()
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

    // std::cout << "NGraphRecv: send_device: " << send_device
    //           << " recv_device: " << recv_device
    //           << " send_device_incarnation: " << send_device_incarnation
    //           << " tensor_name: " << tensor_name
    //           << " key_prefix_: " << key_prefix_ << std::endl;
    // The vast majority of Recv nodes are outside any loop context, so
    // proactively cache the rendezvous key for the top-level.
    string key;
    GetRendezvousKey(key_prefix_, {0, 0}, &key);
    // std::cout << "Key: " << key << std::endl;

    OP_REQUIRES_OK(ctx, Rendezvous::ParseKey(key, &parsed_key_));

    if (!ctx->GetAttr("_hostmem_sendrecv", &hostmem_sendrecv_).ok()) {
      hostmem_sendrecv_ = false;
    }
  }

  //-----------------------------------------------------------------------------
  //  ComputeAsync
  //-----------------------------------------------------------------------------
  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    NGRAPH_VLOG(4) << "NGraphRecv: Step: " << ctx->step_id()
            << " Op: " << ctx->op_kernel().name();
    OP_REQUIRES_ASYNC(
        ctx, ctx->rendezvous() != nullptr,
        errors::Internal("Op kernel context needs to provide a rendezvous."),
        done);

    tf::Rendezvous::Args args;
    args.device_context = ctx->op_device_context();
    args.alloc_attrs = ctx->output_alloc_attr(0);

    NGRAPH_VLOG(4) << "ComputeAsync: DEV-CTX: " << args.device_context;

    tf::FrameAndIter frame_iter = GetFrameAndIter(ctx, hostmem_sendrecv_);
    if (frame_iter == FrameAndIter(0, 0)) {
      NGRAPH_VLOG(4) << "ComputeAsync::Recv " << parsed_key_.FullKey();
      ctx->rendezvous()->RecvAsync(parsed_key_, args,
                                   make_recv_callback(ctx, std::move(done)));
    } else {
      Rendezvous::ParsedKey in_loop_parsed;
      string key;
      GetRendezvousKey(key_prefix_, frame_iter, &key);
      NGRAPH_VLOG(4) << "ComputeAsync::Recv " << in_loop_parsed.FullKey();
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

//-----------------------------------------------------------------------------
//  NGraphSend
//-----------------------------------------------------------------------------
class NGraphSend : public OpKernel {
 public:
  explicit NGraphSend(OpKernelConstruction* ctx) : OpKernel(ctx) {
    auto node_def = ctx->def();
    NGRAPH_VLOG(4) << "NGraphSend::ctor(): Node: " << node_def.name()
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
  //-----------------------------------------------------------------------------
  //  NGraphSend::Compute
  //-----------------------------------------------------------------------------
  void Compute(OpKernelContext* ctx) override {
    NGRAPH_VLOG(4) << "NGraphSend: Step: " << ctx->step_id()
                   << " Op: " << ctx->op_kernel().name();
    OP_REQUIRES(ctx, ctx->rendezvous() != nullptr,
                tf::errors::Internal(
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
      NGRAPH_VLOG(4) << "Send " << parsed_key_.FullKey();
      ctx->SetStatus(ctx->rendezvous()->Send(parsed_key_, args, ctx->input(0),
                                             ctx->is_input_dead()));
      return;
    } else {
      Rendezvous::ParsedKey in_loop_parsed;
      string key;
      GetRendezvousKey(key_prefix_, frame_iter, &key);
      OP_REQUIRES_OK(ctx, Rendezvous::ParseKey(key, &in_loop_parsed));
      NGRAPH_VLOG(4) << "Send " << in_loop_parsed.FullKey();

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

REGISTER_KERNEL_BUILDER(Name("_Recv").Device(ngraph_bridge::DEVICE_NGRAPH),
                        NGraphRecv);
REGISTER_KERNEL_BUILDER(Name("_Send").Device(ngraph_bridge::DEVICE_NGRAPH),
                        NGraphSend);
REGISTER_KERNEL_BUILDER(Name("_HostRecv")
                            .Device(ngraph_bridge::DEVICE_NGRAPH)
                            .HostMemory("tensor"),
                        NGraphRecv);
REGISTER_KERNEL_BUILDER(Name("_HostSend")
                            .Device(ngraph_bridge::DEVICE_NGRAPH)
                            .HostMemory("tensor"),
                        NGraphSend);
