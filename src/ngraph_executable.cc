/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "ngraph_executable.h"
#include "ngraph/builder/xla_tuple.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/call_frame.hpp"
#include "ngraph/runtime/manager.hpp"
#include "ngraph/runtime/tensor_view.hpp"
#include "ngraph_executor.h"
#include "ngraph_log.h"
#include "ngraph_utils.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_evaluator.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {
namespace ngraph_plugin {

NGraphExecutable::NGraphExecutable(
    std::unique_ptr<HloModule> hlo_module,
    std::shared_ptr<ngraph::runtime::Manager> ng_manager,
    std::shared_ptr<ngraph::runtime::ExternalFunction> ng_runtime_function)
    : Executable(std::move(hlo_module), /*hlo_profile_printer=*/nullptr,
                 /*hlo_profile_index_map=*/nullptr),
      m_ng_manager(ng_manager),
      m_ng_runtime_function(ng_runtime_function) {}

NGraphExecutable::~NGraphExecutable() {}

//-----------------------------------------------------------------------------
// NGraphExecutable::ExecuteOnStream()
//-----------------------------------------------------------------------------
/*
StatusOr<se::DeviceMemoryBase> NGraphExecutable::ExecuteOnStream(
    const ServiceExecutableRunOptions* run_options,
    tensorflow::gtl::ArraySlice<se::DeviceMemoryBase> arguments,
    HloExecutionProfile* hlo_execution_profile) {
  // Get NGraphExecutor, for memory allocation
  se::Stream* stream = run_options->stream();
  se::StreamExecutor* executor(stream->parent());
  NGraphExecutor* ngraph_executor(
      static_cast<NGraphExecutor*>(executor->implementation()));

  // Start profiling
  uint64 start_micros = tensorflow::Env::Default()->NowMicros();

  // Get computation
  xla::HloComputation* computation = module().entry_computation();
  if (computation->num_parameters() != static_cast<int64>(arguments.size())) {
    return tensorflow::errors::Internal(
        "Mismatch between argument count and graph parameter count.");
  }

  // ngraph backend
  auto ng_backend = m_ng_manager->allocate_backend();

  // Create the arguments as ngraph Parameters
  std::vector<std::shared_ptr<ngraph::runtime::TensorView>> ng_arg_list;
  TF_CHECK_OK(CreateInputArgs(computation, ng_backend, arguments, ng_arg_list));

  // Output tensor
  xla::HloInstruction* root_instruction = computation->root_instruction();
  xla::Shape root_shape = root_instruction->shape();

  // Handle tuple differently
  se::DeviceMemoryBase ret;
  if (ShapeUtil::IsTuple(root_shape)) {
    // Construct ngraph result tuple op
    std::vector<std::shared_ptr<ngraph::runtime::TensorView>> ng_result_buffers;
    int num_elements = int(ShapeUtil::TupleElementCount(root_shape));
    for (int i = 0; i < num_elements; ++i) {
      auto ng_result_buffer = std::shared_ptr<ngraph::runtime::TensorView>();
      TF_ASSIGN_OR_RETURN(
          ng_result_buffer,
          CreateNGraphTensor(ShapeUtil::GetTupleElementShape(root_shape, i),
                             ng_backend));
      ng_result_buffers.push_back(ng_result_buffer);
    }

    // Create the result of the computation
    auto ng_result_tuple_op = ngraph::xla::make_tuple(ng_result_buffers);

    // Call the executable
    auto call_frame = ng_backend->make_call_frame(m_ng_runtime_function);
    ngraph::xla::call(call_frame, ng_arg_list, {ng_result_tuple_op});

    // Allocate memory on XLA NGraphDevice, top level memory is just void*
    // pointer to elements
    int64 tuple_top_size(xla::ShapeUtil::ByteSizeOf(root_shape, sizeof(void*)));
    void** buf =
        reinterpret_cast<void**>(ngraph_executor->Allocate(tuple_top_size));
    void** buf_rc = buf;

    // Allocate individual buffers and copy values
    num_elements = int(ShapeUtil::TupleElementCount(root_shape));
    for (int i = 0; i < num_elements; ++i) {
      // Get pointer to ngraph values
      // Allocate Host-side memory
      // TODO - figure out how to reuse memory
      int byte_size = int(ShapeUtil::ByteSizeOf(
          ShapeUtil::GetTupleElementShape(root_shape, i)));
      void* element_buf = ngraph_executor->Allocate(byte_size);

      // Copy the result data to the host side (i.e., TensorFlow side)
      auto ng_result_buffer =
          static_cast<ngraph::runtime::TensorView*>(ng_result_buffers[i].get());
      ng_result_buffer->read(element_buf, 0, byte_size);

      // Convert void* to DeviceMemoryBase
      auto out = se::DeviceMemoryBase(element_buf, byte_size);
      *buf++ = out.opaque();
    }

    // Apply to return value
    ret = se::DeviceMemoryBase(buf_rc, tuple_top_size);

  } else {
    // Allocate memory for ngraph-cpp
    // TODO: Investigate whether we can reuse the return memory

    // Create and execute ngraph call frame
    auto call_frame = ng_backend->make_call_frame(m_ng_runtime_function);

    // Allocate memory on XLA NGraphDevice
    int byte_size = ShapeUtil::ByteSizeOf(root_shape);
    void* buf = ngraph_executor->Allocate(byte_size);

    // nGraph result tensor
    auto ng_result_buffer = std::shared_ptr<ngraph::runtime::TensorView>();
    TF_ASSIGN_OR_RETURN(ng_result_buffer,
                        CreateNGraphTensor(root_shape, ng_backend));
    call_frame->call(ng_arg_list, {ng_result_buffer});

    // Copy the return data to host side memory
    ng_result_buffer->read(buf, 0, byte_size);

    // Apply to return value
    ret = se::DeviceMemoryBase(buf, byte_size);
  }

  // Profiling scripts
  uint64 end_micros = tensorflow::Env::Default()->NowMicros();
  {
    tensorflow::mutex_lock lock(mutex_);
    const double nanoseconds = (end_micros - start_micros) * 1000.0;
    execution_profile_.set_compute_time_ns(std::max(nanoseconds, 1.0));
  }

  return ret;
}
*/

//-----------------------------------------------------------------------------
//  NGraphExecutable::CreateNGraphTensor()
//-----------------------------------------------------------------------------
StatusOr<std::shared_ptr<ngraph::runtime::TensorView>>
NGraphExecutable::CreateNGraphTensor(
    const xla::Shape& xla_shape,
    const std::shared_ptr<ngraph::runtime::Backend>& ng_backend) {
  auto ng_tensor = std::shared_ptr<ngraph::runtime::TensorView>();
  auto ng_element_shape = ngraph::Shape(xla_shape.dimensions().begin(),
                                        xla_shape.dimensions().end());
  switch (xla_shape.element_type()) {
    case F32: {
      ng_tensor = ng_backend->make_primary_tensor_view(
          ngraph::element::f32, ngraph::Shape(ng_element_shape));
    } break;
    case S32: {
      ng_tensor = ng_backend->make_primary_tensor_view(
          ngraph::element::i32, ngraph::Shape(ng_element_shape));
    } break;
    case PRED: {
      ng_tensor = ng_backend->make_primary_tensor_view(
          ngraph::element::boolean, ngraph::Shape(ng_element_shape));
    } break;
    default:
      return Unimplemented(
          "CreateNGraphTensor() Data type: '%s'",
          xla::PrimitiveType_Name(xla_shape.element_type()).c_str());
      break;
  }
  return ng_tensor;
}

//-----------------------------------------------------------------------------
//  NGraphExecutable::CreateInputArgs()
//-----------------------------------------------------------------------------
Status NGraphExecutable::CreateInputArgs(
    const xla::HloComputation* computation,
    std::shared_ptr<ngraph::runtime::Backend>& ng_backend,
    tensorflow::gtl::ArraySlice<const ShapedBuffer*> arguments,
    std::vector<std::shared_ptr<ngraph::runtime::TensorView>>& ng_arg_list) {
  // Create the arguments as ngraph Parameters
  for (int64 p = 0; p < computation->num_parameters(); p++) {
    HloInstruction* param = computation->parameter_instruction(p);

    auto ng_input_arg = std::shared_ptr<ngraph::runtime::TensorView>();
    TF_ASSIGN_OR_RETURN(ng_input_arg,
                        CreateNGraphTensor(param->shape(), ng_backend));
    auto data_size = ShapeUtil::ByteSizeOf(param->shape());
    static_cast<ngraph::runtime::TensorView*>(ng_input_arg.get())
        ->write(arguments[p]->root_buffer().opaque(), 0, data_size);
    ng_arg_list.push_back(ng_input_arg);
  }
  return Status::OK();
}

//-----------------------------------------------------------------------------
// NGraphExecutable::ExecuteOnStream()
//-----------------------------------------------------------------------------
StatusOr<std::unique_ptr<ShapedBuffer>> NGraphExecutable::ExecuteOnStream(
    const ServiceExecutableRunOptions* run_options,
    tensorflow::gtl::ArraySlice<const ShapedBuffer*> arguments,
    HloExecutionProfile* hlo_execution_profile) {
  return tensorflow::errors::Unimplemented(
      "ExecuteOnStream is not yet supported on nGraph.");
}

//-----------------------------------------------------------------------------
// NGraphExecutable::ExecuteAsyncOnStream() [not supported]
//-----------------------------------------------------------------------------
StatusOr<std::unique_ptr<ShapedBuffer>> NGraphExecutable::ExecuteAsyncOnStream(
    const ServiceExecutableRunOptions* run_options,
    tensorflow::gtl::ArraySlice<const ShapedBuffer*> arguments) {
  return tensorflow::errors::Unimplemented(
      "ExecuteAsyncOnStream is not yet supported on nGraph.");
}

//-----------------------------------------------------------------------------
// NGraphExecutable::ShapeSizeBytes
//-----------------------------------------------------------------------------
int64 NGraphExecutable::ShapeSizeBytes(const Shape& shape) {
  if (ShapeUtil::IsOpaque(shape)) {
    return sizeof(void*);
  }
  return ShapeUtil::ByteSizeOf(shape, sizeof(void*));
}

}  // namespace ngraph_plugin
}  // namespace xla
