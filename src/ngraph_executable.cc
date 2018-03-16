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
//  NGraphExecutable::CreateInputTensorViews()
//-----------------------------------------------------------------------------
Status NGraphExecutable::CreateInputTensorViews(
    const xla::HloComputation* computation,
    std::shared_ptr<ngraph::runtime::Backend>& ng_backend,
    tensorflow::gtl::ArraySlice<const ShapedBuffer*> arguments,
    std::vector<std::shared_ptr<ngraph::runtime::TensorView>>&
        ng_input_tv_list) {
  for (int64 p = 0; p < computation->num_parameters(); p++) {
    HloInstruction* param = computation->parameter_instruction(p);
    const ShapedBuffer* argument = arguments[p];

    for (auto& idx_bufs : argument->buffers()) {
      auto& shape_index = idx_bufs.first;
      auto& device_mem_base = idx_bufs.second;
      auto& buffer_shape = ShapeUtil::GetSubshape(param->shape(), shape_index);
      if (buffer_shape.element_type() != TUPLE) {
        TF_ASSIGN_OR_RETURN(auto ng_tv,
                            CreateNGraphTensor(buffer_shape, ng_backend));
        auto data_size = ShapeUtil::ByteSizeOf(buffer_shape);
        static_cast<ngraph::runtime::TensorView*>(ng_tv.get())
            ->write(argument->buffer(shape_index).opaque(), 0, data_size);
        ng_input_tv_list.push_back(ng_tv);
      }
    }
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
  // Get NGraphExecutor, for memory allocation
  se::Stream* stream = run_options->stream();
  se::StreamExecutor* executor(stream->parent());
  NGraphExecutor* ngraph_executor(
      static_cast<NGraphExecutor*>(executor->implementation()));

  // Start profiling
  uint64 start_micros = tensorflow::Env::Default()->NowMicros();

  // Get computation
  const xla::HloComputation* computation = module().entry_computation();
  if (computation->num_parameters() != static_cast<int64>(arguments.size())) {
    return tensorflow::errors::Internal(
        "Mismatch between argument count and graph parameter count.");
  }

  // ngraph backend
  auto ng_backend = m_ng_manager->allocate_backend();

  // Create the arguments as ngraph Parameters
  std::vector<std::shared_ptr<ngraph::runtime::TensorView>> ng_input_tv_list;
  TF_CHECK_OK(CreateInputTensorViews(computation, ng_backend, arguments,
                                     ng_input_tv_list));

  // Output tensor
  xla::HloInstruction* root_instruction = computation->root_instruction();
  xla::Shape root_shape = root_instruction->shape();

  // Create nGraph TensorViews and a ShapedBuffer for the result.
  //
  // This proceeds in two phases (it could be one, but we would need a
  // post-order traversal of the ShapedBuffer's buffer tree, and the iterator
  // only allows pre-order as far as I can tell). First, we allocate space
  // (a DeviceMemoryBase) for each node of the ShapedBuffer, and also nGraph
  // tensors for the leaf nodes. Second, we set the pointers inside of the non-
  // leaf nodes' buffers, which must point to each child node's buffer.

  // As we allocate nGraph tensors we will add them to this vector in order of
  // allocation (i.e. the "leftmost leaf" is first).
  std::vector<std::shared_ptr<ngraph::runtime::TensorView>> ng_result_tv_list;

  ShapedBuffer* result_buffer = new ShapedBuffer(
      root_shape, root_shape, executor->platform(), executor->device_ordinal());

  // Pass 1: Allocate DevieMemoryBases and nGraph tensors.
  for (auto& idx_buf : result_buffer->buffers()) {
    auto& shape_index = idx_buf.first;

    auto& sub_buffer_shape = ShapeUtil::GetSubshape(root_shape, shape_index);
    auto sub_buffer_byte_size =
        ShapeUtil::ByteSizeOf(sub_buffer_shape, sizeof(void*));

    TF_ASSIGN_OR_RETURN(se::DeviceMemoryBase sub_buffer_dmb,
                        run_options->allocator()->Allocate(
                            executor->device_ordinal(), sub_buffer_byte_size,
                            /*retry_on_failure=*/false));
    result_buffer->set_buffer(sub_buffer_dmb, shape_index);

    // If this is a leaf node we will create an nGraph tensor.
    if (sub_buffer_shape.element_type() != TUPLE) {
      TF_ASSIGN_OR_RETURN(auto ng_tv,
                          CreateNGraphTensor(sub_buffer_shape, ng_backend));
      ng_result_tv_list.push_back(ng_tv);
    }
  }

  // Pass 2: Set up the pointers to child buffers for the non-leaf nodes.
  for (auto& idx_buf : result_buffer->buffers()) {
    auto& shape_index = idx_buf.first;
    auto& device_mem_base = idx_buf.second;

    auto& sub_buffer_shape = ShapeUtil::GetSubshape(root_shape, shape_index);
    void* sub_buffer_data = device_mem_base.opaque();

    if (sub_buffer_shape.element_type() == TUPLE) {
      auto child_shape_index = shape_index;
      const void** p = (const void**)sub_buffer_data;
      for (int64 i = 0; i < ShapeUtil::TupleElementCount(sub_buffer_shape);
           i++) {
        child_shape_index.push_back(i);
        *p++ = result_buffer->buffer(child_shape_index).opaque();
        child_shape_index.pop_back();
      }
    }
  }

  // Call the nGraph executable.
  auto call_frame = ng_backend->make_call_frame(m_ng_runtime_function);
  call_frame->call(ng_input_tv_list, ng_result_tv_list);

  // Copy data back from the nGraph result tensors to each corresponding leaf
  // buffer.
  size_t i = 0;

  for (auto it = result_buffer->buffers().leaf_begin();
       it != result_buffer->buffers().leaf_end(); ++it) {
    auto shape_index = it->first;
    auto device_mem_base = it->second;

    auto& sub_buffer_shape = ShapeUtil::GetSubshape(root_shape, shape_index);
    auto sub_buffer_byte_size = ShapeUtil::ByteSizeOf(sub_buffer_shape);

    auto ng_result_tv =
        static_cast<ngraph::runtime::TensorView*>(ng_result_tv_list[i].get());
    ng_result_tv->read(device_mem_base.opaque(), 0, sub_buffer_byte_size);

    i++;
  }

  // Profiling scripts
  uint64 end_micros = tensorflow::Env::Default()->NowMicros();
  {
    tensorflow::mutex_lock lock(mutex_);
    const double nanoseconds = (end_micros - start_micros) * 1000.0;
    execution_profile_.set_compute_time_ns(std::max(nanoseconds, 1.0));
  }

  return std::unique_ptr<xla::ShapedBuffer>(result_buffer);
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
