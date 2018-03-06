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

#include "tensorflow/compiler/xla/executable_run_options.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_evaluator.h"
#include "tensorflow/compiler/xla/shape_util.h"

#include "executable.h"

namespace se = ::perftools::gputools;

xla::StatusOr<std::unique_ptr<xla::ShapedBuffer>>
PluginExecutable::ExecuteOnStream(
    const xla::ServiceExecutableRunOptions* run_options,
    tensorflow::gtl::ArraySlice<const xla::ShapedBuffer*> arguments,
    xla::HloExecutionProfile* hlo_execution_profile) {
  se::Stream* stream = run_options->stream();

  se::StreamExecutor* executor = stream->parent();
  const se::Platform* platform = executor->platform();

  std::cout << "Executing the module " << module().name() << std::endl;

  tensorflow::uint64 start_micros = tensorflow::Env::Default()->NowMicros();

  const xla::HloComputation* computation = module().entry_computation();
  if (computation->num_parameters() != arguments.size()) {
    return tensorflow::errors::Internal(
        "Mismatch between argument count and graph parameter count.");
  }

  // WE NEED TO get our own transfer manager
  // TODO
  //...
  std::cout << "Platform: " << platform->Name() << std::endl;

  // Transform the ShapedBuffer arguments into literals which the
  // evaluator consumes.
  std::vector<std::unique_ptr<xla::Literal>> arg_literals;
  for (tensorflow::int64 p = 0; p < computation->num_parameters(); ++p) {
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<xla::Literal> arg_literal,
        m_transfer_manager->TransferLiteralFromDevice(executor, *arguments[p]));
    arg_literals.push_back(std::move(arg_literal));
  }

  // Execute the graph using the HloEvaluator.
  std::cout << "PluginExecutable::ExecuteOnStream Executing computation using "
               "HLO-EVALUATOR"
            << std::endl;
  xla::HloEvaluator evaluator;
  TF_ASSIGN_OR_RETURN(std::unique_ptr<xla::Literal> result_literal,
                      evaluator.Evaluate<std::unique_ptr<xla::Literal>>(
                          *computation, arg_literals));

  // Make sure that the result shape is not empty
  TF_RET_CHECK(!xla::ShapeUtil::IsNil(result_literal->shape()));

  TF_ASSIGN_OR_RETURN(std::unique_ptr<xla::ShapedBuffer> result,
                      m_transfer_manager->AllocateShapedBuffer(
                          result_literal->shape(), run_options->allocator(),
                          run_options->device_ordinal()));

  TF_RETURN_IF_ERROR(m_transfer_manager->TransferLiteralToDevice(
      executor, *result_literal, *result));

  tensorflow::uint64 end_micros = tensorflow::Env::Default()->NowMicros();

  {
    tensorflow::mutex_lock lock(mutex_);
    const double nanoseconds = (end_micros - start_micros) * 1000.0;
    execution_profile_.set_compute_time_ns(std::max(nanoseconds, 1.0));
  }

  return std::move(result);
}
