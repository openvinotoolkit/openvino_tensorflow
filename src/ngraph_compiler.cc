/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
#include "ngraph/builder/xla_tuple.hpp"
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <stdlib.h>
#include <fstream>
#include <string>
#include <vector>
#include "ngraph/serializer.hpp"
#include "ngraph_compiler.h"
#include "ngraph_emitter.h"
#include "ngraph_executable.h"
#include "ngraph_fusion.h"
#include "ngraph_log.h"
#include "ngraph_utils.h"
//#include "tensorflow/compiler/xla/service/algebraic_simplifier.h"
//#include "tensorflow/compiler/xla/service/flatten_call_graph.h"
//#include "tensorflow/compiler/xla/service/hlo_constant_folding.h"
//#include "tensorflow/compiler/xla/service/hlo_cse.h"
//#include "tensorflow/compiler/xla/service/hlo_dce.h"
//#include "tensorflow/compiler/xla/service/hlo_graph_dumper.h"
#include "tensorflow/compiler/xla/service/hlo_pass_fix.h"
#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"
//#include "tensorflow/compiler/xla/service/hlo_subcomputation_unification.h"
// #include "tensorflow/compiler/xla/service/inliner.h"
// #include "tensorflow/compiler/xla/service/reshape_mover.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/stream_executor/lib/initialize.h"
#include "tensorflow/stream_executor/lib/strcat.h"

namespace xla {
namespace ngraph_plugin {

#define XLA_NGRAPH_BACKEND_ENV_VAR "XLA_NGRAPH_BACKEND"
static const std::string XLA_NGRAPH_DEFAULT_BACKEND("CPU");

NGraphCompiler::NGraphCompiler() {}
std::shared_ptr<ngraph::runtime::Manager>
    NGraphCompiler::m_ngraph_runtime_manager;

//---------------------------------------------------------------------------
// RunHloOptimization
//
// Run optimization passes on the module.  The graph is transformed by
// each pass in the optimization pipeline.  The service subdirectory
// contains useful optimization passes.
//
// TODO: Enable or add more passes as needed.
//---------------------------------------------------------------------------
StatusOr<std::unique_ptr<HloModule>> NGraphCompiler::RunHloPasses(
    std::unique_ptr<HloModule> hlo_module,
    perftools::gputools::StreamExecutor* executor,
    DeviceMemoryAllocator* device_allocator) {
  HloPassPipeline pipeline("NGraph");
  // pipeline.AddPass<Inliner>();
  // pipeline.AddPass<HloSubcomputationUnification>();
  // pipeline.AddPass<HloCSE>(false);

  // This is required to break constants op containing tuples, to a tuple op
  // containing multiple constnat ops. E.g.
  // [from]
  // %constant = (f32[], f32[3]{0}, f32[2,3]{1,0}) constant({...})]
  // [to]
  // TupleConstant.v1 () -> (f32[], f32[3], f32[2,3]) {
  //   %constant.1 = f32[] constant(7.3)
  //   %constant.2 = f32[3]{0} constant({1.1, 2, 3.3})
  //   %constant.3 = f32[2,3]{1,0} constant(f32[2,3] { { 1.1, 2.2, 3.5 },
  //                                                   { 4.8, 5, 6.7 }, })
  //   %tuple = (f32[], f32[3]{0}, f32[2,3]{1,0}) tuple(f32[] %constant.1,
  //            f32[3]{0} %constant.2, f32[2,3]{1,0} %constant.3)
  // }
  if (getenv("XLA_NGRAPH_SKIP_FUSION") == nullptr)
    pipeline.AddPass<HloPassFix<NGraphFusion>>(&m_fusion_map);

  // TODO: Currently we're setting the 3rd argument
  //       enable_dot_simplification=false, since we don't have reshape op yet.
  // We are possibly interested only in the Algebraic Simplification
  // pipeline.AddPass<HloPassFix<AlgebraicSimplifier>>(
  //    false, [](const Shape&, const Shape&) { return false; }, false);

  // Possibly remove passes below
  // pipeline.AddPass<ReshapeMover>();
  // pipeline.AddPass<HloConstantFolding>();
  // pipeline.AddPass<HloCSE>(true);

  // pipeline.AddPass<HloDCE>();
  // pipeline.AddPass<FlattenCallGraph>();

  TF_ASSIGN_OR_RETURN(bool changed, pipeline.Run(hlo_module.get()));

  return std::move(hlo_module);
}

// This is a temporary measure to give us warning if we need to support control
// edges for a particular workload.  See NGTF-258.
class NGraphControlEdgeDetector : public DfsHloVisitorWithDefault {
 public:
  NGraphControlEdgeDetector() {}
  ~NGraphControlEdgeDetector() {}
  virtual Status DefaultAction(HloInstruction* hlo_instruction) override {
    TF_RET_CHECK(hlo_instruction != nullptr);
    TF_RET_CHECK(hlo_instruction->control_predecessors().empty() &&
                 hlo_instruction->control_successors().empty())
        << "Encountered control-edges in the XLA graph, but NG++ can't handle "
           "them yet.";
    return tensorflow::Status::OK();
  }
};

//---------------------------------------------------------------------------
// print_embedded_computation
//---------------------------------------------------------------------------
static void print_embedded_computation(const xla::HloComputation* computation,
                                       int nest_level = 0) {
  auto embedded_computations = computation->MakeEmbeddedComputationsList();
  // Keep as cout because this outputs the subgraph for a nice format
  std::cout << "NGRAPH_COMPILER  computation: " << computation->name()
            << "; nest_level: " << nest_level
            << "; num_embedded: " << embedded_computations.size();
  std::cout << computation->ToString() << std::endl;
  for (auto embedded_computation : embedded_computations) {
    print_embedded_computation(embedded_computation, nest_level + 1);
  }
}

//---------------------------------------------------------------------------
// Compile
//---------------------------------------------------------------------------
StatusOr<std::unique_ptr<Executable>> NGraphCompiler::RunBackend(
    std::unique_ptr<HloModule> hlo_module,
    perftools::gputools::StreamExecutor* stream_exec,
    DeviceMemoryAllocator* device_allocator) {
  bool dump_graph = false;
  if (const char* env_str = std::getenv("XLA_NGRAPH_DUMP_GRAPH")) {
    int dum_graph_option = 0;
    if (xla::ngraph_plugin::try_parse<int>(env_str, dum_graph_option)) {
      switch (dum_graph_option) {
        case 0:
          dump_graph = false;
          break;
        case 1:
          dump_graph = true;
          break;
        default:
          return InvalidArgument("XLA_NGRAPH_DUMP_GRAPH invalid value: '%s'",
                                 env_str);
          break;
      }
    } else {
      return InvalidArgument("XLA_NGRAPH_DUMP_GRAPH invalid value: '%s'",
                             env_str);
    }
  }

  if (dump_graph) {
    // Print computation before HloPasses
    print_embedded_computation(hlo_module->entry_computation());
    DebugOptions debug_options;
    debug_options.set_xla_generate_hlo_graph(".*");
    std::string graph_url = hlo_graph_dumper::DumpGraph(
        *hlo_module->entry_computation(),
        "HLO Graph received by nGraph Compiler", debug_options, nullptr);
    std::cout << "HLO Graph output: Cluster: "
              << hlo_module->entry_computation()->name()
              << " Graph name: " << graph_url << std::endl;
  }

  TF_RET_CHECK(stream_exec != nullptr);

  // We don't have a layout pass, so we have to set layout to default values
  hlo_module->mutable_entry_computation_layout()->SetToDefaultLayout();

  // TODO: Enable optimization when the ngraph++ has implemented all the
  // operators that are introcuced as a result of the optimization.
  // TF_RETURN_IF_ERROR(RunHloOptimization(hlo_module.get()));

  // Get computation and root_instruction
  xla::HloComputation* computation = hlo_module->entry_computation();
  xla::HloInstruction* root_instruction = computation->root_instruction();
  xla::Shape root_shape = root_instruction->shape();
  NGRAPH_VLOG(2) << "HLO computation: " << computation->ToString();

  NGRAPH_VLOG(1) << "Generate graph " << hlo_module->name();

  // This is a temporary measure until NG++'s API supports control edges.
  NGraphControlEdgeDetector control_edge_detector;
  TF_CHECK_OK(root_instruction->Accept(&control_edge_detector));

  // Build the graph using ngraph ops
  NGraphBuilder builder(computation->parameter_instructions(), &m_fusion_map);
  DfsHloVisitor* hlo_visitor{};
  TF_ASSIGN_OR_RETURN(hlo_visitor, builder.Visitor());

  TF_CHECK_OK(root_instruction->Accept(hlo_visitor));

  // TODO: CLEANUP Remove Debug
  // Will only be activated at NGRAPH_VLOG(3)
  builder.DebugPrintInstructionsList();

  // Create the nGraph function operator
  std::shared_ptr<ngraph::xla::XLAFunction> ng_function;
  TF_ASSIGN_OR_RETURN(ng_function, builder.NGraphFunction(root_instruction));

  // Serialize
  if (const char* env_str = std::getenv("XLA_NGRAPH_ENABLE_SERIALIZE")) {
    int ngraph_enable_serialize_option;
    if (xla::ngraph_plugin::try_parse<int>(env_str,
                                           ngraph_enable_serialize_option)) {
      if (ngraph_enable_serialize_option == 1) {
        std::string file_name = "tf_function_" + computation->name() + ".js";
        NGRAPH_VLOG(1) << "Serializing graph to: " << file_name;
        std::string js = ngraph::serialize(ng_function, 4);
        {
          std::ofstream f(file_name);
          f << js;
        }
      } else if (ngraph_enable_serialize_option != 0) {
        return InvalidArgument(
            "XLA_NGRAPH_ENABLE_SERIALIZE invalid value: '%s', should be 0 or 1",
            env_str);
      }
    } else {
      return InvalidArgument("nGraph backend specified but cannot be parsed");
    }
  }

  // Create the backend
  {
    std::lock_guard<std::mutex> lock(m_module_mutex);
    if (m_ngraph_runtime_manager == nullptr) {
      // Get the appropriate backend
      std::string ngraph_backend_name(XLA_NGRAPH_DEFAULT_BACKEND);
      if (const char* env_str = std::getenv(XLA_NGRAPH_BACKEND_ENV_VAR)) {
        if (xla::ngraph_plugin::try_parse<std::string>(env_str,
                                                       ngraph_backend_name)) {
        } else {
          return InvalidArgument(
              "nGraph backend specified but cannot be parsed");
        }
      }
      NGRAPH_VLOG(1) << "Using ngraph backend: " << ngraph_backend_name;
      m_ngraph_runtime_manager =
          ngraph::runtime::Manager::get(ngraph_backend_name);
    }
  }

  std::shared_ptr<ngraph::runtime::ExternalFunction> ng_runtime_function =
      m_ngraph_runtime_manager->compile(ng_function);

  // Create executable from only the Hlo module
  std::unique_ptr<Executable> executable;
  executable.reset(new NGraphExecutable(
      std::move(hlo_module), m_ngraph_runtime_manager, ng_runtime_function));

  return std::move(executable);
}

//---------------------------------------------------------------------------
// PlatformId
//---------------------------------------------------------------------------
se::Platform::Id NGraphCompiler::PlatformId() const {
  return kNGraphPlatformId;
}

//---------------------------------------------------------------------------
// ShapeSizeBytesFunction
//---------------------------------------------------------------------------
HloCostAnalysis::ShapeSizeFunction NGraphCompiler::ShapeSizeBytesFunction()
    const {
  return NGraphExecutable::ShapeSizeBytes;
}

//---------------------------------------------------------------------------
// Registration
//---------------------------------------------------------------------------
// REGISTER_MODULE_INITIALIZER(ngraph_compiler, {
//  xla::Compiler::RegisterCompilerFactory(
//      kNGraphPlatformId, []() { return xla::MakeUnique<NGraphCompiler>(); });
//});

}  // namespace ngraph_plugin
}  // namespace xla
