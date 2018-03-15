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

#include <iostream>
#include "ngraph/ngraph.hpp"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/xla_plugin.h"

#include "executable.h"
#include "ngraph_compiler.h"
#include "transfer_manager.h"

//-----------------------------------------------------------------------------
//  Misc. function declarations
//-----------------------------------------------------------------------------

static void print_embeded_computation(const xla::HloComputation* computation,
                                      int nest_level = 0);
static xla::plugin::DeviceInfo s_DeviceInfo = {"nGraphDevice", "NGRAPH",
                                               "NGRAPH_JIT", 1};

//-----------------------------------------------------------------------------
//  We keep a singleton instance of the NGraphCompiler object.
//-----------------------------------------------------------------------------
static xla::ngraph_plugin::NGraphCompiler s_Compiler;

//-----------------------------------------------------------------------------
//  Plugin Interface Implementation functions
//-----------------------------------------------------------------------------
std::string Version() { return "0.0.0.0"; }
xla::plugin::DeviceInfo DeviceInfo() { return s_DeviceInfo; }
bool Init(perftools::gputools::Platform::Id platform_id) {
  // std::cout << "Init Called" << std::endl;
  return true;
}
xla::TransferManagerInterface* GetTransferManager() {
  static std::unique_ptr<xla::TransferManagerInterface> tx_manager =
      std::unique_ptr<xla::TransferManagerInterface>(new TransferManager());
  return tx_manager.get();
}

// from NGraphCompiler
xla::StatusOr<std::unique_ptr<xla::HloModule>> RunHloPasses(
    std::unique_ptr<xla::HloModule> module,
    perftools::gputools::StreamExecutor* executor,
    xla::DeviceMemoryAllocator* device_allocator) {
  return s_Compiler.RunHloPasses(std::move(module), executor, device_allocator)
      .ValueOrDie();
}

// from NGraphCompiler
std::unique_ptr<xla::Executable> RunBackend(
    std::unique_ptr<xla::HloModule> hlo_module,
    ::perftools::gputools::StreamExecutor* stream_exec) {
  // TODO (TEMP HACK)
  //
  // This is what we really want to do once ExecuteOnStream is implemented, but
  // for the moment we are just calling it for its side effects, then returning
  // a PluginExecutable that will run through HloEvaluator:
  //
  // return s_Compiler.RunBackend(std::move(hlo_module), stream_exec,
  // /*device_allocator=*/nullptr).ValueOrDie();

  auto hlo_module_clone = hlo_module->Clone();
  auto dummy = s_Compiler
                   .RunBackend(std::move(hlo_module), stream_exec,
                               /*device_allocator=*/nullptr)
                   .ValueOrDie();

  std::cout << "\n======================================================\n"
               "PluginCompiler::Compile()\n"
               "======================================================"
            << std::endl;
  print_embeded_computation(hlo_module_clone->entry_computation());

  // Create the Executable
  std::unique_ptr<PluginExecutable> executable =
      xla::MakeUnique<PluginExecutable>(std::move(hlo_module_clone),
                                        GetTransferManager());

  return executable;
}

//-----------------------------------------------------------------------------
// Utility functions
//-----------------------------------------------------------------------------
static void print_embeded_computation(const xla::HloComputation* computation,
                                      int nest_level) {
  auto embedded_computations = computation->MakeEmbeddedComputationsList();
  std::cout << "NGRAPH_COMPILER  computation: " << computation->name()
            << "; nest_level: " << nest_level
            << "; num_embedded: " << embedded_computations.size() << std::endl;
  std::cout << computation->ToString() << std::endl;
  for (auto embedded_computation : embedded_computations) {
    print_embeded_computation(embedded_computation, nest_level + 1);
  }
}

//-----------------------------------------------------------------------------
//  Global data for this Plugin
//-----------------------------------------------------------------------------

static xla::plugin::Info s_PluginInfo = {
    Version,      DeviceInfo, Init,   GetTransferManager,
    RunHloPasses, RunBackend, nullptr};

//-----------------------------------------------------------------------------
// DSO Entry point
//-----------------------------------------------------------------------------
extern "C" xla::plugin::Info GetPluginData() { return s_PluginInfo; }
