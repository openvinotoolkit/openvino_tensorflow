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
#include "tensorflow/compiler/xla/xla_plugin.h"
#include "ngraph/ngraph.hpp"

#include "executable.h"
#include "transfer_manager.h"

//-----------------------------------------------------------------------------
//  Misc. function declarations
//-----------------------------------------------------------------------------
static void print_embeded_computation(const xla::HloComputation *computation,
                                      int nest_level = 0);
static xla::plugin::DeviceInfo s_DeviceInfo = {
    "nGraphDevice", "NGRAPH", "NGRAPH_JIT", 1};

//-----------------------------------------------------------------------------
//  Plugin Interface Implementation functions
//-----------------------------------------------------------------------------
std::string Version() { return "0.0.0.0"; }
xla::plugin::DeviceInfo DeviceInfo() { return s_DeviceInfo; }
bool Init(perftools::gputools::Platform::Id platform_id)
{
  //std::cout << "Init Called" << std::endl;
  return true;
}
xla::TransferManagerInterface *GetTransferManager()
{
  static std::unique_ptr<xla::TransferManagerInterface> tx_manager =
      std::unique_ptr<xla::TransferManagerInterface>(new TransferManager());
  return tx_manager.get();
}

xla::StatusOr<std::unique_ptr<xla::HloModule>> RunHloPasses(
    std::unique_ptr<xla::HloModule> module,
    perftools::gputools::StreamExecutor *executor,
    xla::DeviceMemoryAllocator *device_allocator)
{
  std::cout << "RunHloPasses called by Plugin adapter\n";
  // Run the HLO optimization passes here
  return std::move(module);
}

// Temp
void nGraphTest();

std::unique_ptr<xla::Executable> RunBackend(
    std::unique_ptr<xla::HloModule> hlo_module,
    ::perftools::gputools::StreamExecutor *stream_exec)
{
  std::cout << "\n======================================================\n"
               "PluginCompiler::Compile()\n"
               "======================================================"
            << std::endl;
  print_embeded_computation(hlo_module->entry_computation());

  // Execute nGraph
  nGraphTest();

  // Create the Executable
  std::unique_ptr<PluginExecutable> executable =
      xla::MakeUnique<PluginExecutable>(std::move(hlo_module),
                                        GetTransferManager());

  return executable;
}

//-----------------------------------------------------------------------------
// Utility functions
//-----------------------------------------------------------------------------
static void print_embeded_computation(const xla::HloComputation *computation,
                                      int nest_level)
{
  auto embedded_computations = computation->MakeEmbeddedComputationsList();
  std::cout << "NGRAPH_COMPILER  computation: " << computation->name()
            << "; nest_level: " << nest_level
            << "; num_embedded: " << embedded_computations.size() << std::endl;
  std::cout << computation->ToString() << std::endl;
  for (auto embedded_computation : embedded_computations)
  {
    print_embeded_computation(embedded_computation, nest_level + 1);
  }
}

//-----------------------------------------------------------------------------
//  GLobal data for this Plugin
//-----------------------------------------------------------------------------

static xla::plugin::Info s_PluginInfo = {
    Version, DeviceInfo, Init, GetTransferManager,
    RunHloPasses, RunBackend, nullptr};

//-----------------------------------------------------------------------------
// DSO Entry point
//-----------------------------------------------------------------------------
extern "C" xla::plugin::Info GetPluginData() { return s_PluginInfo; }

// TEMP
// nGraph Test
using namespace ngraph;
using namespace std;
template <typename T>
void copy_data(std::shared_ptr<ngraph::runtime::TensorView> tv, const std::vector<T> &data)
{
  size_t data_size = data.size() * sizeof(T);
  tv->write(data.data(), 0, data_size);
}

template <typename T>
std::vector<T> read_vector(std::shared_ptr<ngraph::runtime::TensorView> tv)
{
  if (ngraph::element::from<T>() != tv->get_tensor_view_layout()->get_element_type())
  {
    throw std::invalid_argument("read_vector type must match TensorView type");
  }
  size_t element_count = ngraph::shape_size(tv->get_shape());
  size_t size = element_count * sizeof(T);
  std::vector<T> rc(element_count);
  tv->read(rc.data(), 0, size);
  return rc;
}
void nGraphTest()
{
  auto shape = Shape{2, 2};
  auto A = make_shared<op::Parameter>(element::f32, shape);
  auto B = make_shared<op::Parameter>(element::f32, shape);
  auto C = make_shared<op::Parameter>(element::f32, shape);
  auto f = make_shared<Function>((A + B) * C, op::Parameters{A, B, C});

  // Now make "g(X,Y,Z) = f(X,Y,Z) + f(X,Y,Z)"
  auto X = make_shared<op::Parameter>(element::f32, shape);
  auto Y = make_shared<op::Parameter>(element::f32, shape);
  auto Z = make_shared<op::Parameter>(element::f32, shape);
  auto g = make_shared<Function>(make_shared<op::FunctionCall>(f, Nodes{X, Y, Z}) +
                                     make_shared<op::FunctionCall>(f, Nodes{X, Y, Z}),
                                 op::Parameters{X, Y, Z});

  // Now call g on some test vectors.
  auto manager = runtime::Manager::get("INTERPRETER");
  auto external = manager->compile(g);
  auto backend = manager->allocate_backend();
  auto cf = backend->make_call_frame(external);

  auto x = backend->make_primary_tensor_view(element::f32, shape);
  copy_data(x, vector<float>{1, 2, 3, 4});
  auto y = backend->make_primary_tensor_view(element::f32, shape);
  copy_data(y, vector<float>{5, 6, 7, 8});
  auto z = backend->make_primary_tensor_view(element::f32, shape);
  copy_data(z, vector<float>{9, 10, 11, 12});
  auto result = backend->make_primary_tensor_view(element::f32, shape);

  cf->call({x, y, z}, {result});
  auto r1 = read_vector<float>(result);
  for (auto x : r1)
  {
    cout << "Value: " << x << endl;
  }

  //EXPECT_EQ((vector<float>{108, 160, 220, 288}), read_vector<float>(result));

  cf->call({y, x, z}, {result});
  //EXPECT_EQ((vector<float>{108, 160, 220, 288}), read_vector<float>(result));

  cf->call({x, z, y}, {result});
  //EXPECT_EQ((vector<float>{100, 144, 196, 256}), read_vector<float>(result));
}
