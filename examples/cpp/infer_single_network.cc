/*******************************************************************************
 * Copyright 2019-2020 Intel Corporation
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

#include <thread>

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

#include "inference_engine.h"
#include "ngraph_bridge/backend_manager.h"
#include "ngraph_bridge/ngraph_timer.h"
#include "ngraph_bridge/ngraph_utils.h"
#include "ngraph_bridge/version.h"

using namespace std;
namespace tf = tensorflow;

extern tf::Status PrintTopLabels(const std::vector<tf::Tensor>& outputs,
                                 const string& labels_file_name);
extern tf::Status CheckTopLabel(const std::vector<tf::Tensor>& outputs,
                                int expected, bool* is_expected);

// Prints the available backends
void PrintAvailableBackends() {
  // Get the list of backends
  auto supported_backends =
      tf::ngraph_bridge::BackendManager::GetSupportedBackends();
  vector<string> backends(supported_backends.begin(), supported_backends.end());

  cout << "Available backends: " << endl;
  for (auto& backend_name : backends) {
    cout << "Backend: " << backend_name << std::endl;
  }
}

void PrintVersion() {
  // Tensorflow version info
  std::cout << "Tensorflow version: " << tensorflow::ngraph_bridge::tf_version()
            << std::endl;
  // nGraph Bridge version info
  std::cout << "Bridge version: " << tf::ngraph_bridge::version() << std::endl;
  std::cout << "nGraph version: " << tf::ngraph_bridge::ngraph_version()
            << std::endl;
  std::cout << "CXX11_ABI Used: " << tf::ngraph_bridge::cxx11_abi_flag()
            << std::endl;
  std::cout << "Grappler Enabled? "
            << (tf::ngraph_bridge::is_grappler_enabled() ? std::string("Yes")
                                                         : std::string("No"))
            << std::endl;
  PrintAvailableBackends();
}

//-----------------------------------------------------------------------------
//  The benchmark test for inference does the following
//    1. Preloads the input image buffer (currently single image)
//    2. Creates the TensorFlow Session by loading a frozen inference graph
//    3. Starts the worker threads and runs the test for a specifed iterations
//
//  Each worker thread does the following:
//    1. Gets an image from the image pool
//    2. Copies the data to a TensorFlow Tensor
//    3. Runs the inference using the same session used by others as well
//
//-----------------------------------------------------------------------------
int main(int argc, char** argv) {
  // parameters below need to modified as per model
  string image_file = "grace_hopper.jpg";
  int batch_size = 1;
  string graph = "inception_v3_2016_08_28_frozen.pb";
  string labels = "";
  int label_index = -1;
  int input_width = 299;
  int input_height = 299;
  float input_mean = 0.0;
  float input_std = 255;
  string input_layer = "input";
  string output_layer = "InceptionV3/Predictions/Reshape_1";
  bool use_NCHW = false;
  bool preload_images = true;
  int input_channels = 3;
  int iteration_count = 20;

  std::vector<tf::Flag> flag_list = {
      tf::Flag("image", &image_file, "image to be processed"),
      tf::Flag("graph", &graph, "graph to be executed"),
      tf::Flag("labels", &labels, "name of file containing labels"),
      tf::Flag("label_index", &label_index, "Index of the expected label"),
      tf::Flag("input_width", &input_width,
               "resize image to this width in pixels"),
      tf::Flag("input_height", &input_height,
               "resize image to this height in pixels"),
      tf::Flag("input_mean", &input_mean, "scale pixel values to this mean"),
      tf::Flag("input_std", &input_std,
               "scale pixel values to this std deviation"),
      tf::Flag("input_layer", &input_layer, "name of input layer"),
      tf::Flag("output_layer", &output_layer, "name of output layer"),
      tf::Flag("use_NCHW", &use_NCHW, "Input data in NCHW format"),
      tf::Flag("iteration_count", &iteration_count,
               "How many times to repeat the inference"),
      tf::Flag("preload_images", &preload_images,
               "Repeat the same image for inference"),
      tf::Flag(
          "batch_size", &batch_size,
          "Input bach size. The same images is copied to create the batch"),
  };

  string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_result) {
    std::cout << usage;
    return -1;
  }

  // We need to call this to set up global state for TensorFlow.
  tensorflow::port::InitMain(argv[0], &argc, &argv);
  if (argc > 1) {
    std::cout << "Error: Unknown argument " << argv[1] << "\n" << usage;
    return -1;
  }

  std::cout << "Component versions\n";
  PrintVersion();

  // If batch size is more than one then expand the input
  vector<string> image_files;
  for (int i = 0; i < batch_size; i++) {
    image_files.push_back(image_file);
  }
  // Instantiate the Engine
  benchmark::InferenceEngine inference_engine("Foo");
  TF_CHECK_OK(inference_engine.LoadImage(
      graph, image_files, input_width, input_height, input_mean, input_std,
      input_layer, output_layer, use_NCHW, preload_images, input_channels));

  unique_ptr<Session> the_session;
  TF_CHECK_OK(benchmark::InferenceEngine::CreateSession(graph, the_session));

  Tensor next_image;
  std::vector<Tensor> outputs;
  {
    NG_TRACE("Compilation", "Compilation", "");

    // Call it onces to get the nGraph compilation done
    TF_CHECK_OK(inference_engine.GetNextImage(next_image));
    // Run inference once. This will trigger a compilation
    tf::ngraph_bridge::Timer compilation_time;
    TF_CHECK_OK(the_session->Run({{input_layer, next_image}}, {output_layer},
                                 {}, &outputs));
    compilation_time.Stop();

    cout << "Compilation took: " << compilation_time.ElapsedInMS() << " ms"
         << endl;
  }

  atomic<int> total_time_in_ms{0};
  atomic<int> total_images_processed{0};

  auto worker = [&](int worker_id) {
    ostringstream oss;
    oss << "Worker_" << worker_id;
    for (int i = 0; i < iteration_count; i++) {
      NG_TRACE(oss.str(), to_string(i), "");
      tf::ngraph_bridge::Timer iteration_timer;
      // Get the image
      Tensor next_image;
      TF_CHECK_OK(inference_engine.GetNextImage(next_image));

      // Run inference
      TF_CHECK_OK(the_session->Run({{input_layer, next_image}}, {output_layer},
                                   {}, &outputs));

      // End. Mark time
      iteration_timer.Stop();
      total_time_in_ms += iteration_timer.ElapsedInMS();
      cout << "Iteration: " << i << " Time: " << iteration_timer.ElapsedInMS()
           << endl;
      total_images_processed++;
    }
  };

  tf::ngraph_bridge::Timer benchmark_timer;
  std::thread thread0(worker, 0);
  std::thread thread1(worker, 1);
  std::thread thread2(worker, 2);

  thread0.join();
  thread1.join();
  thread2.join();
  benchmark_timer.Stop();

  // Adjust the total images with the batch size
  total_images_processed = total_images_processed * batch_size;

  cout << "Time for each image: "
       << ((float)total_time_in_ms / (float)total_images_processed) << " ms"
       << endl;
  cout << "Images/Sec: "
       << (float)total_images_processed /
              (benchmark_timer.ElapsedInMS() / 1000.0)
       << endl;
  cout << "Total frames: " << total_images_processed << "\n";
  cout << "Total time: " << benchmark_timer.ElapsedInMS() << " ms\n";
  // Validate the label if provided
  if (!labels.empty()) {
    cout << "Classification results\n";
    // Validate the label
    PrintTopLabels(outputs, labels);
    if (label_index != -1) {
      bool found = false;
      CheckTopLabel(outputs, label_index, &found);
      if (!found) {
        cout << "Error - label doesn't match expected\n";
        return -1;
      }
    }
  }
  return 0;
}
