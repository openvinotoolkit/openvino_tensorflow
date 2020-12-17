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
#include "tensorflow/cc/ops/standard_ops.h"
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

#include "ngraph_bridge/backend_manager.h"
#include "ngraph_bridge/ngraph_timer.h"
#include "ngraph_bridge/ngraph_utils.h"
#include "ngraph_bridge/version.h"

#include "inference_engine.h"
#include "thread_safe_queue.h"

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

// Sets the specified backend. This backend must be set BEFORE running
// the computation
tf::Status SetNGraphBackend(const string& backend_name) {
  // Select a backend
  tf::Status status =
      tf::ngraph_bridge::BackendManager::SetBackend(backend_name);
  return status;
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
//    2. Creates the TensorFlow Session by loading a frozem inference graph
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
  int num_threads = 3;

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
      tf::Flag("num_threads", &num_threads, "Number of threads to use."),
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

  const char* backend = "CPU";
  if (SetNGraphBackend(backend) != tf::Status::OK()) {
    std::cout << "Error: Cannot set the backend: " << backend << std::endl;
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

  //
  // Create the sessions
  //
  map<Session*, string> session_db;
  unique_ptr<Session> session_one;
  TF_CHECK_OK(benchmark::InferenceEngine::CreateSession(graph, session_one));
  session_db[session_one.get()] = "One";

  unique_ptr<Session> session_two;
  TF_CHECK_OK(benchmark::InferenceEngine::CreateSession(graph, session_two));
  session_db[session_two.get()] = "Two";

  unique_ptr<Session> session_three;
  TF_CHECK_OK(benchmark::InferenceEngine::CreateSession(graph, session_three));
  session_db[session_three.get()] = "Three";
  std::vector<Tensor> outputs;
  {
    NG_TRACE("Compilation", "Compilation", "");

    //
    // Warm-up i.e., Call it onces to get the nGraph compilation done
    //
    Tensor next_image;
    TF_CHECK_OK(inference_engine.GetNextImage(next_image));
    // Run inference once. This will trigger a compilation
    tf::ngraph_bridge::Timer compilation_time;
    TF_CHECK_OK(session_one->Run({{input_layer, next_image}}, {output_layer},
                                 {}, &outputs));
    TF_CHECK_OK(session_two->Run({{input_layer, next_image}}, {output_layer},
                                 {}, &outputs));
    TF_CHECK_OK(session_three->Run({{input_layer, next_image}}, {output_layer},
                                   {}, &outputs));
    compilation_time.Stop();

    cout << "Compilation took: " << compilation_time.ElapsedInMS() << " ms"
         << endl;
  }
  //
  // Add these sessions to the queue
  //
  benchmark::ThreadSafeQueue<unique_ptr<Session>> session_queue;
  session_queue.Add(move(session_one));
  session_queue.Add(move(session_two));
  session_queue.Add(move(session_three));

  cout << "Session: " << session_db[session_one.get()] << "\n";
  unordered_map<Session*, pair<float, float>> session_stats;

  //------------------------------------
  // Worker thread function
  //------------------------------------
  auto worker = [&](int worker_id) {
    ostringstream oss;
    oss << "Worker" << worker_id;
    std::vector<Tensor> output_each_thread;

    unordered_map<Session*, pair<float, float>> local_stats;
    unordered_map<Session*, int> num_items;

    //-----------------------------------------
    // Run the inference loop
    //-----------------------------------------
    for (int i = 0; i < iteration_count; i++) {
      NG_TRACE(oss.str(), to_string(i), "");

      tf::ngraph_bridge::Timer get_image_timer;
      //
      // Get the image
      //
      Tensor next_image;
      TF_CHECK_OK(inference_engine.GetNextImage(next_image));
      get_image_timer.Stop();

      //
      // Get the next available network model (i.e., session)
      //
      tf::ngraph_bridge::Timer execute_inference_timer;
      unique_ptr<Session> next_available_session;
      {
        NG_TRACE("Get Session", string("Iteration") + to_string(i), "");
        next_available_session = session_queue.GetNextAvailable();
      }

      //
      // Run inference on this network model (i.e., session)
      //
      {
        NG_TRACE("Run Session", string("Iteration") + to_string(i), "");
        TF_CHECK_OK(next_available_session->Run({{input_layer, next_image}},
                                                {output_layer}, {},
                                                &output_each_thread));
      }
      Session* next_session_ptr = next_available_session.get();
      session_queue.Add(move(next_available_session));
      execute_inference_timer.Stop();

      //
      // Update the stats
      //
      local_stats[next_session_ptr].first += get_image_timer.ElapsedInMS();
      local_stats[next_session_ptr].second +=
          execute_inference_timer.ElapsedInMS();
      num_items[next_session_ptr]++;
    }

    //-----------------------------------------
    // Calculate the average across all the
    // sessions used by this thread
    //-----------------------------------------
    map<Session*, vector<float>> get_avg;
    map<Session*, vector<float>> infer_avg;
    for (auto& next : num_items) {
      Session* next_session = next.first;
      int total_inferences = next.second;
      float avg_get_time = local_stats[next_session].first / total_inferences;
      float avg_infer_time =
          local_stats[next_session].second / total_inferences;

      get_avg[next_session].push_back(avg_get_time);
      infer_avg[next_session].push_back(avg_infer_time);
    }

    // Now calcuate the average for each session
    unordered_map<Session*, pair<float, float>> per_session_stats;
    for (auto& next : get_avg) {
      Session* next_session = next.first;
      int total_inferences = next.second.size();
      float avg_img_get_time = 0.0;
      for (int i = 0; i < total_inferences; i++) {
        avg_img_get_time += next.second[i];
      }
      avg_img_get_time = avg_img_get_time / total_inferences;

      per_session_stats[next_session].first = avg_img_get_time;
    }

    for (auto& next : infer_avg) {
      Session* next_session = next.first;
      int total_inferences = next.second.size();

      float avg_infer_time = 0.0;
      for (int i = 0; i < total_inferences; i++) {
        avg_infer_time += next.second[i];
      }

      per_session_stats[next_session].second = avg_infer_time;
    }

    // Print the stats
    for (auto& next : per_session_stats) {
      Session* next_session = next.first;
      cout << "Worker: " << worker_id
           << " Session: " << session_db[next_session]
           << " Get Img Avg: " << next.second.first << " ms "
           << " Inf Avg: " << next.second.second << " ms "
           << "\n";
    }
  };

  //
  // Spawn the threads
  //
  tf::ngraph_bridge::Timer benchmark_timer;
  vector<thread> threads;

  for (int i = 0; i < num_threads; i++) {
    std::thread thread_next(worker, i);
    threads.push_back(move(thread_next));
  }

  for (int i = 0; i < num_threads; i++) {
    threads[i].join();
  }

  benchmark_timer.Stop();
  cout << "Total time: " << benchmark_timer.ElapsedInMS() << " ms\n";

  //
  // Validate the label if provided
  //
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
