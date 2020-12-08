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
#ifndef _INFERENCE_ENGINE_H_
#define _INFERENCE_ENGINE_H_

#include <unistd.h>
#include <chrono>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/public/session.h"

using tensorflow::Status;
using tensorflow::Session;
using tensorflow::Tensor;

using std::string;
using std::unique_ptr;
using std::thread;
using std::atomic;
using std::function;

namespace benchmark {
class InferenceEngine {
 public:
  InferenceEngine(const string& name);
  ~InferenceEngine();

  Status LoadImage(const string& network,
                   const std::vector<string>& image_files, int input_width,
                   int input_height, float input_mean, float input_std,
                   const string& input_layer, const string& output_layer,
                   bool use_NCHW, bool preload_images, int input_channels);

  Status GetNextImage(Tensor& image) const {
    image = m_image_to_repeat;
    return Status::OK();
  }

  static Status CreateSession(const string& network,
                              unique_ptr<Session>& session);

 private:
  const string m_name;
  unique_ptr<Session> m_session;
  thread m_worker;
  atomic<bool> m_terminate_worker{false};
  std::function<void(int)> m_step_callback{nullptr};

  // Image related info
  std::vector<string> m_image_files;
  int m_input_width;
  int m_input_height;
  float m_input_mean;
  float m_input_std;
  string m_input_layer;
  string m_output_layer;
  bool m_use_NCHW;
  bool m_preload_images;
  int m_input_channels;
  Tensor m_image_to_repeat;
};
}  // namespace benchmark
#endif  // _INFERENCE_ENGINE_H_
