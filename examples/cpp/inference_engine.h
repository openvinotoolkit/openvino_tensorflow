/*******************************************************************************
 * Copyright (C) 2021 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
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
  int m_input_width=0;
  int m_input_height=0;
  float m_input_mean=0.0;
  float m_input_std=0.0;
  string m_input_layer;
  string m_output_layer;
  bool m_use_NCHW=false;
  bool m_preload_images=true;
  int m_input_channels=3;
  Tensor m_image_to_repeat;
};
}  // namespace benchmark
#endif  // _INFERENCE_ENGINE_H_
