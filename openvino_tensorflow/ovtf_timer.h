/*******************************************************************************
 * Copyright (C) 2021-2022 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/
#ifndef OPENVINO_TF_BRIDGE_TIMER_H_
#define OPENVINO_TF_BRIDGE_TIMER_H_

#ifndef _WIN32
#include <unistd.h>
#endif

#include <chrono>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>

namespace tensorflow {
namespace openvino_tensorflow {

class Timer {
 public:
  Timer() : m_start(std::chrono::high_resolution_clock::now()) {
    m_stop = m_start;
  }
  int ElapsedInMS() {
    Stop();
    return std::chrono::duration_cast<std::chrono::milliseconds>(m_stop -
                                                                 m_start)
        .count();
  }
  int ElapsedInMicroSec() {
    Stop();
    return std::chrono::duration_cast<std::chrono::microseconds>(m_stop -
                                                                 m_start)
        .count();
  }
  void Reset() { m_start = std::chrono::high_resolution_clock::now(); }

  void Stop() {
    if (m_stopped) return;
    m_stopped = true;
    m_stop = std::chrono::high_resolution_clock::now();
  }

 private:
  std::chrono::time_point<std::chrono::high_resolution_clock> m_start;
  std::chrono::time_point<std::chrono::high_resolution_clock> m_stop;
  bool m_stopped{false};
};

}  // namespace openvino_tensorflow
}  // namespace tensorflow

#endif  // OPENVINO_TF_BRIDGE_TIMER_H_
