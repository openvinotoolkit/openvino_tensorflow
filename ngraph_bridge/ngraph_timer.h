/*******************************************************************************
 * Copyright 2019 Intel Corporation
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
#ifndef NGRAPH_TF_BRIDGE_TIMER_H_
#define NGRAPH_TF_BRIDGE_TIMER_H_

#include <unistd.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>

namespace tensorflow {
namespace ngraph_bridge {

class Timer {
 public:
  Timer() : m_start(std::chrono::high_resolution_clock::now()) {
    m_stop = m_start;
  }
  int ElapsedInMS() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::high_resolution_clock::now() - m_start)
        .count();
  }
  void Stop() {
    if (m_stopped) return;
    m_stopped = true;
    m_stop = std::chrono::high_resolution_clock::now();
  }

 private:
  const std::chrono::time_point<std::chrono::high_resolution_clock> m_start;
  std::chrono::time_point<std::chrono::high_resolution_clock> m_stop;
  bool m_stopped{false};
};

}  // namespace ngraph_bridge
}  // namespace tensorflow

#endif  // NGRAPH_TF_BRIDGE_TIMER_H_
