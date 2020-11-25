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

#ifndef THREAD_SAFE_QUEUE_H_
#define THREAD_SAFE_QUEUE_H_
#pragma once

#include <queue>

#include "absl/synchronization/mutex.h"

using namespace std;

namespace benchmark {

template <typename T>
class ThreadSafeQueue {
 public:
  // Return T and a status thing so that in case or termination,
  // caller can check
  // TODO
  T GetNextAvailable() {
    m_mutex.Lock();
    while (m_queue.empty()) {
      m_cv.Wait(&m_mutex);
    }

    T next = std::move(m_queue.front());
    m_queue.pop();
    m_mutex.Unlock();
    return next;
  }

  void Add(T item) {
    m_mutex.Lock();
    m_queue.push(std::move(item));
    m_cv.SignalAll();
    m_mutex.Unlock();
  }

  void Terminate() {
    // TODO
    //
  }

 private:
  queue<T> m_queue;
  absl::CondVar m_cv;
  absl::Mutex m_mutex;
};

}  // namespace benchmark

#endif  // THREAD_SAFE_QUEUE_H_
