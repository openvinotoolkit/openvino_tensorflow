/*******************************************************************************
 * Copyright (C) 2021 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
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
