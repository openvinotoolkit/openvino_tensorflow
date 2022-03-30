/*******************************************************************************
 * Copyright (C) 2021-2022 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/
#include <cstdlib>
#include <thread>
#include <utility>

#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorflow/core/public/session.h"

#include "../examples/classification_sample/thread_safe_queue.h"
#include "gtest/gtest.h"
#include "openvino_tensorflow/ovtf_utils.h"

using namespace std;

namespace tensorflow {
namespace openvino_tensorflow {

namespace testing {
TEST(ThreadSafeQueue, Simple) {
  benchmark::ThreadSafeQueue<unique_ptr<Session>> queue;
  typedef enum {
    INIT = 0,
    WAITING_FOR_ITEM,
    READY_TO_WAIT,
    GOT_ITEM,
  } CONSUMER_STATE;

  atomic<CONSUMER_STATE> consumer_state{INIT};
  atomic<bool> consumer_do_wait{true};
  atomic<int> item_count{0};

  // Create two threads
  auto consumer = [&]() {
    while (item_count < 3) {
      {
        while (consumer_do_wait) {
          // cout << "\033[1;32mConsumer waiting\033[0m\n";
          absl::SleepFor(absl::Milliseconds(1));
        }
      }

      {
        consumer_state = WAITING_FOR_ITEM;
        // cout << "\033[1;32mWaiting\033[0m" << endl;
        queue.GetNextAvailable();
      }
      // cout << "\033[1;32mGot Item: " << item_count << "\033[0m\n";
      item_count++;
      consumer_state = GOT_ITEM;
      consumer_do_wait = true;
      // //cout << "Starting waiting" << endl;
      consumer_state = READY_TO_WAIT;
      // cout << "\033[1;32mWaiting for command\033[0m" << endl;
    }
    // cout << "\033[1;34mConsumer completed tasks\033[0m" << endl;
  };

  std::thread thread0(consumer);
  {
    // Ensure that the consumer is in waiting state
    ASSERT_TRUE(consumer_do_wait);

    consumer_do_wait = false;
    while (consumer_state != WAITING_FOR_ITEM) {
      absl::SleepFor(absl::Milliseconds(1));
    }
  }

  // cout << "Now adding an item\n";
  { queue.Add(nullptr); }
  // Wait until the consumer has a chance to move forward
  // cout << "Producer: Waiting for consumer to get ready" << endl;

  while (consumer_state != READY_TO_WAIT) {
    absl::SleepFor(absl::Milliseconds(1));
  }
  ASSERT_EQ(item_count, 1);

  // The consumer is now waiting again until consumer_do_wait is signaled
  // Add two more items
  // //cout << "Now adding two items\n";
  {
    queue.Add(nullptr);
    queue.Add(nullptr);
  }

  // cout << "Producer: Waiting for consumer to get ready" << endl;

  // Wait until the consumer pulls one item from the queue
  // and ready to receive the next command
  while (consumer_state != READY_TO_WAIT) {
    absl::SleepFor(absl::Milliseconds(1));
  }

  consumer_do_wait = false;
  // cout << "Producer: Done Waiting for consumer to get ready" << endl;

  while (item_count != 3) {
    absl::SleepFor(absl::Milliseconds(1));
    consumer_do_wait = false;
    // cout << "Producer waiting\n";
  }

  thread0.join();
}
}

}  // namespace openvino_tensorflow
}  // namespace tensorflow
