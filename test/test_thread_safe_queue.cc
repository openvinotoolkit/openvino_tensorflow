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
#include <cstdlib>
#include <thread>
#include <utility>

#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorflow/core/public/session.h"

#include "gtest/gtest.h"
#include "ngraph/event_tracing.hpp"
#include "ngraph_bridge/thread_safe_queue.h"

using namespace std;

namespace tensorflow {
namespace ngraph_bridge {

namespace testing {
TEST(ThreadSafeQueue, Simple) {
  ThreadSafeQueue<Session> queue;
  typedef enum {
    INIT = 0,
    WAITING_FOR_ITEM,
    READY_TO_WAIT,
    GOT_ITEM,
  } CONSUMER_STATE;

  atomic<CONSUMER_STATE> consumer_state{INIT};
  atomic<bool> consumer_do_wait{true};
  atomic<int> item_count{0};

  ngraph::Event::enable_event_tracing();

  // Create two threads
  auto consumer = [&]() {
    while (item_count < 3) {
      ngraph::Event evt_consumer_do_wait("Consumer", "Do Wait", "");
      while (consumer_do_wait) {
        // cout << "\033[1;32mConsumer waiting\033[0m\n";
        absl::SleepFor(absl::Milliseconds(1));
      }
      evt_consumer_do_wait.Stop();
      ngraph::Event evt_consumer_waiting_for_item("Consumer", "Waiting", "");
      consumer_state = WAITING_FOR_ITEM;
      // cout << "\033[1;32mWaiting\033[0m" << endl;
      queue.GetNextAvailable();
      evt_consumer_waiting_for_item.Stop();
      // cout << "\033[1;32mGot Item: " << item_count << "\033[0m\n";
      item_count++;
      consumer_state = GOT_ITEM;
      consumer_do_wait = true;
      // //cout << "Starting waiting" << endl;
      consumer_state = READY_TO_WAIT;
      ngraph::Event::write_trace(evt_consumer_do_wait);
      ngraph::Event::write_trace(evt_consumer_waiting_for_item);
      // cout << "\033[1;32mWaiting for command\033[0m" << endl;
    }
    // cout << "\033[1;34mConsumer completed tasks\033[0m" << endl;
  };

  std::thread thread0(consumer);
  ngraph::Event evt_producer_wait("Producer", "Waiting", "");
  // Ensure that the consumer is in waiting state
  ASSERT_TRUE(consumer_do_wait);

  consumer_do_wait = false;
  while (consumer_state != WAITING_FOR_ITEM) {
    absl::SleepFor(absl::Milliseconds(1));
  }
  evt_producer_wait.Stop();
  ngraph::Event::write_trace(evt_producer_wait);

  // cout << "Now adding an item\n";
  ngraph::Event evt_producer_add("Producer", "Add", "");
  queue.Add(nullptr);
  evt_producer_add.Stop();
  ngraph::Event::write_trace(evt_producer_add);

  // Wait until the consumer has a chance to move forward
  // cout << "Producer: Waiting for consumer to get ready" << endl;

  while (consumer_state != READY_TO_WAIT) {
    absl::SleepFor(absl::Milliseconds(1));
  }
  ASSERT_EQ(item_count, 1);

  // The consumer is now waiting again until consumer_do_wait is signaled
  // Add two more items
  // //cout << "Now adding two items\n";

  ngraph::Event evt_producer_add_again("Producer", "Add-2", "");

  queue.Add(nullptr);
  queue.Add(nullptr);
  evt_producer_add_again.Stop();
  ngraph::Event::write_trace(evt_producer_add_again);

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

}  // namespace ngraph_bridge
}  // namespace tensorflow
