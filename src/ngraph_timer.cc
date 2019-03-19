/*******************************************************************************
 * Copyright 2017-2018 Intel Corporation
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

#include "ngraph_timer.h"

namespace tensorflow {
namespace ngraph_bridge {
std::mutex Event::s_file_mutex;
std::ofstream Event::s_event_log;

void Event::WriteTrace(const Event& event) {
  std::lock_guard<std::mutex> lock(s_file_mutex);
  if (!IsTracingEnabled()) return;

  static bool initialized = false;
  if (!initialized) {
    // Open the file
    s_event_log.open("ngraph_event_trace.json", std::ios_base::trunc);
    s_event_log << "[\n";
    s_event_log << event << "\n";
    initialized = true;
    return;
  }

  s_event_log << ",\n";
  s_event_log << event << "\n" << std::flush;
}
std::ostream& operator<<(std::ostream& out_stream, const Event& event) {
  out_stream << "{"
             << "\"name\": \"" << event.m_name << "\", \"cat\": \""
             << event.m_category << "\", "
             << "\"ph\": \"B\", \"ts\": "
             << event.m_start.time_since_epoch().count() / 1000 << ", "
             << "\"pid\": " << event.m_pid
             << ", \"tid\": " << std::this_thread::get_id() << ","
             << "\n\"args\":\n\t{\n\t\t\"arg1\": "
             << "\"" << event.m_args << "\"\n\t}\n"
             << "},\n";

  out_stream << "{"
             << "\"name\": \"" << event.m_name << "\", \"cat\": \""
             << event.m_category << "\", "
             << "\"ph\": \"E\", \"ts\": "
             << event.m_stop.time_since_epoch().count() / 1000 << ", "
             << "\"pid\": " << event.m_pid
             << ", \"tid\": " << std::this_thread::get_id() << ", "
             << "\n\"args\":\n\t{\n\t\t\"arg1\": "
             << "\"" << event.m_args << "\"\n\t}\n"
             << "}\n";

  return out_stream;
}

}  // namespace ngraph_bridge
}  // namespace tensorflow
