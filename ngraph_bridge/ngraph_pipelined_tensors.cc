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

#include "ngraph_bridge/ngraph_pipelined_tensors.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

IndexLibrary::IndexLibrary(size_t depth) : m_depth(depth) {
  for (size_t i = 0; i < depth; i++) {
    m_free_depth_indexes.insert(i);
  }
}

void IndexLibrary::return_index(size_t id) {
  if (m_depth == 0) {
    throw std::runtime_error(
        "Depth=0, so no one should be calling return_index");
  }
  if (id > m_depth - 1) {
    throw std::runtime_error("Depth = " + to_string(m_depth) +
                             " but passed an index to return ( = " +
                             to_string(id) + "), which is too large");
  }
  if (is_free(id)) {
    throw std::runtime_error(
        "Attempted to return index " + to_string(id) +
        " but it is already present in the free indices set");
  }

  insert_to_free_set(id);
}

int IndexLibrary::get_index() {
  if (m_depth == 0) {
    return -1;
  } else {
    std::lock_guard<std::mutex> lock(m_mtx);
    if (m_free_depth_indexes.size() == 0) {
      return -1;
    } else {
      // Find and return the smallest free integer
      int min_idx = m_depth;  // nothing can be >= m_depth
      for (auto i : m_free_depth_indexes) {
        if (i < min_idx) {
          min_idx = i;
        }
      }
      if (min_idx >= m_depth || min_idx < 0) {
        throw std::runtime_error(
            "get_index can only return values between 0 to depth-1 from here, "
            "but attempted to return " +
            to_string(min_idx));
      }
      m_free_depth_indexes.erase(min_idx);
      return min_idx;
    }
  }
}

void IndexLibrary::insert_to_free_set(size_t id) {
  std::lock_guard<std::mutex> lock(m_mtx);
  m_free_depth_indexes.insert(id);
}

bool IndexLibrary::is_free(size_t id) {
  std::lock_guard<std::mutex> lock(m_mtx);
  if (id > m_depth - 1) {
    throw std::runtime_error("Asked to check if id=" + to_string(id) +
                             " is free, but depth=" + to_string(m_depth));
  }
  return m_free_depth_indexes.find(id) != m_free_depth_indexes.end();
}

PipelinedTensorsStore::PipelinedTensorsStore(PipelinedTensorMatrix in,
                                             PipelinedTensorMatrix out)
    : m_in_tensors(in), m_out_tensors(out) {
  auto m_depth_in = in.size();
  auto m_depth_out = out.size();

  if (m_depth_in != m_depth_out) {
    throw std::runtime_error(
        "Input and Output depth does not match. Input depth: " +
        to_string(m_depth_in) + " Output depth: " + to_string(m_depth_out));
  }

  // We assume that input and output depths are same
  m_depth = m_depth_in;

  idx_lib = make_shared<IndexLibrary>(m_depth);
}

tuple<int, PipelinedTensorVector, PipelinedTensorVector>
PipelinedTensorsStore::get_tensors() {
  int i = idx_lib->get_index();
  return make_tuple(i, (i < 0 ? PipelinedTensorVector{} : get_group(true, i)),
                    (i < 0 ? PipelinedTensorVector{} : get_group(false, i)));
}

void PipelinedTensorsStore::return_tensors(size_t id) {
  idx_lib->return_index(id);
}

PipelinedTensorVector PipelinedTensorsStore::get_group(bool is_input,
                                                       size_t i) {
  if (is_input) {
    return m_in_tensors[i];
  } else {
    return m_out_tensors[i];
  }
}
}
}
