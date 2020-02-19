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

#ifndef NGRAPH_TF_BRIDGE_PIPELINED_TENSORS_H_
#define NGRAPH_TF_BRIDGE_PIPELINED_TENSORS_H_
#pragma once

#include "ngraph/runtime/backend.hpp"

// Consider an ng executable, which has a inputs and b outputs. Let d_input[i]
// be the depth of the pipeline for input i. Similarly d_output[j] is the depth
// of the pipeline for output j.

// Simplifying assumptions about pipeline depths: for all 0 <= i < a, 0 <= j <
// b, d_input[i] ==  d_output[i] == d. Most likely, d = 2

// Pipelined tensors Matrix: When the executable is used to create tensors, it
// will
// create non-ragged matrices of a x d and b x d tensors.

// Input Group m: A set of a input tensors that can be used to feed data to the
// executable. This represents the m'th column of the input pipelined tensor
// matrix defined above

// Output Group n: A set of b input tensors that can be used to collect data
// from the executable. This represents the n'th column of the input pipelined
// tensor matrix defined above

// Simplifying assumption: We assume m == n, that is we use the same pipeline
// depth index when using call() on an executable. Because of this assumption we
// can store the input and output pipelined tensor matrix in the same class
// object. If we decide we can relax this constraint, then we can split up this
// class into 2, one handling inputs, one for outputs.

// To implement the above design, we use the class PipelinedTensorsStore.
// It acts as a store for 2 PipelinedTensorMatrix (input and output) and
// supports 2 public functions get_tensors and return_tensors
// get_tensors: get_tensors is used to get an index (representing the pipeline
// depth)
// and 2 PipelinedTensorVector (for inputs and outputs).
// Note that get_tensors can return -1 as the index to indicate that
// no tensors are available at the moment
// return_tensors: Once we are done using it, we call return_tensors
// with the checked out index from get_tensors to indicate to
// PipelinedTensorsStore
// that we are done using the tensors of that pipeline depth,
// and it can give it to other threads that request tensors.

// PipelinedTensorsStore relies on IndexLibrary to be threadsafe.
// IndexLibrary manages a set of integers: 0,1,...depth-1
// It supports 2 functions get_index and return_index
// get_index returns the smallest int from the set of free indices
// (it returns -1 if none are available)
// return_index accepts back a number that was checkedout earlier
// IndexLibrary can be used safely in a multithreaded scenario since
// the underlying store of free indices is locked by a mutex

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

typedef vector<shared_ptr<ng::runtime::Tensor>> PipelinedTensorVector;
typedef vector<PipelinedTensorVector> PipelinedTensorMatrix;

// IndexLibrary is a class that accepts an unsigned int "depth". This means that
// this class now owns integers from 0, 1, 2, ... depth-1

// See sample usage in test/test_index_library.cpp
class IndexLibrary {
 public:
  IndexLibrary(size_t depth);

  // If available return the smallest free integer (0<=i<depth-1)
  // and if nothing is free, return -1
  // An integer once checked out will never be returned by get_index again,
  // till it is returned using return_index
  int get_index();
  // the user returns a checked out (using get_index) integer,
  // so its available again for reuse when get_index is called again
  void return_index(size_t id);

  // TODO: if needed implement get_depth() and get_num_free_idxs()
  // Implementing get_depth() might make some sense because if one receives an
  // IndexLibrary object that only gives return_index()==-1 then one might want
  // to know is there any point in waiting for it (it will never return anything
  // other than -1 if depth==0). So the user of the object can query depth and
  // throw an error or take appropriate steps if its 0

 private:
  set<int> m_free_depth_indexes;
  size_t m_depth;
  std::mutex m_mtx;  // protects m_free_depth_indexes

  // insert id in m_free_depth_indexes
  void insert_to_free_set(size_t id);
  // check if id already exists in m_free_depth_indexes
  bool is_free(size_t id);
};

class PipelinedTensorsStore {
 public:
  PipelinedTensorsStore(PipelinedTensorMatrix in, PipelinedTensorMatrix out);

  // returns a tuple of idx, and 2 vectors of ng tensors (input and output
  // groups). If the idx is negative, then its an invalid group (because
  // pipeline is filled right now)
  tuple<int, PipelinedTensorVector, PipelinedTensorVector> get_tensors();

  // Return an integer that was checked out by get_tensors.
  // This indicates that the tensors corresponding to depth=id in the pipeline
  // are ready for reuse and can be returned when get_tensors is called again
  void return_tensors(size_t id);

 private:
  PipelinedTensorMatrix m_in_tensors;
  PipelinedTensorMatrix m_out_tensors;
  size_t m_depth;
  shared_ptr<IndexLibrary> idx_lib;

  // Get the i'th depth tensors for inputs if is_input is true, else for outputs
  PipelinedTensorVector get_group(bool is_input, size_t i);
};
}
}

#endif  // NGRAPH_TF_BRIDGE_PIPELINED_TENSORS_H_
