/* Copyright 2017-2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef NGRAPH_UTILS_H_
#define NGRAPH_UTILS_H_

#include <dlfcn.h>
#include <stdio.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include "ngraph/ngraph.hpp"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/platform/macros.h"

namespace xla {
namespace ngraph_plugin {

template <typename Iter>
std::string container2string(Iter it_begin, Iter it_end) {
  std::stringstream result;
  result << "[";
  std::copy(it_begin, it_end,
            std::ostream_iterator<decltype(*it_begin)>(result, ", "));
  result << "]";
  return result.str();
}

// Attempt to parse 's' as an object of type 'T'.  If successful, set 'out' to
// the resulting value
// and return true.  Otherwise return false.
template <typename T>
bool try_parse(const std::string s, T& out) {
  std::istringstream iss(s);
  T temp;
  if ((iss >> temp) && (iss.eof())) {
    out = temp;
    return true;
  } else {
    return false;
  }
}

//_shape_to_ngraph_shape Return true iff \p v has exactly the value:
/// [ (start_val), (start_val + 1*increment_val), (start_val + 2*increment_val),
/// ... ]
template <typename T>
bool IsLinearIncreasingVector(const std::vector<T>& v, const T start_val,
                              const T increment_val = 1) {
  for (size_t i = 1; i < v.size(); ++i) {
    const T delta = v[i] - v[i - 1];
    if (delta != increment_val) {
      return false;
    }
  }

  return true;
}

/// Return true iff \p v is a permutation of the vector
/// [ 0, 1, ..., (v.size()-1) ].
template <typename T>
bool IsPermutationOfZeroBasedVector(const std::vector<T>& v) {
  std::vector<T> v2 = v;
  std::sort(v2.begin(), v2.end());
  return IsLinearIncreasingVector(v2, T(0));
};

/// Every element of \p v_permutations must be a valid index into \p v_src.
/// Return a new vector whose elements are drawn from \p v_src, according to the
/// indices specifeid in \p v_permutations.
/// There are no additional assumptions about the relationship between \p v_src
/// and \p v_permutations.
/// Throw an exception if any problem is encountered.
template <typename T>
std::vector<T> GetShuffledVector(const std::vector<T>& v_src,
                                 const std::vector<size_t>& v_permutations) {
  std::vector<T> v2;
  v2.reserve(v_permutations.size());

  for (size_t i : v_permutations) {
    v2.push_back(v_src.at(i));
  }

  return v2;
}

ngraph::Shape XLAShapeToNgraphShape(const xla::Shape& s);

/// If \p axes_permutation is the vector [0, 1, ..., (rank-1)], then do not
/// create
/// any additional graph nodes.
///
/// If \p axes_permutation is some other permutation of [0, 1, ..., (rank-1)],
/// then wrap
/// \p ng_instr_in with an nGraph Reshape operation that effects the specified
/// axis-reordering.
///
/// \param[in] ng_instr_in
/// \param[in] axes_shuffle
/// \param[out] ng_instr_out - If no new Reshape operation was created, set this
///    to \p ng_instr_in.  Otherwise set this to point at the Reshape operation
///    created
///    by this function.  If the function returns an error status, the output
///    value of this
///    parameter is unspecified.
tensorflow::Status MaybeAddAxesShuffle(
    const std::shared_ptr<ngraph::Node>& ng_instr_in,
    const ngraph::AxisVector& axes_shuffle,
    std::shared_ptr<ngraph::Node>& ng_instr_out);

}  // namespace ngraph_plugin
}  // namespace xla

#endif  // NGRAPH_UTILS_H_
