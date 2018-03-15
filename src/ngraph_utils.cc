
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

#include "ngraph_utils.h"
#include <cstdlib>
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/window_util.h"

namespace xla {
namespace ngraph_plugin {

ngraph::Shape XLAShapeToNgraphShape(const xla::Shape& s) {
  const int rank = s.dimensions_size();

  ngraph::Shape s_out(rank);
  for (int i = 0; i < rank; ++i) {
    s_out[i] = s.dimensions(i);
  }

  return s_out;
}

tensorflow::Status MaybeAddAxesShuffle(
    const std::shared_ptr<ngraph::Node>& ng_instr_in,
    const ngraph::AxisVector& axes_shuffle,
    std::shared_ptr<ngraph::Node>& ng_instr_out) {
  const long rank = ng_instr_in->get_shape().size();
  TF_RET_CHECK(rank == long(axes_shuffle.size()));
  TF_RET_CHECK(IsPermutationOfZeroBasedVector(axes_shuffle));

  // If 'axes_shuffle' has the identity value [0, 1, ..., (rank-1)], then
  // there's not need
  // to reshape 'xla_instr_in'...
  if (IsLinearIncreasingVector<size_t>(axes_shuffle, size_t(0))) {
    ng_instr_out = ng_instr_in;
    return tensorflow::Status::OK();
  }

  const ngraph::Shape& input_ngraph_shape = ng_instr_in->get_shape();

  const ngraph::Shape output_ngraph_shape =
      GetShuffledVector(input_ngraph_shape, axes_shuffle);

  ng_instr_out = std::make_shared<ngraph::op::Reshape>(
      ng_instr_in, axes_shuffle, output_ngraph_shape);

  return tensorflow::Status::OK();
}

}  // namespace ngraph_plugin
}  // namespace xla
