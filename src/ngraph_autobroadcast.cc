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

#include "ngraph_autobroadcast.h"

namespace xla {
namespace ngraph_plugin {

//---------------------------------------------------------------------------
// AutoBroadcast::AutoBroadcast()
//---------------------------------------------------------------------------
AutoBroadcast::AutoBroadcast(NgraphNodePtr lhsNode,
                             const ngraph::Shape& lhsShape,
                             NgraphNodePtr rhsNode,
                             const ngraph::Shape& rhsShape)
    : lhs_({lhsNode, lhsShape}), rhs_({rhsNode, rhsShape}) {
  if (lhsNode == nullptr || rhsNode == nullptr) {
    std::cout << "NGRAPH_BRIDGE: AutoBroadcast: null input pointer"
              << std::endl;
    // TODO: CLEANUP More graceful exit with tensorflow Status
    throw "NGRAPH_BRIDGE: AutoBroadcast: null input pointer";
  }
  if (std::find(lhsShape.begin(), lhsShape.end(), 0) != lhsShape.end() ||
      std::find(rhsShape.begin(), rhsShape.end(), 0) != rhsShape.end()) {
    std::cout << "NGRAPH_BRIDGE: AutoBroadcast: invalid input shape"
              << std::endl;
    // TODO: CLEANUP More graceful exit with tensorflow Status
    throw "NGRAPH_BRIDGE: AutoBroadcast: invalid input shape";
  }

  // if auto broadcast is necessary
  if (lhs_.shape != rhs_.shape) {
    SetShapesAndAxes();

    // if auto broadcast is possible
    if (broadcastshape_.size()) {
      ReshapeAndBroadcast(lhs_);
      ReshapeAndBroadcast(rhs_);
    }
  }
}

//---------------------------------------------------------------------------
// AutoBroadcast::SetShapesAndAxes()
//---------------------------------------------------------------------------
void AutoBroadcast::SetShapesAndAxes() {
  auto lhsSize = lhs_.shape.size();
  auto rhsSize = rhs_.shape.size();
  auto axis = std::max(lhsSize, rhsSize) - 1;

  // per numpy definition of broadcast:
  // start with trailing dimensions and work forward
  // two dimensions are compatible:
  //  * if they are equal
  //  * if one of them is 1
  while (lhsSize >= 1 || rhsSize >= 1) {
    auto lhsDim = lhsSize ? lhs_.shape[lhsSize - 1] : 1;
    auto rhsDim = rhsSize ? rhs_.shape[rhsSize - 1] : 1;

    if (lhsDim == rhsDim) {
      // add dimension to broadcast shape + lhs/rhs reshape
      broadcastshape_.insert(broadcastshape_.begin(), lhsDim);
      lhs_.reshape.insert(lhs_.reshape.begin(), lhsDim);
      rhs_.reshape.insert(rhs_.reshape.begin(), rhsDim);

    } else if (rhsDim == 1) {
      // add lhs dimension to broadcast shape and lhs reshape
      broadcastshape_.insert(broadcastshape_.begin(), lhsDim);
      lhs_.reshape.insert(lhs_.reshape.begin(), lhsDim);
      // add current axis to rhs broadcast axes
      rhs_.axes.insert(rhs_.axes.begin(), axis);

    } else if (lhsDim == 1) {
      // add rhs dimension to broadcast shape and rhs reshape
      broadcastshape_.insert(broadcastshape_.begin(), rhsDim);
      rhs_.reshape.insert(rhs_.reshape.begin(), rhsDim);
      // add current axis to lhs broadcast axes
      lhs_.axes.insert(lhs_.axes.begin(), axis);

    } else {
      // auto broadcast not possible
      broadcastshape_.clear();
      break;
    }

    if (lhsSize) --lhsSize;
    if (rhsSize) --rhsSize;
    if (axis) --axis;
  }
}

//---------------------------------------------------------------------------
// AutoBroadcast::ReshapeAndBroadcast()
//---------------------------------------------------------------------------
void AutoBroadcast::ReshapeAndBroadcast(Node& node) {
  if (node.shape != node.reshape) {
    // tell reshape to examine input dimensions in order
    ngraph::AxisVector order(node.shape.size());
    std::iota(order.begin(), order.end(), 0);
    node.ptr =
        std::make_shared<ngraph::op::Reshape>(node.ptr, order, node.reshape);
  }

  if (broadcastshape_ != node.reshape) {
    node.ptr = std::make_shared<ngraph::op::Broadcast>(
        node.ptr, broadcastshape_, node.axes);
  }
}

}  // namespace ngraph_plugin
}  // namespace xla
