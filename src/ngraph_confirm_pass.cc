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

#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/graph/graph.h"

#include "ngraph_utils.h"

using namespace std;
namespace ngraph_bridge {

// TODO(amprocte): this decl should probably be in a header.
extern const char* const DEVICE_NGRAPH;

//
// In some cases, we require more complex placement constraints than than
// TensorFlow's native "soft-placement" machinery is capable of handling. To
// handle this, we insert a pass called the "confirmation" pass during the
// pre-placement phase.
//
// For example, we can only handle Reshape if the "shape" input is a constant,
// so this is okay:
//
//   Foo       Const[2,4,2]
//     \       /
//      Reshape                     (1)
//
// but this is not:
//
//   Foo       Placeholder
//     \       /
//      Reshape                     (2)
//
// We want to reject placement of Reshape on NGRAPH for the second graph, but
// allow it for the first. We also want to attach some more metadata to the
// Reshape node so that we can remember the requested output shape even if the
// Const node winds up being placed in a different subgraph.
//
// This pass exploits a feature of the placement engine that allows a kernel
// builder registration request to restrict use of the kernel to nodes that
// have a particular value set for the "_kernel" attribute. In this case, we
// will check every node that has a requested placement on NGRAPH, and make
// sure that it conforms to certain (op-dependent) constraints. If the
// constraints are satisfied, we will tag the node with a "_kernel" value of
// "ngraph". The stub kernels, in turn, are registered with the constraint
// that _kernel="ngraph". This means that during the placement pass, our
// kernels will not be allowed for nodes we did not mark during this pass, and
// placement will fall back on CPU.
//
// The per-op checks are implemented by callbacks of the type:
//
//      std::function<tf::Status(tf::Node*, bool*)>.
//
// A confirmation function should return true/false by reference through its
// second parameter: true if placement is "accepted", and false if it is
// "rejected". For example, the confirmation function for "Reshape" will return
// true for (1) above, and false for (2) above.
//
// A confirmation function can also, as a side effect, add attributes to the
// node being checked, which can be used later in ngraph_builder. In this case,
// the "Reshape" confirmation function extracts the tensor data from the Const
// input node, and adds an attribute to the "Reshape" node with the name
// "_ngraph_reshape_static_shape", which is an array of int values (here, the
// values [2,4,2]). (Note that in general such attributes will need to start
// with "_" to mark them as "internal" or "system" attributes, as otherwise
// TensorFlow attempts to validate them as against the op schema.)
//
class NGraphConfirmPass : public tensorflow::GraphOptimizationPass {
 public:
  tf::Status Run(const tf::GraphOptimizationPassOptions& options) {
    return ConfirmPlacement(options.graph->get());
  }

 private:
  using ConfirmationFunction = std::function<tf::Status(tf::Node*, bool*)>;

  //
  // Utility function to check if placement on the NGRAPH device has been
  // requested.
  //
  static bool NGraphPlacementRequested(const tf::Node* node) {
    tf::DeviceNameUtils::ParsedName parsed;

    if (!tf::DeviceNameUtils::ParseFullName(node->requested_device(),
                                            &parsed)) {
      return false;
    }

    return (parsed.has_type && parsed.type == DEVICE_NGRAPH);
  }

  //
  // Utility function to extract data from a constant node used to express a
  // shape (or strides, axis indices, etc.). Only works on nodes of data type
  // INT32 or INT64.
  //
  static tf::Status ExtractConstantData(tf::Node* node,
                                        std::vector<tf::int64>* values) {
    if (node->type_string() != "Const") {
      return tf::errors::InvalidArgument(
          "Tried to extract constant data from a non-Const node");
    }

    tf::DataType dtype;
    TF_RETURN_IF_ERROR(tf::GetNodeAttr(node->attrs(), "dtype", &dtype));

    tf::TensorShapeProto shape_proto;

    switch (dtype) {
      case tf::DataType::DT_INT32: {
        std::vector<tf::int32> values_int32;
        TF_RETURN_IF_ERROR(ValuesFromConstNode<tf::int32>(
            node->def(), &shape_proto, &values_int32));
        values->resize(values_int32.size());
        for (size_t i = 0; i < values_int32.size(); i++) {
          (*values)[i] = (tf::int64)values_int32[i];
        }
      } break;
      case tf::DataType::DT_INT64:
        TF_RETURN_IF_ERROR(
            ValuesFromConstNode<tf::int32>(node->def(), &shape_proto, values));
        break;
      default:
        return tf::errors::InvalidArgument(
            "Tried to extract constant data from a Const node that is neither "
            "DT_INT32 nor DT_INT64");
    }

    return tf::Status::OK();
  }

  //
  // Main entry point for the confirmation pass.
  //
  tf::Status ConfirmPlacement(tf::Graph* graph) {
    // A map of op types (e.g. "Add") to confirmation functions.
    static std::map<std::string, ConfirmationFunction> confirmation_functions;

    tf::mutex init_mu;
    static bool initialized = false;

    // If the confirmation function map has not been initialized, initialize
    // it.
    //
    // IF YOU ARE ADDING A NEW OP IMPLEMENTATION, ADD A CONFIRMATION FUNCTION
    // FOR THE OP HERE.
    {
      tf::mutex_lock l(init_mu);

      if (!initialized) {
        // Trivial confirmation function which always accepts placement.
        ConfirmationFunction always = [](tf::Node* n, bool* result) {
          *result = true;
          return tf::Status::OK();
        };
        // Trivial confirmation function which always rejects placement.
        ConfirmationFunction never = [](tf::Node* n, bool* result) {
          *result = false;
          return tf::Status::OK();
        };

        //
        // Please keep these in alphabetical order.
        //
        confirmation_functions["Abs"] = always;
        confirmation_functions["Add"] = always;
        confirmation_functions["AvgPool"] = always;
        confirmation_functions["BiasAdd"] = always;

        // Constraint: axis selection input must be Const.
        confirmation_functions["ConcatV2"] = [](tf::Node* n, bool* result) {
          tf::Node* tf_axis_node;
          TF_RETURN_IF_ERROR(n->input_node(n->num_inputs() - 1, &tf_axis_node));

          std::vector<tf::int64> tf_static_axis;
          if (ExtractConstantData(tf_axis_node, &tf_static_axis) !=
                  tf::Status::OK() ||
              tf_static_axis.size() != 1) {
            *result = false;
            return tf::Status::OK();
          }

          n->AddAttr("_ngraph_concat_static_axis", tf_static_axis[0]);
          *result = true;
          return tf::Status::OK();
        };

        confirmation_functions["Conv2D"] = always;
        confirmation_functions["DepthwiseConv2dNative"] = always;
        confirmation_functions["Equal"] = always;
        confirmation_functions["Floor"] = always;
        confirmation_functions["FusedBatchNorm"] = always;
        confirmation_functions["Log"] = always;
        confirmation_functions["MatMul"] = always;
        confirmation_functions["MaxPool"] = always;

        // Constraints: "keep_dims" is not supported, reduction-axes input
        // must be Const.
        confirmation_functions["Mean"] = [](tf::Node* n, bool* result) {
          bool tf_keep_dims;

          if (tf::GetNodeAttr(n->attrs(), "keep_dims", &tf_keep_dims) !=
              tf::Status::OK()) {
            tf_keep_dims = false;
          }

          tf::Node* tf_axes_node;
          TF_RETURN_IF_ERROR(n->input_node(1, &tf_axes_node));

          std::vector<tf::int64> tf_static_axes;
          if (ExtractConstantData(tf_axes_node, &tf_static_axes) !=
              tf::Status::OK()) {
            *result = false;
            return tf::Status::OK();
          }

          n->AddAttr("_ngraph_mean_static_axes", tf_static_axes);
          *result = true;
          return tf::Status::OK();
        };

        confirmation_functions["Mul"] = always;

        // Constraint: padding-widths input must be Const.
        confirmation_functions["Pad"] = [](tf::Node* n, bool* result) {
          tf::Node* tf_paddings_node;
          TF_RETURN_IF_ERROR(n->input_node(1, &tf_paddings_node));

          std::vector<tf::int64> tf_static_paddings;
          if (ExtractConstantData(tf_paddings_node, &tf_static_paddings) !=
              tf::Status::OK()) {
            *result = false;
            return tf::Status::OK();
          }

          n->AddAttr("_ngraph_pad_static_paddings", tf_static_paddings);
          *result = true;
          return tf::Status::OK();
        };

        confirmation_functions["Relu"] = always;
        confirmation_functions["Relu6"] = always;

        // Constraint: shape input must be Const.
        confirmation_functions["Reshape"] = [](tf::Node* n, bool* result) {
          tf::Node* tf_shape_node;
          TF_RETURN_IF_ERROR(n->input_node(1, &tf_shape_node));

          std::vector<tf::int64> tf_static_shape;
          if (ExtractConstantData(tf_shape_node, &tf_static_shape) !=
              tf::Status::OK()) {
            *result = false;
            return tf::Status::OK();
          }

          n->AddAttr("_ngraph_reshape_static_shape", tf_static_shape);
          *result = true;
          return tf::Status::OK();
        };

        confirmation_functions["Sign"] = always;
        confirmation_functions["Snapshot"] = always;
        confirmation_functions["Squeeze"] = always;

        // Constraints: "keep_dims" is not supported, reduction-axes input
        // must be Const.
        confirmation_functions["Sum"] = [](tf::Node* n, bool* result) {
          // For now, the "keep_dims" option is not supported.
          bool tf_keep_dims;

          if (tf::GetNodeAttr(n->attrs(), "keep_dims", &tf_keep_dims) !=
              tf::Status::OK()) {
            tf_keep_dims = false;
          }

          tf::Node* tf_axes_node;
          TF_RETURN_IF_ERROR(n->input_node(1, &tf_axes_node));

          std::vector<tf::int64> tf_static_axes;
          if (ExtractConstantData(tf_axes_node, &tf_static_axes) !=
              tf::Status::OK()) {
            *result = false;
            return tf::Status::OK();
          }

          n->AddAttr("_ngraph_sum_static_axes", tf_static_axes);
          *result = true;
          return tf::Status::OK();
        };

        // Constraint: permutation input must be Const.
        confirmation_functions["Transpose"] = [](tf::Node* n, bool* result) {
          tf::Node* tf_permutation_node;
          TF_RETURN_IF_ERROR(n->input_node(1, &tf_permutation_node));

          std::vector<tf::int64> tf_static_permutation;
          if (ExtractConstantData(tf_permutation_node,
                                  &tf_static_permutation) != tf::Status::OK()) {
            *result = false;
            return tf::Status::OK();
          }

          n->AddAttr("_ngraph_transpose_static_permutation",
                     tf_static_permutation);
          *result = true;
          return tf::Status::OK();
        };
      }

      initialized = true;
    }

    for (auto node : graph->op_nodes()) {
      if (NGraphPlacementRequested(node)) {
        bool confirmed = false;

        auto it = confirmation_functions.find(node->type_string());

        if (it != confirmation_functions.end()) {
          TF_RETURN_IF_ERROR(it->second(node, &confirmed));
        }

        if (confirmed) {
          node->AddAttr("_kernel", "ngraph");
        } else {
          NGRAPH_VLOG(4) << "Rejecting: " << node->name() << "["
                         << node->type_string() << "]";
        }
      }
    }

    return tf::Status::OK();
  }
};
}  // namespace ngraph_bridge

namespace tensorflow {
REGISTER_OPTIMIZATION(OptimizationPassRegistry::PRE_PLACEMENT, 90,
                      ngraph_bridge::NGraphConfirmPass);
}  // namespace tensorflow
