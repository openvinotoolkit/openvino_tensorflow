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
//   ...       Const[2,4,2]
//     \       /
//      Reshape                     (1)
//
// but this is not:
//
//   ...       Placeholder
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
// "ngraph", along with some op-specific metadata (if applicable). The stub
// kernels, in turn, are registered with the constraint that _kernel="ngraph".
// This means that during the placement pass, our kernels will not be allowed
// for nodes we did not mark during this pass, and placement will fall back on
// CPU.
//
// Taking Reshape as an example, the pass ensures that the "shape" input is
// constant, and if so, it adds to the Reshape node the "_kernel=ngraph"
// attribute, along with some metadata recording the value of the constant.
// Thus graph (1) is transformed as follows:
//
//   ...       Const[2,4,2][_kernel="ngraph"]
//     \       /
//      Reshape[_kernel="ngraph",
//              _ngraph_reshape_static_shape={2,4,2}]
//
// while graph (2) would be left unchanged, meaning that soft placement will
// fall back on non-nGraph implementations.
//
// Internally, there are two pieces. The first is a type constraint checker,
// which supplants the type checking machinery usually used with
// REGISTER_KERNEL_BUILDER. This ensures that any constraints on the data types
// of input tensors are satisfied---for example, we do not support DT_STRING.
// The second part is a set of finer-grained per-op checks called "confirmation
// functions", implementing more specific checks like the one described for
// Reshape above.
//
// The confirmation functions are implemented as callbacks of the type:
//
//      std::function<tf::Status(tf::Node*, bool*)>.
//
// A confirmation function returns true/false by reference through its second
// parameter: true if placement is "accepted", and false if it is "rejected".
// For example, the confirmation function for "Reshape" will return true
// for (1) above, and false for (2).
//
// A confirmation function can also, as a side effect, add attributes to the
// node being checked, which can be used later in ngraph_builder. (Note that in
// general such attributes will need to start with "_" to mark them as
// "internal" or "system" attributes, as otherwise TensorFlow attempts to
// validate them as against the op schema.)
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
            ValuesFromConstNode<tf::int64>(node->def(), &shape_proto, values));
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
    //
    // A map of op types (e.g. "Add") to type constraint maps. For (fake)
    // example:
    //
    //  type_constraint_map["Cast"]["SrcT"] = {tf::DT_FLOAT, tf::DT_BOOL};
    //  type_constraint_map["Cast"]["DstT"] = {tf::DT_DOUBLE, tf::DT_INT16};
    //
    // ...would mean that for the "Cast" op, the "SrcT" type variable can be
    // DT_FLOAT or DT_BOOL, and the "DstT" type variable can be DT_DOUBLE or
    // DT_INT16.
    //
    static std::map<std::string,
                    std::map<std::string, tf::gtl::ArraySlice<tf::DataType>>>
        type_constraint_map;

    //
    // A map of op types (e.g. "Add") to confirmation functions. These can be
    // used to check arbitrary constraints, and attach information to the node
    // in the process. For example:
    //
    //    confirmation_functions["MyOp"] = [](tf::Node* n, bool* result) {
    //      tf::Node* tf_arg_node;
    //      TF_RETURN_IF_ERROR(n->input_node(0, &tf_arg_node));
    //
    //      std::vector<tf::int64> tf_const_data;
    //      if (ExtractConstantData(tf_arg_node, &tf_const_data) !=
    //              tf::Status::OK() ||
    //          tf_const_data.size() != 1) {
    //        *result = false;
    //        return tf::Status::OK();
    //      }
    //
    //      n->AddAttr("_ngraph_myop_constant_input", tf_const_data[0]);
    //      *result = true;
    //      return tf::Status::OK();
    //    };
    //
    // The foregoing function checks every "MyOp" node to make sure that its
    // zeroth input node is a constant scalar, and if it is, extracts the value
    // of that scalar, and attaches it to the node as the
    // "_ngraph_myop_constant_input" attribute. Placement fails if the input is
    // not a constant scalar (since "false" is written to *result).
    //
    static std::map<std::string, ConfirmationFunction> confirmation_functions;

    tf::mutex init_mu;
    static bool initialized = false;

    // If the type constraint and confirmation function maps have not been
    // initialized, initialize them.
    //
    // IF YOU ARE ADDING A NEW OP IMPLEMENTATION, ADD TYPE CONSTRAINTS AND A
    // CONFIRMATION FUNCTION FOR THE OP HERE. The constraint function should
    // refuse placement if the node is not supported in the builder, and tag
    // the node with any data that will be needed in case the graph is broken
    // up in a later rewrite pass (for example, constant data).
    {
      tf::mutex_lock l(init_mu);

      if (!initialized) {
        //
        // Initialize type constraint map.
        //
        type_constraint_map["Abs"]["T"] = NGraphNumericDTypes();
        type_constraint_map["Add"]["T"] = NGraphNumericDTypes();
        type_constraint_map["AvgPool"]["T"] = NGraphNumericDTypes();
        type_constraint_map["AvgPoolGrad"]["T"] = NGraphNumericDTypes();
        type_constraint_map["BatchMatMul"]["T"] = NGraphNumericDTypes();
        type_constraint_map["BiasAdd"]["T"] = NGraphNumericDTypes();
        type_constraint_map["Cast"]["SrcT"] = NGraphDTypes();
        type_constraint_map["Cast"]["DstT"] = NGraphDTypes();
        type_constraint_map["ConcatV2"]["T"] = NGraphDTypes();
        type_constraint_map["ConcatV2"]["Tidx"] = NGraphIndexDTypes();
        type_constraint_map["Conv2D"]["T"] = NGraphNumericDTypes();
        type_constraint_map["Conv2DBackpropFilter"]["T"] =
            NGraphNumericDTypes();
        type_constraint_map["Conv2DBackpropInput"]["T"] = NGraphNumericDTypes();
        type_constraint_map["DepthwiseConv2dNative"]["T"] =
            NGraphNumericDTypes();
        type_constraint_map["Equal"]["T"] = NGraphDTypes();
        type_constraint_map["Exp"]["T"] = NGraphNumericDTypes();
        type_constraint_map["ExpandDims"]["T"] = NGraphDTypes();
        type_constraint_map["Floor"]["T"] = NGraphNumericDTypes();
        type_constraint_map["FusedBatchNorm"]["T"] = NGraphNumericDTypes();
        type_constraint_map["FusedBatchNormGrad"]["T"] = NGraphNumericDTypes();
        type_constraint_map["Greater"]["T"] = NGraphDTypes();
        type_constraint_map["GreaterEqual"]["T"] = NGraphDTypes();
        type_constraint_map["Less"]["T"] = NGraphDTypes();
        type_constraint_map["LessEqual"]["T"] = NGraphDTypes();
        type_constraint_map["Log"]["T"] = NGraphNumericDTypes();
        // LogicalAnd has no type attributes, ("T", if it existed, would always
        // be bool).
        type_constraint_map["MatMul"]["T"] = NGraphNumericDTypes();
        type_constraint_map["Maximum"]["T"] = NGraphNumericDTypes();
        type_constraint_map["MaxPool"]["T"] = NGraphNumericDTypes();
        type_constraint_map["Mean"]["T"] = NGraphNumericDTypes();
        type_constraint_map["Mean"]["Tidx"] = NGraphIndexDTypes();
        type_constraint_map["Minimum"]["T"] = NGraphNumericDTypes();
        type_constraint_map["Mul"]["T"] = NGraphNumericDTypes();
        type_constraint_map["Pack"]["T"] = NGraphDTypes();
        type_constraint_map["Pad"]["T"] = NGraphDTypes();
        type_constraint_map["Pad"]["Tpaddings"] = NGraphIndexDTypes();
        type_constraint_map["Pow"]["T"] = NGraphNumericDTypes();
        type_constraint_map["Prod"]["T"] = NGraphNumericDTypes();
        type_constraint_map["Prod"]["Tidx"] = NGraphIndexDTypes();
        type_constraint_map["RealDiv"]["T"] = NGraphNumericDTypes();
        type_constraint_map["Reciprocal"]["T"] = NGraphNumericDTypes();
        type_constraint_map["Relu"]["T"] = NGraphNumericDTypes();
        type_constraint_map["Relu6"]["T"] = NGraphNumericDTypes();
        type_constraint_map["ReluGrad"]["T"] = NGraphNumericDTypes();
        type_constraint_map["Reshape"]["T"] = NGraphDTypes();
        type_constraint_map["Reshape"]["Tshape"] = NGraphIndexDTypes();
        type_constraint_map["Rsqrt"]["T"] = NGraphDTypes();
        type_constraint_map["Slice"]["T"] = NGraphDTypes();
        type_constraint_map["Slice"]["Index"] = NGraphIndexDTypes();
        type_constraint_map["Sign"]["T"] = NGraphNumericDTypes();
        type_constraint_map["Sigmoid"]["T"] = NGraphNumericDTypes();
        type_constraint_map["Snapshot"]["T"] = NGraphDTypes();
        type_constraint_map["Softmax"]["T"] = NGraphNumericDTypes();
        type_constraint_map["Split"]["T"] = NGraphDTypes();
        type_constraint_map["SplitV"]["T"] = NGraphDTypes();
        type_constraint_map["SplitV"]["Tlen"] = NGraphIndexDTypes();
        type_constraint_map["Square"]["T"] = NGraphDTypes();
        type_constraint_map["SquaredDifference"]["T"] = NGraphDTypes();
        type_constraint_map["Squeeze"]["T"] = NGraphDTypes();
        type_constraint_map["StridedSlice"]["T"] = NGraphDTypes();
        type_constraint_map["StridedSlice"]["Index"] = NGraphIndexDTypes();
        type_constraint_map["Sub"]["T"] = NGraphNumericDTypes();
        type_constraint_map["Sum"]["T"] = NGraphNumericDTypes();
        type_constraint_map["Sum"]["Tidx"] = NGraphIndexDTypes();
        type_constraint_map["Tanh"]["T"] = NGraphNumericDTypes();
        type_constraint_map["Tile"]["T"] = NGraphNumericDTypes();
        type_constraint_map["Tile"]["Tmultiples"] = NGraphIndexDTypes();
        type_constraint_map["Transpose"]["T"] = NGraphDTypes();
        type_constraint_map["Transpose"]["Tperm"] = NGraphIndexDTypes();

        //
        // Initialize confirmation function map.
        //

        // Trivial confirmation function which always accepts placement.
        ConfirmationFunction always = [](tf::Node* n, bool* result) {
          *result = true;
          return tf::Status::OK();
        };

        //
        // Please keep these in alphabetical order by op name.
        //
        confirmation_functions["Abs"] = always;
        confirmation_functions["Add"] = always;
        confirmation_functions["AvgPool"] = always;
        confirmation_functions["AvgPoolGrad"] = [](tf::Node* n, bool* result) {
          tf::Node* tf_orig_input_shape;
          TF_RETURN_IF_ERROR(n->input_node(0, &tf_orig_input_shape));

          std::vector<tf::int64> tf_orig_input_shape_vec;
          if (ExtractConstantData(tf_orig_input_shape, &tf_orig_input_shape_vec) !=
                  tf::Status::OK() ||
              tf_orig_input_shape_vec.size() != 4) {
            *result = false;
            return tf::Status::OK();
          }

          n->AddAttr("_ngraph_avgpoolgrad_static_input_shape", tf_orig_input_shape_vec);
          *result = true;
          return tf::Status::OK();
        };
        confirmation_functions["BiasAdd"] = always;
        confirmation_functions["BatchMatMul"] = always;
        confirmation_functions["Cast"] = always;

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
        confirmation_functions["Conv2DBackpropFilter"] = [](tf::Node* n,
                                                            bool* result) {
          tf::Node* tf_filter_sizes;
          TF_RETURN_IF_ERROR(n->input_node(1, &tf_filter_sizes));

          std::vector<tf::int64> tf_static_filter_sizes(4);
          if (ExtractConstantData(tf_filter_sizes, &tf_static_filter_sizes) !=
                  tf::Status::OK() ||
              tf_static_filter_sizes.size() != 4) {
            *result = false;
            return tf::Status::OK();
          }

          n->AddAttr("_ngraph_static_filter_sizes", tf_static_filter_sizes);
          *result = true;
          return tf::Status::OK();
        };
        confirmation_functions["Conv2DBackpropInput"] = [](tf::Node* n,
                                                           bool* result) {
          tf::Node* tf_input_sizes;
          TF_RETURN_IF_ERROR(n->input_node(0, &tf_input_sizes));

          std::vector<tf::int64> tf_static_input_sizes(4);
          if (ExtractConstantData(tf_input_sizes, &tf_static_input_sizes) !=
                  tf::Status::OK() ||
              tf_static_input_sizes.size() != 4) {
            *result = false;
            return tf::Status::OK();
          }

          n->AddAttr("_ngraph_static_input_sizes", tf_static_input_sizes);
          *result = true;
          return tf::Status::OK();
        };

        confirmation_functions["DepthwiseConv2dNative"] = always;
        confirmation_functions["Equal"] = always;
        confirmation_functions["Exp"] = always;
        confirmation_functions["ExpandDims"] = [](tf::Node* n, bool* result) {
          tf::Node* tf_dim_node;
          TF_RETURN_IF_ERROR(n->input_node(1, &tf_dim_node));

          std::vector<tf::int64> tf_static_dim;
          if (ExtractConstantData(tf_dim_node, &tf_static_dim) !=
              tf::Status::OK()) {
            *result = false;
            return tf::Status::OK();
          }

          n->AddAttr("_ngraph_expanddims_static_dim", tf_static_dim);

          *result = true;
          return tf::Status::OK();
        };

        confirmation_functions["Fill"] = [](tf::Node* n, bool* result) {
          tf::Node* tf_dims_node;
          TF_RETURN_IF_ERROR(n->input_node(0, &tf_dims_node));

          std::vector<tf::int64> tf_dims;
          if (ExtractConstantData(tf_dims_node, &tf_dims) != tf::Status::OK()) {
            *result = false;
            return tf::Status::OK();
          }

          n->AddAttr("_ngraph_fill_static_dims", tf_dims);
          *result = true;
          return tf::Status::OK();
        };

        confirmation_functions["Floor"] = always;
        confirmation_functions["FusedBatchNorm"] = always;
        confirmation_functions["FusedBatchNormGrad"] = always;
        confirmation_functions["Greater"] = always;
        confirmation_functions["GreaterEqual"] = always;
        confirmation_functions["Less"] = always;
        confirmation_functions["LessEqual"] = always;
        confirmation_functions["Log"] = always;
        confirmation_functions["LogicalAnd"] = always;
        confirmation_functions["MatMul"] = always;
        confirmation_functions["Maximum"] = always;
        confirmation_functions["MaxPool"] = always;

        // Constraints: "keep_dims" is not supported, reduction-axes input
        // must be Const.
        confirmation_functions["Mean"] = [](tf::Node* n, bool* result) {
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

        confirmation_functions["Minimum"] = always;
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

        confirmation_functions["Pow"] = always;

        // Constraints: "keep_dims" is not supported, reduction-axes input
        // must be Const.
        confirmation_functions["Prod"] = [](tf::Node* n, bool* result) {
          bool tf_keep_dims;

          if (tf::GetNodeAttr(n->attrs(), "keep_dims", &tf_keep_dims) ==
              tf::Status::OK()) {
            if (tf_keep_dims) {
              *result = false;
              return tf::Status::OK();
            }
          }

          tf::Node* tf_axes_node;
          TF_RETURN_IF_ERROR(n->input_node(1, &tf_axes_node));

          std::vector<tf::int64> tf_static_axes;
          if (ExtractConstantData(tf_axes_node, &tf_static_axes) !=
              tf::Status::OK()) {
            *result = false;
            return tf::Status::OK();
          }

          n->AddAttr("_ngraph_prod_static_axes", tf_static_axes);
          *result = true;
          return tf::Status::OK();
        };

        confirmation_functions["RealDiv"] = always;
        confirmation_functions["Reciprocal"] = always;
        confirmation_functions["Relu"] = always;
        confirmation_functions["Relu6"] = always;
        confirmation_functions["ReluGrad"] = always;
        confirmation_functions["Rsqrt"] = always;

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

        confirmation_functions["Sigmoid"] = always;
        confirmation_functions["Sign"] = always;

        // Constraint: begin and size input must be Const.
        confirmation_functions["Slice"] = [](tf::Node* n, bool* result) {
          tf::Node* tf_begin_node;
          tf::Node* tf_size_node;

          TF_RETURN_IF_ERROR(n->input_node(1, &tf_begin_node));
          TF_RETURN_IF_ERROR(n->input_node(2, &tf_size_node));

          std::vector<tf::int64> tf_static_begin;
          if (ExtractConstantData(tf_begin_node, &tf_static_begin) !=
              tf::Status::OK()) {
            *result = false;
            return tf::Status::OK();
          }
          std::vector<tf::int64> tf_static_size;
          if (ExtractConstantData(tf_size_node, &tf_static_size) !=
              tf::Status::OK()) {
            *result = false;
            return tf::Status::OK();
          }

          n->AddAttr("_ngraph_slice_static_begin", tf_static_begin);
          n->AddAttr("_ngraph_slice_static_size", tf_static_size);

          *result = true;
          return tf::Status::OK();
        };

        confirmation_functions["Snapshot"] = always;
        confirmation_functions["Softmax"] = always;
        confirmation_functions["Split"] = [](tf::Node* n, bool* result) {
          tf::Node* tf_split_dim_node;
          TF_RETURN_IF_ERROR(n->input_node(0, &tf_split_dim_node));

          std::vector<tf::int64> tf_split_dim;
          if (ExtractConstantData(tf_split_dim_node, &tf_split_dim) !=
              tf::Status::OK()) {
            *result = false;
            return tf::Status::OK();
          }

          n->AddAttr("_ngraph_split_static_dim", tf_split_dim[0]);
          *result = true;
          return tf::Status::OK();
        };
        confirmation_functions["SplitV"] = always;
        confirmation_functions["Square"] = always;
        confirmation_functions["SquaredDifference"] = always;
        confirmation_functions["Squeeze"] = always;

        // Constraint: begin, end, and stride inputs must be Const
        confirmation_functions["StridedSlice"] = [](tf::Node* n, bool* result) {
          // reject if tf.newaxis in strided slice
          // TODO support tf.newaxis
          int tf_new_axis_mask;
          TF_RETURN_IF_ERROR(
              tf::GetNodeAttr(n->attrs(), "new_axis_mask", &tf_new_axis_mask));
          if (tf_new_axis_mask != 0) {
            *result = false;
            return tf::Status::OK();
          }
          tf::Node* tf_begin_node;
          tf::Node* tf_end_node;
          tf::Node* tf_stride_node;

          TF_RETURN_IF_ERROR(n->input_node(1, &tf_begin_node));
          TF_RETURN_IF_ERROR(n->input_node(2, &tf_end_node));
          TF_RETURN_IF_ERROR(n->input_node(3, &tf_stride_node));

          std::vector<tf::int64> tf_static_begin;
          if (ExtractConstantData(tf_begin_node, &tf_static_begin) !=
              tf::Status::OK()) {
            *result = false;
            return tf::Status::OK();
          }
          std::vector<tf::int64> tf_static_end;
          if (ExtractConstantData(tf_end_node, &tf_static_end) !=
              tf::Status::OK()) {
            *result = false;
            return tf::Status::OK();
          }
          std::vector<tf::int64> tf_static_stride;
          if (ExtractConstantData(tf_stride_node, &tf_static_stride) !=
              tf::Status::OK()) {
            *result = false;
            return tf::Status::OK();
          }

          n->AddAttr("_ngraph_stridedslice_static_begin", tf_static_begin);
          n->AddAttr("_ngraph_stridedslice_static_end", tf_static_end);
          n->AddAttr("_ngraph_stridedslice_static_stride", tf_static_stride);

          *result = true;
          return tf::Status::OK();
        };

        confirmation_functions["Pack"] = always;
        confirmation_functions["Sub"] = always;

        // Constraints: reduction-axes input must be Const.
        confirmation_functions["Sum"] = [](tf::Node* n, bool* result) {
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

        confirmation_functions["Tanh"] = always;
        confirmation_functions["Tile"] = [](tf::Node* n, bool* result) {
          tf::Node* tf_multiples;
          TF_RETURN_IF_ERROR(n->input_node(1, &tf_multiples));

          std::vector<tf::int64> tf_static_multiples;
          if (ExtractConstantData(tf_multiples, &tf_static_multiples) !=
              tf::Status::OK()) {
            *result = false;
            return tf::Status::OK();
          }

          n->AddAttr("_ngraph_tile_static_multiples", tf_static_multiples);
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

        initialized = true;
      }
    }

    for (auto node : graph->op_nodes()) {
      if (NGraphPlacementRequested(node)) {
        bool type_constraints_ok = true;

        // First check type constraints.
        for (auto& name_and_set : type_constraint_map[node->type_string()]) {
          auto& type_attr_name = name_and_set.first;
          auto& allowed_types = name_and_set.second;

          tf::DataType dt;

          if (tf::GetNodeAttr(node->attrs(), type_attr_name, &dt) !=
                  tf::Status::OK() ||
              std::find(allowed_types.begin(), allowed_types.end(), dt) ==
                  allowed_types.end()) {
            type_constraints_ok = false;
            break;
          }
        }

        // If type constraints are satisfied, check for a confirmation
        // function.

        bool confirmed = false;
        if (type_constraints_ok) {
          auto it = confirmation_functions.find(node->type_string());

          if (it != confirmation_functions.end()) {
            TF_RETURN_IF_ERROR(it->second(node, &confirmed));
          }
        }
        // Set the _kernel attribute if type constraints are satisfied and the
        // confirmation function (if any) has returned true.

        if (confirmed) {
          NGRAPH_VLOG(4) << "Accepting: " << node->name() << "["
                         << node->type_string() << "]";
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
