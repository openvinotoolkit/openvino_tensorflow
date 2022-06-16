/*******************************************************************************
 * Copyright (C) 2021-2022 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/

#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

#ifdef _WIN32
#define EXPAND(x) x
#define TF_NEW_ID_FOR_INIT_2(m, c, ...) \
  EXPAND(m(c, __VA_ARGS__))  // L145 selective_registration.h
#endif

namespace tensorflow {
namespace openvino_tensorflow {

REGISTER_OP("_OpenVINOEncapsulate")
    .Input("args: Targuments")
    .Attr("Targuments: list(type) >= 0")
    .Output("results: Tresults")
    .Attr("Tresults: list(type) >= 0")
    .Attr("ovtf_cluster: int")
    .Attr("openvino_graph_id: int")
    .Attr("cluster_cost: int")
    .SetIsStateful()
    .Doc("OpenVINO Encapsulation Op. For use by the OpenVINO JIT only.");

}  // namespace openvino_tensorflow
}  // namespace tensorflow