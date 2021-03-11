/*******************************************************************************
 * Copyright (C) 2021 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/

#include "tensorflow/core/common_runtime/optimization_registry.h"

namespace tensorflow {
namespace openvino_tensorflow {

REGISTER_OP("_nGraphEncapsulate")
    .Input("args: Targuments")
    .Attr("Targuments: list(type) >= 0")
    .Output("results: Tresults")
    .Attr("Tresults: list(type) >= 0")
    .Attr("ovtf_cluster: int")
    .Attr("ngraph_graph_id: int")
    .SetIsStateful()
    .Doc("nGraph Encapsulation Op. For use by the nGraph JIT only.");

}  // namespace openvino_tensorflow
}  // namespace tensorflow