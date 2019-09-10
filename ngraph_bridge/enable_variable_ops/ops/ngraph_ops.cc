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

#include "tensorflow/core/common_runtime/optimization_registry.h"

namespace tensorflow {

namespace ngraph_bridge {

// ------------------------------------------------------------------
REGISTER_OP("NGraphApplyGradientDescent")
    .Input("var: Ref(T)")
    .Input("alpha: T")
    .Input("delta: T")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .Attr("just_looking: bool = false")
    .Attr("is_tf_just_looking: bool = false")
    .Attr("copy_to_tf: bool = false")
    .Attr("ngraph_graph_id: int");

// ------------------------------------------------------------------
REGISTER_OP("NGraphAssign")
    .Input("ref: Ref(T)")
    .Input("value: T")
    .Output("output_ref: Ref(T)")
    .Attr("T: type")
    .Attr("validate_shape: bool = true")
    .Attr("use_locking: bool = true")
    .Attr("just_looking: bool = false")
    .Attr("is_tf_just_looking: bool = false")
    .Attr("copy_to_tf: bool = false")
    .Attr("ngraph_graph_id: int");

// ------------------------------------------------------------------
REGISTER_OP("NGraphAssignAdd")
    .Input("ref: Ref(T)")
    .Input("value: T")
    .Output("output_ref: Ref(T)")
    .Attr("T: type")
    .Attr("validate_shape: bool = true")
    .Attr("use_locking: bool = true")
    .Attr("just_looking: bool = false")
    .Attr("is_tf_just_looking: bool = false")
    .Attr("copy_to_tf: bool = false")
    .Attr("ngraph_graph_id: int");

// ------------------------------------------------------------------
REGISTER_OP("NGraphAssignSub")
    .Input("ref: Ref(T)")
    .Input("value: T")
    .Output("output_ref: Ref(T)")
    .Attr("T: type")
    .Attr("validate_shape: bool = true")
    .Attr("use_locking: bool = true")
    .Attr("just_looking: bool = false")
    .Attr("is_tf_just_looking: bool = false")
    .Attr("copy_to_tf: bool = false")
    .Attr("ngraph_graph_id: int");

// ------------------------------------------------------------------
REGISTER_OP("NGraphEncapsulate")
    .Input("args: Targuments")
    .Attr("Targuments: list(type) >= 0")
    .Output("results: Tresults")
    .Attr("Tresults: list(type) >= 0")
    .Attr("ngraph_cluster: int")
    .Attr("ngraph_graph_id: int")
    .Attr("ngraph_backend: string")
    .Attr("ngraph_device_id: string")
    .SetIsStateful()
    .Doc("nGraph Encapsulation Op. For use by the nGraph JIT only.");

// ------------------------------------------------------------------
REGISTER_OP("NGraphVariable")
    .Output("ref: Ref(dtype)")
    .Attr("shape: shape")
    .Attr("dtype: type")
    .Attr("just_looking: bool = false")
    .Attr("is_tf_just_looking: bool = false")
    .Attr("copy_to_tf: bool = false")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("ngraph_graph_id: int")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ExplicitShape);

}  // namespace ngraph_bridge
}  // namespace tensorflow
