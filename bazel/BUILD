# ==============================================================================
#  Copyright 2019 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ==============================================================================
load("//:cxx_abi_option.bzl", "CXX_ABI")
load(
    "@org_tensorflow//tensorflow:tensorflow.bzl", 
    "tf_cc_binary")

cc_library(
    name = "ngraph_bridge_headers",
    hdrs = glob([
        "ngraph_bridge/ngraph_api.h",
        "ngraph_bridge/ngraph_assign_clusters.h",
        "ngraph_bridge/ngraph_builder.h",
        "ngraph_bridge/ngraph_backend_config.h",
        "ngraph_bridge/ngraph_backend_manager.h",
        "ngraph_bridge/ngraph_capture_variables.h",
        "ngraph_bridge/ngraph_cluster_manager.h",
        "ngraph_bridge/ngraph_conversions.h",
        "ngraph_bridge/ngraph_deassign_clusters.h",
        "ngraph_bridge/ngraph_encapsulate_clusters.h",
        "ngraph_bridge/ngraph_encapsulate_impl.h",
        "ngraph_bridge/ngraph_executor.h",
        "ngraph_bridge/ngraph_encapsulate_op.h",
        "ngraph_bridge/ngraph_freshness_tracker.h",
        "ngraph_bridge/ngraph_mark_for_clustering.h",
        "ngraph_bridge/ngraph_partial_shapes.h",
        "ngraph_bridge/ngraph_pipelined_tensors.h",
        "ngraph_bridge/ngraph_rewrite_for_tracking.h",
        "ngraph_bridge/ngraph_timer.h",
        "ngraph_bridge/ngraph_utils.h",
        "ngraph_bridge/ngraph_version_utils.h",
        "ngraph_bridge/tf_deadness_analysis.h",
        "ngraph_bridge/tf_graphcycles.h",
        "ngraph_bridge/thread_safe_queue.h",
        "ngraph_bridge/version.h",
        "ngraph_bridge/grappler/ngraph_optimizer.h",
        "ngraph_bridge/grappler/ngraph_add_identityn.h",
        "logging/ngraph_log.h",
        "logging/tf_graph_writer.h",
    ]),
    copts = [
        "-I external/ngraph/src",
        "-I logging",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = 'ngraph_bridge_lib',
    srcs = [
        "ngraph_bridge/ngraph_api.cc",
        "ngraph_bridge/ngraph_assign_clusters.cc",
        "ngraph_bridge/ngraph_builder.cc",
        "ngraph_bridge/ngraph_backend_manager.cc",
        "ngraph_bridge/ngraph_capture_variables.cc",
        "ngraph_bridge/ngraph_cluster_manager.cc",
        "ngraph_bridge/ngraph_conversions.cc",
        "ngraph_bridge/ngraph_deassign_clusters.cc",
        "ngraph_bridge/ngraph_encapsulate_clusters.cc",
        "ngraph_bridge/ngraph_encapsulate_impl.cc",
        "ngraph_bridge/ngraph_executor.cc",
        "ngraph_bridge/ngraph_encapsulate_op.cc",
        "ngraph_bridge/ngraph_freshness_tracker.cc",
        "ngraph_bridge/ngraph_mark_for_clustering.cc",
        "ngraph_bridge/ngraph_partial_shapes.cc",
        "ngraph_bridge/ngraph_pipelined_tensors.cc",
        "ngraph_bridge/ngraph_rewrite_for_tracking.cc",
        "ngraph_bridge/ngraph_tracked_variable.cc",
        "ngraph_bridge/ngraph_utils.cc",
        "ngraph_bridge/tf_deadness_analysis.cc",
        "ngraph_bridge/tf_graphcycles.cc",
        "ngraph_bridge/ops/ngraph_ops.cc",
        "ngraph_bridge/version.cc",
        "ngraph_bridge/grappler/ngraph_optimizer.cc",
        "ngraph_bridge/grappler/ngraph_add_identityn.cc",
        "logging/ngraph_log.cc",
        "logging/tf_graph_writer.cc",
    ],
    deps = [
        ":ngraph_bridge_headers",
        "@org_tensorflow//tensorflow/core:core_cpu_headers_lib",
        "@org_tensorflow//tensorflow/core/grappler/clusters:cluster",
        "@org_tensorflow//tensorflow/core/grappler/optimizers:custom_graph_optimizer_registry",
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@ngraph//:ngraph_headers",
        "@ngraph//:ngraph_core",
        "@ngraph//:ngraph_version",
    ],
    copts = [
        "-pthread", 
        "-std=c++11", 
        "-D_FORTIFY_SOURCE=2",
        "-Wall",
        "-Wextra",
        "-Wno-comment",
        "-Wno-sign-compare",
        "-Wformat",
        "-Wformat-security",
        "-fstack-protector-all",
        '-D NGRAPH_BRIDGE_STATIC_LIB_ENABLE',
        '-D NGRAPH_TF_USE_GRAPPLER_OPTIMIZER',
        "-I ngraph_bridge",
        "-I logging",
        "-I external/ngraph/src",
    ] + CXX_ABI,
    linkstatic = True,
    linkopts = [
        "-Wl,-z,noexecstack",
        "-Wl,-z,relro",
        "-Wl,-z,now",
    ],
    visibility = ["//visibility:public"],
    alwayslink = 1,
)

tf_cc_binary(
    name = 'hello_tf',
    srcs = [
        "examples/cpp/hello_tf.cpp",
    ],
    deps = [
        "@org_tensorflow//tensorflow/cc:client_session",
        "@org_tensorflow//tensorflow/cc:ops",
        "@org_tensorflow//tensorflow/core:tensorflow",
        ":ngraph_bridge_lib",
        "@ngraph//:cpu_backend",
    ],
    copts = [
        "-pthread",
        "-std=c++11",
        "-D_FORTIFY_SOURCE=2",
        "-Wformat",
        "-Wformat-security",
        "-fstack-protector-all",
        "-I ngraph_bridge",
        "-I external/ngraph/src",
        "-I logging",
    ] + CXX_ABI,
    linkstatic=True,
    visibility = ["//visibility:public"],
)

tf_cc_binary(
    name = 'infer_multiple_networks',
    srcs = [
        "examples/cpp/inference/infer_multiple_networks.cc",
        "examples/cpp/inference/inference_engine.h",
        "examples/cpp/inference/inference_engine.cc",
        "examples/cpp/inference/tf_label_image_utils.cc",
    ],
    deps = [
        "@org_tensorflow//tensorflow/cc:cc_ops",
        "@org_tensorflow//tensorflow/cc:client_session",
        "@org_tensorflow//tensorflow/core:tensorflow",
        ":ngraph_bridge_lib",
        "@ngraph//:cpu_backend",
    ],
    copts = [
        "-pthread",
        "-std=c++11",
        "-D_FORTIFY_SOURCE=2",
        "-Wformat",
        "-Wformat-security",
        "-Wformat",
        "-fstack-protector-all",
        "-I ngraph_bridge",
        "-I external/ngraph/src",
        "-I logging",
    ] + CXX_ABI,
    linkstatic=False,
    visibility = ["//visibility:public"],
)
