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
        "ngraph_bridge/version.h",
        "ngraph_bridge/grappler/ngraph_optimizer.h",
        "ngraph_bridge/grappler/ngraph_add_identityn.h",
        "logging/ngraph_log.h",
        "logging/tf_graph_writer.h",
    ]),
    copts = [
        "-I external/ngraph_bridge/src",
        "-I external/ngraph_bridge/logging",
    ],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = 'libngraph_bridge.so',
    srcs = [
        "ngraph_bridge/ngraph_api.cc",
        "ngraph_bridge/ngraph_assign_clusters.cc",
        "ngraph_bridge/ngraph_assign_clusters.h",
        "ngraph_bridge/ngraph_builder.cc",
        "ngraph_bridge/ngraph_backend_manager.cc",
        "ngraph_bridge/ngraph_capture_variables.cc",
        "ngraph_bridge/ngraph_cluster_manager.cc",
        "ngraph_bridge/ngraph_deassign_clusters.cc",
        "ngraph_bridge/ngraph_encapsulate_clusters.cc",
        "ngraph_bridge/ngraph_encapsulate_impl.cc",
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
        "ngraph_bridge/version.cc",
        "ngraph_bridge/grappler/ngraph_optimizer.cc",
        "ngraph_bridge/grappler/ngraph_add_identityn.cc",
        "logging/ngraph_log.cc",
        "logging/tf_graph_writer.cc",
    ],
    linkshared = 1,
    deps = [
        ":ngraph_bridge_headers",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
        "@ngraph//:ngraph_headers",
        "@ngraph//:ngraph_core",
        "@ngraph//:ngraph_version",
    ],
    copts = [
        "-pthread", 
        "-std=c++11", 
        "-D_FORTIFY_SOURCE=2",
        "-Wformat",
        "-Wformat-security",
        "-Wformat",
        "-fstack-protector-all",
        "-D NDEBUG",
        '-D SHARED_LIB_PREFIX=\\"lib\\"',
        '-D SHARED_LIB_SUFFIX=\\".so\\"',
        '-D NGRAPH_TF_USE_GRAPPLER_OPTIMIZER',
        "-I external/ngraph/src",
    ] + CXX_ABI,
    linkopts = [
        "-Wl,-z,noexecstack",
        "-Wl,-z,relro",
        "-Wl,-z,now",
    ],
    visibility = ["//visibility:public"],
)
