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

cc_binary(
    name = 'libngraph_bridge.so',
    srcs = [
        "src/ngraph_api.cc",
        "src/ngraph_api.h",
        "src/ngraph_assign_clusters.cc",
        "src/ngraph_assign_clusters.h",
        "src/ngraph_builder.cc",
        "src/ngraph_builder.h",
        "src/ngraph_backend_manager.h",
        "src/ngraph_backend_manager.cc",
        "src/ngraph_capture_variables.cc",
        "src/ngraph_capture_variables.h",
        "src/ngraph_cluster_manager.cc",
        "src/ngraph_cluster_manager.h",
        "src/ngraph_conversions.h",
        "src/ngraph_deassign_clusters.cc",
        "src/ngraph_deassign_clusters.h",
        "src/ngraph_encapsulate_clusters.cc",
        "src/ngraph_encapsulate_clusters.h",
        "src/ngraph_encapsulate_op.cc",
        "src/ngraph_freshness_tracker.cc",
        "src/ngraph_freshness_tracker.h",
        "src/ngraph_mark_for_clustering.cc",
        "src/ngraph_mark_for_clustering.h",
        "src/ngraph_rewrite_for_tracking.cc",
        "src/ngraph_rewrite_for_tracking.h",
        "src/ngraph_rewrite_pass.cc",
        "src/ngraph_tracked_variable.cc",
        "src/ngraph_utils.cc",
        "src/ngraph_utils.h",
        "src/ngraph_version_utils.h",
        "src/tf_deadness_analysis.cc",
        "src/tf_deadness_analysis.h",
        "src/tf_graphcycles.cc",
        "src/tf_graphcycles.h",
        "src/version.h",
        "src/version.cc",
        "logging/ngraph_log.h",
        "logging/ngraph_log.cc",
        "logging/tf_graph_writer.h",
        "logging/tf_graph_writer.cc",
    ],
    linkshared = 1,
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
        "@ngraph//:ngraph_headers",
        "@ngraph//:ngraph_core",
    ],
    copts = [
        "-pthread", 
        "-std=c++11", 
        "-D_FORTIFY_SOURCE=2",
        "-Wformat",
        "-Wformat-security",
        "-Wformat",
        "-fstack-protector-strong",
        "-D NDEBUG",
        '-D SHARED_LIB_PREFIX=\\"lib\\"',
        '-D SHARED_LIB_SUFFIX=\\".so\\"',
        "-I logging",
        "-I external/ngraph/src",
    ] + CXX_ABI,
    linkopts = [
        "-Wl,-z,noexecstack",
        "-Wl,-z,relro",
        "-Wl,-z,now",
    ],
    visibility = ["//visibility:public"],
)
