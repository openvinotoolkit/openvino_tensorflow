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

licenses(["notice"])

exports_files(["LICENSE"])


load(
    "@//tf_configure:tf_configure.bzl",
    "template_rule",
)

config_setting(
    name = "clang_linux_x86_64",
    values = {
        "cpu": "k8",
        "define": "using_clang=true",
    },
)

# Create the file mkldnn_version.h with MKL-DNN version numbers.
# Currently, the version numbers are hard coded here. If MKL-DNN is upgraded then
# the version numbers have to be updated manually. The version numbers can be
# obtained from the PROJECT_VERSION settings in CMakeLists.txt. The variable is
# set to "version_major.version_minor.version_patch". The git hash version can
# be set to NA.
# TODO(agramesh1) Automatically get the version numbers from CMakeLists.txt.

template_rule(
    name = "mkldnn_version_h",
    src = "include/mkldnn_version.h.in",
    out = "include/mkldnn_version.h",
    substitutions = {
        "@MKLDNN_VERSION_MAJOR@": "0",
        "@MKLDNN_VERSION_MINOR@": "21",
        "@MKLDNN_VERSION_PATCH@": "2",
        "@MKLDNN_VERSION_HASH@": "N/A",
    },
)

cc_library(
    name = "mkl_dnn",
    srcs = glob([
        "src/common/*.cpp",
        "src/common/*.hpp",
        "src/cpu/*.cpp",
        "src/cpu/*.hpp",
        "src/cpu/gemm/*.cpp",
        "src/cpu/gemm/*.hpp",
        "src/cpu/gemm/bf16/*.hpp",
        "src/cpu/gemm/bf16/*.cpp",
        "src/cpu/gemm/f32/*.cpp",
        "src/cpu/gemm/f32/*.hpp",
        "src/cpu/gemm/s8x8s32/*.cpp",
        "src/cpu/gemm/s8x8s32/*.hpp",
        "src/cpu/rnn/*.cpp",
        "src/cpu/rnn/*.hpp",
        "src/cpu/xbyak/*.h",
    ]) + [":mkldnn_version_h"],
    hdrs = glob(["include/*"]),
    copts = [
        "-fexceptions",
        "-fstack-protector-all",
        "-march=native",
        "-mtune=native",
        "-Wall",
        "-Wno-unknown-pragmas",
        "-fvisibility=internal",
        "-Wformat",
        "-Wformat-security",
        "-Wmissing-field-initializers",
        "-Wno-strict-overflow",
        "-std=c++11",
        "-D_FORTIFY_SOURCE=2",
        "-fopenmp",
        "-DUSE_MKL",
        "-DUSE_CBLAS",
        "-UUSE_MKL",
        "-UUSE_CBLAS",
        "-DMKLDNN_ENABLE_CONCURRENT_EXEC",
        "-DMKLDNN_THR=MKLDNN_THR_OMP",
        "-DMKLDNN_DLL",
        "-DMKLDNN_DLL_EXPORTS",
        "-O3",
    #] + select({
    #   "@org_tensorflow//tensorflow:linux_x86_64": [
            "-fopenmp",  # only works with gcc
    #    ],
        # TODO(ibiryukov): enable openmp with clang by including libomp as a
        # dependency.
    #    ":clang_linux_x86_64": [],
    #    "//conditions:default": [],
    #}),
    ],
    includes = [
        "include",
        "src",
        "src/common",
        "src/cpu",
        "src/cpu/gemm",
        "src/cpu/xbyak",
    ],
    nocopts = "-fno-exceptions",
    visibility = ["//visibility:public"],
    deps = [
            "@mkl_linux//:mkl_headers",
            "@mkl_linux//:mkl_libs_linux",
        ],
)

cc_library(
    name = "mkldnn_single_threaded",
    srcs = glob([
        "src/common/*.cpp",
        "src/common/*.hpp",
        "src/cpu/*.cpp",
        "src/cpu/*.hpp",
        "src/cpu/gemm/*.cpp",
        "src/cpu/gemm/*.hpp",
        "src/cpu/gemm/bf16/*.hpp",
        "src/cpu/gemm/bf16/*.cpp",
        "src/cpu/gemm/f32/*.cpp",
        "src/cpu/gemm/f32/*.hpp",
        "src/cpu/gemm/s8x8s32/*.cpp",
        "src/cpu/gemm/s8x8s32/*.hpp",
        "src/cpu/rnn/*.cpp",
        "src/cpu/rnn/*.hpp",
        "src/cpu/xbyak/*.h",
    ]) + [":mkldnn_version_h"],
    hdrs = glob(["include/*"]),
    copts = [
        "-fexceptions",
        "-DMKLDNN_THR=MKLDNN_THR_SEQ",  # Disables threading.
    ],
    includes = [
        "include",
        "src",
        "src/common",
        "src/cpu",
        "src/cpu/gemm",
        "src/cpu/xbyak",
    ],
    visibility = ["//visibility:public"],
)
