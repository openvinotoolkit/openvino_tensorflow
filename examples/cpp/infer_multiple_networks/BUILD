load("//:cxx_abi_option.bzl", "CXX_ABI")
load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_cc_binary")

tf_cc_binary(
    name = 'infer_multi',
    srcs = [
        "main.cc",
        "inference_engine.h",
        "inference_engine.cc",
        "tf_label_image_utils.cc"
    ],
    deps = [
        "@org_tensorflow//tensorflow/cc:cc_ops",
        "@org_tensorflow//tensorflow/cc:client_session",
        "@org_tensorflow//tensorflow/core:tensorflow",
        "//:ngraph_bridge_lib",
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