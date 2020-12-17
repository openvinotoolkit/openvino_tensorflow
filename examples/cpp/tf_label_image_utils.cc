/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

//-----------------------------------------------------------------------------
// NOTE: This file is taken from tensorflow/examples/label_image/main.cc
// and modified for this example:
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/label_image/main.cc
//-----------------------------------------------------------------------------

/*******************************************************************************
 * Copyright 2019-2020 Intel Corporation
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

// A minimal but useful C++ example showing how to load an Imagenet-style object
// recognition TensorFlow model, prepare input images for it, run them through
// the graph, and interpret the results.
//
// It's designed to have as few dependencies and be as clear as possible, so
// it's more verbose than it could be in production code. In particular, using
// auto for the types of a lot of the returned values from TensorFlow calls can
// remove a lot of boilerplate, but I find the explicit types useful in sample
// code to make it simple to look up the classes involved.
//
// To use it, compile and then run in a working directory with the
// learning/brain/tutorials/label_image/data/ folder below it, and you should
// see the top five labels for the example Lena image output. You can then
// customize it to use your own models or images by changing the file names at
// the top of the main() function.
//
// The googlenet_graph.pb file included by default is created from Inception.
//
// Note that, for GIF inputs, to reuse existing code, only single-frame ones
// are supported.

#include <fstream>
#include <utility>
#include <vector>

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

#include "ngraph_bridge/api.h"

// These are all common classes it's handy to reference with no namespace.
// using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;

//-----------------------------------------------------------------------------
// Takes a file name, and loads a list of labels from it, one per line, and
// returns a vector of the strings. It pads with empty strings so the length
// of the result is a multiple of 16, because our model expects that.
//-----------------------------------------------------------------------------
Status ReadLabelsFile(const string& file_name, std::vector<string>* result,
                      size_t* found_label_count) {
  std::ifstream file(file_name);
  if (!file) {
    return tensorflow::errors::NotFound("Labels file ", file_name,
                                        " not found.");
  }
  result->clear();
  string line;
  while (std::getline(file, line)) {
    result->push_back(line);
  }
  *found_label_count = result->size();
  const int padding = 16;
  while (result->size() % padding) {
    result->emplace_back();
  }
  return Status::OK();
}

//-----------------------------------------------------------------------------
// Reads a binary file and returns a Tensor with the contents read from the
// file pointed to by the "filename"
//-----------------------------------------------------------------------------
static Status ReadEntireFile(tensorflow::Env* env, const string& filename,
                             Tensor* output) {
  tensorflow::uint64 file_size = 0;
  TF_RETURN_IF_ERROR(env->GetFileSize(filename, &file_size));

  string contents;
  contents.resize(file_size);

  std::unique_ptr<tensorflow::RandomAccessFile> file;
  TF_RETURN_IF_ERROR(env->NewRandomAccessFile(filename, &file));

  tensorflow::StringPiece data;
  TF_RETURN_IF_ERROR(file->Read(0, file_size, &data, &(contents)[0]));
  if (data.size() != file_size) {
    return tensorflow::errors::DataLoss("Truncated read of '", filename,
                                        "' expected ", file_size, " got ",
                                        data.size());
  }
  output->scalar<tensorflow::tstring>()() = tensorflow::tstring(data);
  return Status::OK();
}

//-----------------------------------------------------------------------------
// Given an image file name, read in the data, try to decode it as an image,
// resize it to the requested size, and then scale the values as desired.
//-----------------------------------------------------------------------------
Status ReadTensorFromImageFile(const std::vector<string>& file_names,
                               const int input_height, const int input_width,
                               const float input_mean, const float input_std,
                               bool use_NCHW, const int input_channels,
                               std::vector<Tensor>* out_tensors) {
  auto root = tensorflow::Scope::NewRootScope();
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

  string input_name = "file_reader";
  string output_name = "normalized";
  std::vector<std::pair<string, tensorflow::Tensor>> inputs;
  std::vector<tensorflow::Output> div_tensors;

  for (int i = 0; i < file_names.size(); i++) {
    // read file_name into a tensor named input
    Tensor input(tensorflow::DT_STRING, tensorflow::TensorShape());
    TF_RETURN_IF_ERROR(
        ReadEntireFile(tensorflow::Env::Default(), file_names[i], &input));

    // use a placeholder to read input data
    auto file_reader =
        Placeholder(root.WithOpName("input_" + std::to_string(i)),
                    tensorflow::DataType::DT_STRING);

    inputs.push_back({"input_" + std::to_string(i), input});

    // Now try to figure out what kind of file it is and decode it.
    tensorflow::Output image_reader;
    if (absl::EndsWith(file_names[i], ".png")) {
      image_reader = DecodePng(root.WithOpName("png_reader"), file_reader,
                               DecodePng::Channels(input_channels));
    } else if (absl::EndsWith(file_names[i], ".gif")) {
      // gif decoder returns 4-D tensor, remove the first dim
      image_reader =
          Squeeze(root.WithOpName("squeeze_first_dim"),
                  DecodeGif(root.WithOpName("gif_reader"), file_reader));
    } else if (absl::EndsWith(file_names[i], ".bmp")) {
      image_reader = DecodeBmp(root.WithOpName("bmp_reader"), file_reader);
    } else {
      // Assume if it's neither a PNG nor a GIF then it must be a JPEG.
      image_reader = DecodeJpeg(root.WithOpName("jpeg_reader"), file_reader,
                                DecodeJpeg::Channels(input_channels));
    }
    // Now cast the image data to float so we can do normal math on it.
    auto float_caster = Cast(root.WithOpName("float_caster"), image_reader,
                             tensorflow::DT_FLOAT);
    // The convention for image ops in TensorFlow is that all images are
    // expected
    // to be in batches, so that they're four-dimensional arrays with indices of
    // [batch, height, width, channel]. Because we only have a single image, we
    // have to add a batch dimension of 1 to the start with ExpandDims().
    auto dims_expander = ExpandDims(root, float_caster, 0);
    // Bilinearly resize the image to fit the required dimensions.
    auto resized = ResizeBilinear(
        root, dims_expander,
        Const(root.WithOpName("size"), {input_height, input_width}));

    tensorflow::Output div;
    if (use_NCHW) {
      auto converted_input = Transpose(root, resized, {0, 3, 1, 2});
      // Subtract the mean and divide by the scale.
      div = Div(root.WithOpName("output_" + std::to_string(i)),
                Sub(root, converted_input, {input_mean}), {input_std});
    } else {
      // Subtract the mean and divide by the scale.
      div = Div(root.WithOpName("output_" + std::to_string(i)),
                Sub(root, resized, {input_mean}), {input_std});
    }
    div_tensors.push_back(div);
  }

  // skipping concat for a single image
  if (file_names.size() > 1) {
    Concat(root.WithOpName(output_name), div_tensors, 0);
  }

  // This runs the GraphDef network definition that we've just constructed, and
  // returns the results in the output tensor.
  tensorflow::GraphDef graph;
  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

  // Ideally we want to use the nGraph CPU backend for running the following
  // network. But for now we're just disabling nGraph
  // as for some of the devices, these Ops are not implemented
  // tf::Status status =
  //   tf::ngraph_bridge::BackendManager::SetBackendName(backend_name);

  tensorflow::ngraph_bridge::api::disable();
  std::unique_ptr<tensorflow::Session> session(
      tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_RETURN_IF_ERROR(session->Create(graph));
  TF_RETURN_IF_ERROR(
      session->Run({inputs}, {file_names.size() > 1 ? output_name : "output_0"},
                   {}, out_tensors));
  tensorflow::ngraph_bridge::api::enable();

  return Status::OK();
}

//-----------------------------------------------------------------------------
// Reads a model graph definition from disk, and creates a session object you
// can use to run it.
//-----------------------------------------------------------------------------
Status LoadGraph(const string& graph_file_name,
                 std::unique_ptr<tensorflow::Session>* session,
                 const tensorflow::SessionOptions& options) {
  tensorflow::GraphDef graph_def;
  Status load_graph_status =
      ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
  if (!load_graph_status.ok()) {
    return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                        graph_file_name, "'");
  }
  session->reset(tensorflow::NewSession(options));
  Status session_create_status = (*session)->Create(graph_def);
  if (!session_create_status.ok()) {
    return session_create_status;
  }
  return Status::OK();
}

//-----------------------------------------------------------------------------
// Analyzes the output of the Inception graph to retrieve the highest scores and
// their positions in the tensor, which correspond to categories.
//-----------------------------------------------------------------------------
Status GetTopLabels(const std::vector<Tensor>& outputs, int how_many_labels,
                    Tensor* indices, Tensor* scores) {
  auto root = tensorflow::Scope::NewRootScope();
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

  string output_name = "top_k";
  TopK(root.WithOpName(output_name), outputs[0], how_many_labels);
  // This runs the GraphDef network definition that we've just constructed, and
  // returns the results in the output tensors.
  tensorflow::GraphDef graph;
  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

  std::unique_ptr<tensorflow::Session> session(
      tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_RETURN_IF_ERROR(session->Create(graph));
  // The TopK node returns two outputs, the scores and their original indices,
  // so we have to append :0 and :1 to specify them both.
  std::vector<Tensor> out_tensors;
  TF_RETURN_IF_ERROR(session->Run({}, {output_name + ":0", output_name + ":1"},
                                  {}, &out_tensors));
  *scores = out_tensors[0];
  *indices = out_tensors[1];
  return Status::OK();
}

//-----------------------------------------------------------------------------
// Given the output of a model run, and the name of a file containing the labels
// this prints out the top five highest-scoring values.
//-----------------------------------------------------------------------------
Status PrintTopLabels(const std::vector<Tensor>& outputs,
                      const string& labels_file_name) {
  // Disable nGraph so that we run these using TensorFlow on CPU
  tensorflow::ngraph_bridge::api::disable();
  std::vector<string> labels;
  size_t label_count;
  Status read_labels_status =
      ReadLabelsFile(labels_file_name, &labels, &label_count);
  if (!read_labels_status.ok()) {
    LOG(ERROR) << read_labels_status;
    return read_labels_status;
  }
  const int how_many_labels = std::min(5, static_cast<int>(label_count));
  Tensor indices;
  Tensor scores;
  TF_RETURN_IF_ERROR(GetTopLabels(outputs, how_many_labels, &indices, &scores));
  tensorflow::TTypes<float>::Flat scores_flat = scores.flat<float>();
  tensorflow::TTypes<int32>::Flat indices_flat = indices.flat<int32>();
  for (int pos = 0; pos < how_many_labels; ++pos) {
    const int label_index = indices_flat(pos);
    const float score = scores_flat(pos);
    std::cout << labels[label_index] << " (" << label_index << "): " << score
              << "\n";
  }
  return Status::OK();
}

//-----------------------------------------------------------------------------
// This is a testing function that returns whether the top label index is the
// one that's expected.
//-----------------------------------------------------------------------------
Status CheckTopLabel(const std::vector<Tensor>& outputs, int expected,
                     bool* is_expected) {
  tensorflow::ngraph_bridge::api::disable();
  *is_expected = false;
  Tensor indices;
  Tensor scores;
  const int how_many_labels = 1;
  TF_RETURN_IF_ERROR(GetTopLabels(outputs, how_many_labels, &indices, &scores));
  tensorflow::TTypes<int32>::Flat indices_flat = indices.flat<int32>();
  if (indices_flat(0) != expected) {
    LOG(ERROR) << "Expected label #" << expected << " but got #"
               << indices_flat(0);
    *is_expected = false;
  } else {
    *is_expected = true;
  }
  return Status::OK();
}
