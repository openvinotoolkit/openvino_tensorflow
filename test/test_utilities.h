/*******************************************************************************
 * Copyright 2017-2020 Intel Corporation
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
#ifndef NGRAPH_TF_BRIDGE_TESTUTILITIES_H_
#define NGRAPH_TF_BRIDGE_TESTUTILITIES_H_

#include "gtest/gtest.h"

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/public/session.h"

#include "ngraph/ngraph.hpp"
#include "ngraph_bridge/version.h"

// Define useful macros used by others
#if !defined(ASSERT_OK)
#define ASSERT_OK(x) ASSERT_EQ((x), ::tensorflow::Status::OK())
#endif

#if !defined(ASSERT_NOT_OK)
#define ASSERT_NOT_OK(x) ASSERT_NE((x), ::tensorflow::Status::OK());
#endif

using namespace std;
namespace ng = ngraph;
namespace tf = tensorflow;

namespace tensorflow {
namespace ngraph_bridge {
namespace testing {

// Activate and Deactivate NGraph
void ActivateNGraph();
void DeactivateNGraph();

// Store Env Variables
// This function takes a list of env var that the user would
// want to change for his particular test scenario and hence
// save the current value for if it is set/unset
unordered_map<string, string> StoreEnv(list<string> env_vars);

// Restore Env Variables
// This function takes the map of <env var, val> created using
// StoreEnv and restores the env variables to their
// previous state
void RestoreEnv(const unordered_map<string, string>& map);

// EnvVariable Utilities
bool IsEnvVariableSet(const string& env_var_name);
string GetEnvVariable(const string& env_var_name);
void UnsetEnvVariable(const string& env_var_name);
void SetEnvVariable(const string& env_var_name, const string& env_var_val);

// NGRAPH_TF_BACKEND related
bool IsNGraphTFBackendSet();
string GetBackendFromEnvVar();
void UnsetBackendUsingEnvVar();
void SetBackendUsingEnvVar(const string& bname);

// Print Functions
void PrintTensor(const Tensor& T1);
void PrintTensorAllValues(
    const Tensor& T1,
    int64 max_entries);  // print max_entries of elements in the Tensor

// Get a scalar value from a tensor, optionally at an element offset
template <typename T>
static T GetScalarFromTensor(const std::shared_ptr<ngraph::runtime::Tensor>& t,
                             size_t element_offset = 0) {
  T result;
  t->read(&result, sizeof(T));
  return result;
}

// Prints the tensor to the given output stream
// TODO: internally convert ng types to cpptypes
// so that users do not have to specify the template arg T
template <typename T>
std::ostream& DumpNGTensor(std::ostream& s, const string& name,
                           const std::shared_ptr<ngraph::runtime::Tensor>& t) {
  // std::shared_ptr<ngraph::runtime::Tensor> t{get_tensor()};
  const ngraph::Shape& shape = t->get_shape();
  s << "Tensor<" << name << ": ";
  auto type = t->get_element_type();
  bool T_is_integral = std::is_integral<T>::value;
  bool type_is_integral = type.is_integral();
  if (type_is_integral != T_is_integral) {
    std::stringstream err_msg;
    err_msg << "Tensor type " << type << " is"
            << (type_is_integral ? " " : " not ")
            << "integral but passed template is"
            << (T_is_integral ? " " : " not ") << "integral";
    throw std::invalid_argument(err_msg.str());
  }

  for (size_t i = 0; i < shape.size(); ++i) {
    s << shape.at(i);
    if (i + 1 < shape.size()) {
      s << ", ";
    }
  }
  size_t pos = 0;
  s << ">{";
  size_t rank = shape.size();
  if (rank == 0) {
    s << GetScalarFromTensor<T>(t, pos++);
  } else if (rank <= 2) {
    s << "[";
    for (size_t i = 0; i < shape.at(0); ++i) {
      if (rank == 1) {
        s << GetScalarFromTensor<T>(t, pos++);
      } else if (rank == 2) {
        s << "[";
        for (size_t j = 0; j < shape.at(1); ++j) {
          s << GetScalarFromTensor<T>(t, pos++);

          if (j + 1 < shape.at(1)) {
            s << ", ";
          }
        }
        s << "]";
      }
      if (i + 1 < shape.at(0)) {
        s << ", ";
      }
    }
    s << "]";
  }
  // TODO: extend for > 2 rank
  s << "}";
  return s;
}

std::vector<string> ConvertToString(const std::vector<tensorflow::Tensor>);

// Generating Random Seed
unsigned int GetSeedForRandomFunctions();

// Assignment Functions
// TODO : Retire AssignInputValuesAnchor and AssignInputValuesRandom
void AssignInputValuesAnchor(Tensor& A, float x);  // value assigned = x * index
void AssignInputValuesRandom(Tensor& A);

// Assigns value x to all the elements of the tensor
template <typename T>
void AssignInputValues(Tensor& A, T x) {
  auto A_flat = A.flat<T>();
  auto A_flat_data = A_flat.data();
  for (int i = 0; i < A_flat.size(); i++) {
    A_flat_data[i] = x;
  }
}

template <>
void AssignInputValues(Tensor& A, int8 x);

// Assigns values from the vector x to the Tensor
template <typename T>
void AssignInputValues(Tensor& A, vector<T> x) {
  auto A_flat = A.flat<T>();
  auto A_flat_data = A_flat.data();
  assert(A_flat.size() == x.size());
  for (int i = 0; i < A_flat.size(); i++) {
    A_flat_data[i] = x[i];
  }
}

// Assigns random values in range [min, max] to the Tensor
// Randomly generate data with specified type to populate the Tensor
// Random data is generated within range (min, max)
template <typename T>
void AssignInputValuesRandom(Tensor& A, T min, T max) {
  auto A_flat = A.flat<T>();
  auto A_flat_data = A_flat.data();
  srand(GetSeedForRandomFunctions());
  for (int i = 0; i < A_flat.size(); i++) {
    T value =
        // randomly generate a number between 0 and (max-min) inclusive
        static_cast<T>(rand()) / static_cast<T>(RAND_MAX / (max - min + 1));
    value = value + min;  // transform the range to (min, max) inclusive
    A_flat_data[i] = value;
  }
}

// Compares two Tensor vectors
void Compare(const vector<Tensor>& v1, const vector<Tensor>& v2,
             float rtol = static_cast<float>(1e-05),
             float atol = static_cast<float>(1e-08));

bool Compare(std::vector<string> arg0, std::vector<string> arg1);

// Compares two Tensors
// Right now only tensors contain float values will modify the tolerance
// parameters
template <typename T>
void Compare(const Tensor& T1, const Tensor& T2,
             float rtol = static_cast<float>(1e-05),
             float atol = static_cast<float>(1e-08));

// Compares Tensors considering tolerance
void Compare(Tensor& T1, Tensor& T2, float tol);

tf::SessionOptions GetSessionOptions();

Status CreateSession(const string& graph_filename,
                     unique_ptr<tf::Session>& session);

Status LoadGraph(const string& graph_file_name,
                 std::unique_ptr<tensorflow::Session>* session,
                 const tensorflow::SessionOptions& options);

Status LoadGraphFromPbTxt(const string& pb_file, Graph* input_graph);

template <typename T>
size_t count_ops_of_type(std::shared_ptr<ng::Function> f) {
  size_t count = 0;
  for (auto op : f->get_ops()) {
    if (ng::is_type<T>(op)) {
      count++;
    }
  }

  return count;
}

}  // namespace testing
}  // namespace ngraph_bridge
}  // namespace tensorflow

#endif  // NGRAPH_TF_BRIDGE_TESTUTILITIES_H_
