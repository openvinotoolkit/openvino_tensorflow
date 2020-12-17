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

#include "test/test_utilities.h"
#include <assert.h>
#include <cassert>
#include <cstdlib>
#include <ctime>
#include "logging/ngraph_log.h"

using namespace std;

namespace ng = ngraph;

namespace tensorflow {
namespace ngraph_bridge {
namespace testing {

void ActivateNGraph() {
  setenv("NGRAPH_TF_DISABLE_DEASSIGN_CLUSTERS", "1", 1);
  unsetenv("NGRAPH_TF_DISABLE");
}

void DeactivateNGraph() {
  unsetenv("NGRAPH_TF_DISABLE_DEASSIGN_CLUSTERS");
  setenv("NGRAPH_TF_DISABLE", "1", 1);
}

// EnvVariable Utilities
bool IsEnvVariableSet(const string& env_var_name) {
  const char* ng_backend_env_value = std::getenv(env_var_name.c_str());
  return (ng_backend_env_value != nullptr);
}

string GetEnvVariable(const string& env_var_name) {
  const char* ng_backend_env_value = std::getenv(env_var_name.c_str());
  NGRAPH_VLOG(5) << "Got Env Variable " << env_var_name << " : "
                 << std::string(ng_backend_env_value);
  return std::string(ng_backend_env_value);
}

void UnsetEnvVariable(const string& env_var_name) {
  NGRAPH_VLOG(5) << "Unsetting " << env_var_name;
  unsetenv(env_var_name.c_str());
}

void SetEnvVariable(const string& env_var_name, const string& env_var_val) {
  setenv(env_var_name.c_str(), env_var_val.c_str(), 1);
  NGRAPH_VLOG(5) << "Setting Env Variable " << env_var_name << " : "
                 << env_var_val;
}

// Store Env Variables
unordered_map<string, string> StoreEnv(list<string> env_vars) {
  unordered_map<string, string> env_map;
  for (auto it = env_vars.begin(); it != env_vars.end(); ++it) {
    string env_name = *it;
    if (IsEnvVariableSet(env_name)) {
      env_map[env_name] = GetEnvVariable(env_name);
      UnsetEnvVariable(env_name);
    }
  }
  return env_map;
}

// Restore
void RestoreEnv(const unordered_map<string, string>& map) {
  for (auto itr : map) {
    setenv(itr.first.c_str(), itr.second.c_str(), 1);
  }
}

// NGRAPH_TF_BACKEND related
bool IsNGraphTFBackendSet() { return IsEnvVariableSet("NGRAPH_TF_BACKEND"); }

string GetBackendFromEnvVar() { return GetEnvVariable("NGRAPH_TF_BACKEND"); }

void UnsetBackendUsingEnvVar() { UnsetEnvVariable("NGRAPH_TF_BACKEND"); }

void SetBackendUsingEnvVar(const string& backend_name) {
  SetEnvVariable("NGRAPH_TF_BACKEND", backend_name);
}

// Generating Seed for PseudoRandomNumberGenerator
unsigned int GetSeedForRandomFunctions() {
  const string& env_name = "NGRAPH_TF_SEED";
  unsigned int seed = static_cast<unsigned>(time(0));
  if (!IsEnvVariableSet(env_name)) {
    NGRAPH_VLOG(5) << "Got seed " << seed;
    return seed;
  }

  string seedstr = GetEnvVariable(env_name);
  try {
    int temp_seed = stoi(seedstr);
    if (temp_seed < 0) {
      throw std::invalid_argument{"Cannot set negative seed"};
    }
    seed = static_cast<unsigned>(temp_seed);
  } catch (const std::exception& exp) {
    throw std::invalid_argument{"Cannot set " + env_name + " with value " +
                                seedstr + ", got exception " + exp.what()};
  }

  NGRAPH_VLOG(5) << "Got seed from " << env_name << " : " << seed;
  return seed;
}

// Input x will be used as an anchor
// Actual value assigned equals to x * i
void AssignInputValuesAnchor(Tensor& A, float x) {
  auto A_flat = A.flat<float>();
  auto A_flat_data = A_flat.data();
  for (int i = 0; i < A_flat.size(); i++) {
    A_flat_data[i] = x * i;
  }
}

// Randomly generate a float number between -10.00 ~ 10.99
void AssignInputValuesRandom(Tensor& A) {
  auto A_flat = A.flat<float>();
  auto A_flat_data = A_flat.data();
  srand(GetSeedForRandomFunctions());
  for (int i = 0; i < A_flat.size(); i++) {
    // give a number between 0 and 20
    float value =
        static_cast<float>(rand()) / static_cast<float>(RAND_MAX / 20.0f);
    value = (value - 10.0f);              // range from -10 to 10
    value = roundf(value * 100) / 100.0;  // change the precision of the float
                                          // to 2 number after the decimal
    A_flat_data[i] = value;
  }
}

void PrintTensor(const Tensor& T1) {
  LOG(INFO) << "print tensor values" << T1.DebugString();
}

// Only displays values in tensor without shape information etc.
void PrintTensorAllValues(const Tensor& T1, int64 max_entries) {
  LOG(INFO) << "all tensor values" << T1.SummarizeValue(max_entries) << endl;
}

std::vector<string> ConvertToString(const std::vector<tensorflow::Tensor> T1) {
  std::vector<string> out;
  for (auto i = 0; i < T1.size(); i++) {
    int total_enteries = 1;
    for (int j = 0; j < T1[i].dims(); j++) {
      total_enteries = total_enteries * T1[i].dim_size(j);
    }
    out.push_back(T1[i].SummarizeValue(total_enteries));
  }
  return out;
}

// Compares Tensors considering tolerance
void Compare(Tensor& T1, Tensor& T2, float tol) {
  // Assert rank
  ASSERT_EQ(T1.dims(), T2.dims())
      << "Ranks unequal for T1 and T2. T1.shape = " << T1.shape()
      << " T2.shape = " << T2.shape();

  // Assert each dimension
  for (int i = 0; i < T1.dims(); i++) {
    ASSERT_EQ(T1.dim_size(i), T2.dim_size(i))
        << "T1 and T2 shapes do not match in dimension " << i
        << ". T1.shape = " << T1.shape() << " T2.shape = " << T2.shape();
  }

  // Assert type
  ASSERT_EQ(T1.dtype(), T2.dtype()) << "Types of T1 and T2 did not match";

  auto T_size = T1.unaligned_flat<float>().size();
  auto T1_data = T1.unaligned_flat<float>().data();
  auto T2_data = T2.unaligned_flat<float>().data();
  for (int k = 0; k < T_size; k++) {
    auto a = T1_data[k];
    auto b = T2_data[k];

    if (std::isnan(a) && std::isnan(b)) {
      continue;
    }
    if (std::isinf(a) && std::isinf(b)) {
      if ((a < 0 && b < 0) || (a > 0 && b > 0)) {
        continue;
      } else {
        ASSERT_TRUE(false) << "Mismatch inf and -inf";
      }
    }
    if (a == 0) {
      EXPECT_NEAR(a, b, tol);
    } else {
      auto rel = a - b;
      auto rel_div = std::abs(rel / a);
      EXPECT_TRUE(rel_div <= tol);
    }
  }
}

// Compares Tensor vectors
void Compare(const vector<Tensor>& v1, const vector<Tensor>& v2, float rtol,
             float atol) {
  ASSERT_EQ(v1.size(), v2.size()) << "Length of 2 tensor vectors do not match.";
  for (size_t i = 0; i < v1.size(); i++) {
    NGRAPH_VLOG(3) << "Comparing output at index " << i;
    auto expected_dtype = v1[i].dtype();
    switch (expected_dtype) {
      case DT_FLOAT:
        Compare<float>(v1[i], v2[i], rtol, atol);
        break;
      case DT_INT8:
        Compare<int8>(v1[i], v2[i]);
        break;
      case DT_INT16:
        Compare<int16>(v1[i], v2[i]);
        break;
      case DT_INT32:
        Compare<int>(v1[i], v2[i]);
        break;
      case DT_INT64:
        Compare<int64>(v1[i], v2[i]);
        break;
      case DT_BOOL:
        Compare<bool>(v1[i], v2[i]);
        break;
      case DT_QINT8:
        Compare<qint8>(v1[i], v2[i]);
        break;
      case DT_QUINT8:
        Compare<quint8>(v1[i], v2[i]);
        break;
      default:
        ASSERT_TRUE(false)
            << "Could not find the corresponding function for the "
               "expected output datatype."
            << expected_dtype;
    }
  }
}

//
template <typename T>
static bool compare_integral_data(T a, T b) {
  return a == b;
}

static bool compare_float_data(float a, float b, float rtol, float atol) {
  if (std::isnan(a) && std::isnan(b)) {
    return true;
  }
  if (std::isinf(a) && std::isinf(b)) {
    return ((a < 0 && b < 0) || (a > 0 && b > 0));
  }
  return std::abs(a - b) <= (atol + rtol * std::abs(a));
}

template <typename T>
void Compare(const Tensor& T1, const Tensor& T2, float rtol, float atol) {
  // Assert rank
  ASSERT_EQ(T1.dims(), T2.dims())
      << "Ranks unequal for T1 and T2. T1.shape = " << T1.shape()
      << " T2.shape = " << T2.shape();

  // Assert each dimension
  for (int i = 0; i < T1.dims(); i++) {
    ASSERT_EQ(T1.dim_size(i), T2.dim_size(i))
        << "T1 and T2 shapes do not match in dimension " << i
        << ". T1.shape = " << T1.shape() << " T2.shape = " << T2.shape();
  }

  // Assert type
  ASSERT_EQ(T1.dtype(), T2.dtype()) << "Types of T1 and T2 did not match";

  bool float_compare = false;
  auto dtype = T1.dtype();
  switch (dtype) {
    case DT_FLOAT:
      float_compare = true;
      break;
    case DT_INT8:
    case DT_INT16:
    case DT_INT32:
    case DT_INT64:
    case DT_BOOL:
    case DT_QINT8:
    case DT_QUINT8:
      float_compare = false;
      break;
    default:
      ASSERT_TRUE(false) << "Could not find the corresponding function for the "
                            "expected output datatype."
                         << dtype;
  }
  auto T_size = T1.flat<T>().size();
  auto T1_data = T1.flat<T>().data();
  auto T2_data = T2.flat<T>().data();
  bool is_comparable = false;

  for (int k = 0; k < T_size; k++) {
    auto a = T1_data[k];
    auto b = T2_data[k];

    if (float_compare) {
      is_comparable = compare_float_data(a, b, rtol, atol);
    } else {
      is_comparable = compare_integral_data<T>(a, b);
    }

    EXPECT_TRUE(is_comparable) << " TF output " << a << endl
                               << " NG output " << b;
  }
}

bool Compare(std::vector<string> desired, std::vector<string> actual) {
  if (desired.size() != actual.size()) {
    return false;
  }
  for (auto i = 0; i < desired.size(); i++) {
    if (desired[i] != actual[i]) {
      return false;
    }
  }
  return true;
}

tf::SessionOptions GetSessionOptions() {
  tf::SessionOptions options;
  options.config.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_opt_level(tf::OptimizerOptions_Level_L0);
  options.config.mutable_graph_options()
      ->mutable_rewrite_options()
      ->set_constant_folding(tf::RewriterConfig::OFF);

  if (is_grappler_enabled()) {
    auto* custom_config = options.config.mutable_graph_options()
                              ->mutable_rewrite_options()
                              ->add_custom_optimizers();

    custom_config->set_name("ngraph-optimizer");
    options.config.mutable_graph_options()
        ->mutable_rewrite_options()
        ->set_min_graph_nodes(-1);

    options.config.mutable_graph_options()
        ->mutable_rewrite_options()
        ->set_meta_optimizer_iterations(tf::RewriterConfig::ONE);
  }
  return options;
}

Status CreateSession(const string& graph_filename,
                     unique_ptr<tf::Session>& session) {
  auto options = GetSessionOptions();
  return LoadGraph(graph_filename, &session, options);
}

Status LoadGraph(const string& graph_file_name,
                 std::unique_ptr<tensorflow::Session>* session,
                 const tensorflow::SessionOptions& options) {
  tensorflow::GraphDef graph_def;
  auto load_graph_status =
      ReadTextProto(Env::Default(), graph_file_name, &graph_def);
  if (!load_graph_status.ok()) {
    return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                        graph_file_name, "'");
  }
  session->reset(tensorflow::NewSession(options));
  return (*session)->Create(graph_def);
}

Status LoadGraphFromPbTxt(const string& pb_file, Graph* input_graph) {
  // Read the graph
  tensorflow::GraphDef graph_def;
  TF_RETURN_IF_ERROR(ReadTextProto(Env::Default(), pb_file, &graph_def));
  GraphConstructorOptions opts;
  opts.allow_internal_ops = true;
  auto status = ConvertGraphDefToGraph(opts, graph_def, input_graph);
  return status;
}

template <>
void AssignInputValues(Tensor& A, int8 x) {
  auto A_flat = A.flat<int8>();
  auto A_flat_data = A_flat.data();
  for (int i = 0; i < A_flat.size(); i++) {
    A_flat_data[i] = x;
  }
  cout << endl;
}

}  // namespace testing
}  // namespace ngraph_bridge
}  // namespace tensorflow
