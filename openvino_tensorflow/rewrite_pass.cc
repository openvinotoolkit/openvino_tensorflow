/*******************************************************************************
 * Copyright (C) 2021 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/

#include <iomanip>
#include <codecvt>

#include "absl/synchronization/mutex.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/graph/graph.h"

#include "api.h"
#include "logging/ovtf_log.h"
#include "logging/tf_graph_writer.h"
#include "openvino_tensorflow/assign_clusters.h"
#include "openvino_tensorflow/backend_manager.h"
#include "openvino_tensorflow/cluster_manager.h"
#include "openvino_tensorflow/deassign_clusters.h"
#include "openvino_tensorflow/encapsulate_clusters.h"
#include "openvino_tensorflow/mark_for_clustering.h"
#include "openvino_tensorflow/ovtf_utils.h"

#include "ocm/include/ocm_nodes_checker.h"
#include "ias3.h" // Telemetry

using namespace std;

namespace tensorflow {
namespace openvino_tensorflow {

//Telemetry
#define FAIL_ON_ERR(e) (void(0))

template < typename T, uint32_t N >
uint32_t countof(T(&)[N])
{
    return std::extent< T[N] >::value;
}
//Telemetry end

class NGraphRewritePass : public GraphOptimizationPass {
 public:
  virtual Status Run(const GraphOptimizationPassOptions& options) = 0;

 protected:
  // Returns a fresh "serial number" to avoid filename collisions in the graph
  // dumps.
  static int FreshIndex() {
    mutex_lock l(s_serial_counter_mutex);
    return s_serial_counter++;
  }

  static int s_serial_counter GUARDED_BY(s_serial_counter_mutex);
  static mutex s_serial_counter_mutex;
};

int NGraphRewritePass::s_serial_counter = 0;
mutex NGraphRewritePass::s_serial_counter_mutex;

//
// Pass that rewrites the graph for nGraph operation.
//
// The pass has several phases, each executed in the below sequence:
//
//   1. Marking [mark_for_clustering.cc]
//   2. Cluster Assignment [assign_clusters.cc]
//   3. Cluster Deassignment [deassign_clusters.cc]
//   4. Cluster Encapsulation [encapsulate_clusters.cc]

class NGraphEncapsulationPass : public NGraphRewritePass {
 public:
  Status Run(const GraphOptimizationPassOptions& options) override {
    // If we don't get a main graph, log that fact and bail.
    if (options.graph == nullptr) {
      OVTF_VLOG(0) << "NGraphEncapsulationPass: options.graph == nullptr";
      return Status::OK();
    }

    if (std::getenv("OPENVINO_TF_DYNAMIC_FALLBACK") != nullptr) {
      int dyn_fallback = std::stoi(std::getenv("OPENVINO_TF_DYNAMIC_FALLBACK"));
      if (dyn_fallback == 0) {
        NGraphClusterManager::DisableClusterFallback();
      } else {
        NGraphClusterManager::EnableClusterFallback();
      }
    } else {
      NGraphClusterManager::EnableClusterFallback();
    }
    bool dynamic_fallback_enabled =
        NGraphClusterManager::IsClusterFallbackEnabled();

    tensorflow::Graph* graph = options.graph->get();
    if (dynamic_fallback_enabled) {
      for (Node* node : graph->nodes()) {
        int cluster;
        Status s = GetNodeAttr(node->attrs(), "_ovtf_cluster", &cluster);
        if (s == Status::OK()) {
          if (NGraphClusterManager::CheckClusterFallback(cluster))
            return Status::OK();
          else
            break;
        } else if (!node->IsSink() && !node->IsSource() &&
                   !node->IsControlFlow() && !node->IsArg() &&
                   !node->IsRetval()) {
          break;
        }
      }
    }

    // For filename generation purposes, grab a fresh index. This is just an
    // arbitrary integer to avoid filename collisions resulting from subsequent
    // runs of this pass.
    int idx = FreshIndex();

    // If requested, dump unmarked graphs.
    util::DumpTFGraph(graph, idx, "unmarked");

    // If ngraph is disabled via openvino_tensorflow api or OPENVINO_TF_DISABLE
    // is set
    // we will not do anything; all subsequent
    // passes become a no-op.
    bool ovtf_not_enabled =
        (!api::IsEnabled()) || (std::getenv("OPENVINO_TF_DISABLE") != nullptr);
    bool already_processed = util::IsAlreadyProcessed(graph);
    if (!already_processed && ovtf_not_enabled) {
      OVTF_VLOG(0) << "NGraph is available but disabled.";
    }
    if (ovtf_not_enabled || already_processed) {
      OVTF_VLOG(1) << std::string("Rewrite pass will not run because ") +
                          (already_processed ? "graph is already preprocessed"
                                             : "ngraph is disabled");
      NGraphClusterManager::EvictAllClusters();
      return Status::OK();
    }

    // Now Process the Graph

    // 1. Mark for clustering then, if requested, dump the graphs.
    std::set<string> skip_these_nodes = {};

    // OCM call for marking supported nodes
    std::string device;
    BackendManager::GetBackendName(device);
    const char* device_id(device.c_str());
    std::string ov_version;
#if defined(OPENVINO_2021_2)
    ov_version = "2021.2";
#elif defined(OPENVINO_2021_3)
    ov_version = "2021.3";
#elif defined(OPENVINO_2021_4)
    ov_version = "2021.4";
#endif

    //Ovtf telemetry initialization
    std::wstring app_name{ L"OVTF" };
    std::wstring telemetry_id{ L"1dac89cb-04a1-4695-86b0-1d95c3d93731" };
    std::wstring app_version{ L"ovtf0.4" };
    std::wstring store_folder;
    std::wstring tel_options{ LR"({"post_did":"generate"})" };
    const wchar_t* const init_keys[] = { L"ikey1", L"ikey2" };
    const wchar_t* const init_vals[] = { L"ival1", L"ival2" };

    ias_handle_t ovtf_handle = IAS3_INVALID_SDK_HANDLE;
    auto res = InitializeEx(
      &ovtf_handle,
      app_name.c_str(),
      app_version.c_str(),
      telemetry_id.c_str(),
      tel_options.c_str(),
      store_folder.c_str(),
      init_keys,
      init_vals,
      countof(init_keys));
    std::wcout << "\033[1;33mInitializeEx finished with res: 0x" << res << "\n";
    FAIL_ON_ERR(res);
    //Telemetry initialization End

    //Ovtf telemetry recording that the TF app has initialized 
    const wchar_t* const ekeys[] = { L"Layer1", L"Layer2", L"Layer3" };
    const wchar_t* const evals[] = { L"eval1", L"eval2", L"eval3" };

    std::wstring event_name{ L"TF_App_Initialized" };
    res = RecordEventEx(ovtf_handle, nullptr, event_name.c_str(), 1, 1.0, ekeys, evals, countof(ekeys));
    std::wcout << "RecordEventEx Res: 0x" << res << std::endl;
    std::wcout << "Event recorded: " << event_name.c_str() << std::endl;
    FAIL_ON_ERR(res);
    //End: Ovtf telemetry recording that the TF app has initialized 

    ocm::Framework_Names fName = ocm::Framework_Names::TF;
    ocm::FrameworkNodesChecker FC(fName, device_id, ov_version,
                                  options.graph->get());
    std::set<std::string> disabled_ops_set = api::GetDisabledOps();
    if (device == "HDDL" && std::getenv("OPENVINO_TF_ENABLE_BATCHING")) {
      std::vector<std::string> batched_disabled_ops = {"Shape"};
      for (int i = 0; i < batched_disabled_ops.size(); i++) {
        disabled_ops_set.insert(batched_disabled_ops[i]);
      }
    }
    FC.SetDisabledOps(disabled_ops_set);
    std::vector<void*> nodes_list = FC.MarkSupportedNodes();

    // cast back the nodes in the TF format and mark the nodes for clustering
    // (moved out from MarkForClustering function)
    const std::map<std::string, SetAttributesFunction>& set_attributes_map =
        GetAttributeSetters();
    for (auto void_node : nodes_list) {
      // TODO(amprocte): move attr name to a constant
      tensorflow::Node* node = (tensorflow::Node*)void_node;
      node->AddAttr("_ovtf_marked_for_clustering", true);
      auto it = set_attributes_map.find(node->type_string());
      if (it != set_attributes_map.end()) {
        it->second(node);
      }
    }

    util::DumpTFGraph(graph, idx, "marked");

    // 2. Assign clusters then, if requested, dump the graphs.
    TF_RETURN_IF_ERROR(AssignClusters(graph));
    util::DumpTFGraph(graph, idx, "clustered");

    // 3. Deassign trivial clusters then, if requested, dump the graphs.
    TF_RETURN_IF_ERROR(DeassignClusters(graph));
    util::DumpTFGraph(graph, idx, "declustered");
      
    // Ovtf telemetry going through the nodes to see which ones are marked for ovtf cluster
    // Naming the ops accordingly and recording them as telemetry events
    bool ovtf_initialized = false;
    bool model_fully_supported = true; // no fallback on tensorflow
    tensorflow::Graph* tel_graph = options.graph->get();
    for (Node* node : tel_graph->nodes()) {
      const auto& optype = node->type_string();
      int cluster = -1; 
      Status s = GetNodeAttr(node->attrs(), "_ovtf_cluster", &cluster);
      std::wstring wop_type; 
      if (s == Status::OK()) {
        if (!ovtf_initialized) ovtf_initialized = true;
        wop_type = std::wstring_convert<std::codecvt_utf8<wchar_t>>().from_bytes(optype);
      } else {
        if ( model_fully_supported && 
             optype.compare("NoOp") != 0 &&
             optype.compare("_Arg") != 0 &&
             optype.compare("_Retval") != 0 ) {
          model_fully_supported = false; 
          wop_type = std::wstring_convert<std::codecvt_utf8<wchar_t>>().from_bytes(optype);
          std::wcout << L"\033[31mNot supported optype: " << wop_type << "\033[0m" << std::endl;
        } 
        std::string ntype;
        std::string fback{"_FallBack"};
        ntype.append(optype);
        ntype.append(fback);
        wop_type = std::wstring_convert<std::codecvt_utf8<wchar_t>>().from_bytes(ntype);
      }
      auto res = RecordEventEx(ovtf_handle, nullptr, wop_type.c_str(), 1, 1.0, ekeys, evals, countof(ekeys));
      std::wcout << "RecordEventEx Res: 0x" << res << std::endl;
      std::wcout << "Recorded event: " << wop_type.c_str() << std::endl;
      FAIL_ON_ERR(res);
    } // End of for loop through nodes
    if (ovtf_initialized) {
      event_name = L"OVTF_Initialized";
      auto res = RecordEventEx(ovtf_handle, nullptr, event_name.c_str(), 1, 1.0, ekeys, evals, countof(ekeys));
      std::wcout << "RecordEventEx Res: 0x" << res << std::endl;
      std::wcout << "Event recorded " << event_name.c_str() << std::endl;
      FAIL_ON_ERR(res);
    }
    event_name = (model_fully_supported) ? L"Model_Fully_Supported" : L"Model_NOT_Fully_Supported";
    res = RecordEventEx(ovtf_handle, nullptr, event_name.c_str(), 1, 1.0, ekeys, evals, countof(ekeys));
    std::wcout << "RecordEventEx Res: 0x" << res << std::endl;
    std::wcout << "Event recorded " << event_name.c_str() << std::endl;
    FAIL_ON_ERR(res);
    // End: Ovtf telemetry recording all ops and flags

    // OVTF telemetry deinitialization and upload
    res = Deinitialize(ovtf_handle);
    std::wcout << "Deinitializing in OVTF " << std::endl;
    std::wcout << "start uploading ..." << std::endl;
    res = Upload(telemetry_id.c_str(), LR"({"show":false, "wait": true})");
    std::wcout << "Upload 0x" << std::setw(8) << res << "\n";
    std::wcout << "uploading finished " << std::endl;
    std::wcout << "--------------------------------------------------------------" << std::endl;
    FAIL_ON_ERR(res);
    // End: OVTF telemetry deinitialization and upload 

    // 4. Encapsulate clusters then, if requested, dump the graphs.
    std::unordered_map<std::string, std::string> config_map;
    auto status = EncapsulateClusters(graph, idx, config_map);
    if (status != Status::OK()) {
      return status;
    }

    util::DumpTFGraph(graph, idx, "encapsulated");
    return Status::OK();
  }
};

}  // namespace openvino_tensorflow

REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_REWRITE_FOR_EXEC, 0,
                      openvino_tensorflow::NGraphEncapsulationPass);
}  // namespace tensorflow
