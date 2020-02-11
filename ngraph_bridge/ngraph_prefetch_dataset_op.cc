/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
/*******************************************************************************

This file is a copy of
Github repository: https://github.com/tensorflow/tensorflow
Revision: 87989f69597d6b2d60de8f112e1e3cea23be7298
File: tensorflow/core/kernels/data/prefetch_dataset_op.cc

*******************************************************************************/

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

#include "ngraph_bridge/ngraph_prefetch_dataset_op.h"

#include <deque>

#include "tensorflow/core/common_runtime/metrics.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/stats_aggregator.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/error_codes.pb.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"

#include "ngraph_bridge/ngraph_prefetch_shared_data.h"
#include "ngraph_bridge/ngraph_utils.h"
#include "ngraph_bridge/stats_utils.h"

namespace tensorflow {
namespace data {

// See documentation in ../../ops/dataset_ops.cc for a high-level
// description of the following op.

// Determines the fraction of slack time by which to delay prefetching of data.
constexpr double kSleepFactor = 0.2;
constexpr char kDatasetName[] = "NGraphPrefetch";

class NGraphPrefetchDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, const DatasetBase* input, int64 buffer_size,
          int64 slack_period)
      : DatasetBase(DatasetContext(ctx)),
        input_(input),
        buffer_size_(buffer_size),
        slack_period_(slack_period) {
    input_->Ref();
    m_resource_mgr = ctx->resource_manager();
  }

  ~Dataset() override { input_->Unref(); }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return absl::make_unique<Iterator>(
        Iterator::Params{this, strings::StrCat(prefix, "::", kDatasetName)},
        m_resource_mgr);
  }

  const DataTypeVector& output_dtypes() const override {
    return input_->output_dtypes();
  }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    return input_->output_shapes();
  }

  string DebugString() const override {
    return "NGraphPrefetchDatasetOp::Dataset";
  }

  int64 Cardinality() const override { return input_->Cardinality(); }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
    Node* input_graph_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));
    Node* buffer_size = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(buffer_size_, &buffer_size));
    AttrValue slack_period_attr;
    b->BuildAttrValue(slack_period_, &slack_period_attr);
    TF_RETURN_IF_ERROR(b->AddDataset(
        this, {input_graph_node, buffer_size},
        {std::make_pair("slack_period", slack_period_attr)}, output));
    return Status::OK();
  }

 private:
  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const Params& params, ResourceMgr* rm)
        : DatasetIterator<Dataset>(params),
          auto_tuner_(params.dataset->buffer_size_),
          m_resource_mgr(rm),
          m_buffer_size(params.dataset->buffer_size_) {
      slack_us_ = 0;
    }

    ~Iterator() override {
      // Signal the prefetch thread to terminate it. We will then
      // join that thread when we delete `this->prefetch_thread_`.
      //
      // TODO(mrry): Replace this cancellation logic with a
      // CancellationManager. The syntax would be more heavyweight,
      // but it would be possible to thread a cancellation manager
      // through the IteratorContext to upstream,
      // potentially-blocking iterators, when we add these.
      {
        mutex_lock l(mu_);
        cancelled_ = true;
        cond_var_.notify_all();
      }
    }

    string BuildTraceMeName() override {
      int64 buffer_limit;
      {
        tf_shared_lock l(mu_);
        buffer_limit = auto_tuner_.buffer_limit();
      }
      return strings::StrCat(prefix(), "#buffer_limit=", buffer_limit, "#");
    }

    Status Initialize(IteratorContext* ctx) override {
      return dataset()->input_->MakeIterator(ctx, prefix(), &input_impl_);
    }

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
      const auto& stats_aggregator = ctx->stats_aggregator();
      {
        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(EnsurePrefetchThreadStarted(ctx));
        // Wait until the next element in the buffer has been
        // produced, or we are shutting down.
        while (!cancelled_ && buffer_.empty() && !prefetch_thread_finished_ &&
               auto_tuner_.buffer_limit() != 0) {
          auto_tuner_.RecordEmpty();
          RecordStop(ctx);
          cond_var_.wait(l);
          RecordStart(ctx);
        }

        if (cancelled_) {
          return errors::Cancelled(
              "NGraphPrefetchDatasetOp::Dataset::Iterator::GetNext");
        }

        if (!buffer_.empty()) {
          return Consume(ctx, out_tensors, end_of_sequence);
        }

        if (prefetch_thread_finished_) {
          *end_of_sequence = true;
          return Status::OK();
        }

        DCHECK_EQ(auto_tuner_.buffer_limit(), 0);
      }

      mutex_lock parent_l(parent_mu_);
      mutex_lock l(mu_);
      if (stats_aggregator) {
        stats_aggregator->AddScalar(
            stats_utils::BufferSizeScalarName(dataset()->node_name()),
            static_cast<float>(buffer_.size()), num_elements());
        stats_aggregator->AddScalar(
            stats_utils::BufferCapacityScalarName(dataset()->node_name()),
            static_cast<float>(auto_tuner_.buffer_limit()), num_elements());
      }
      return input_impl_->GetNext(ctx, out_tensors, end_of_sequence);
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeAsyncKnownRatioNode(std::move(args),
                                            /*ratio=*/1,
                                            /*parameters=*/{});
    }

    Status SaveInternal(IteratorStateWriter* writer) override {
      // Acquire both locks to ensure that the prefetch thread and
      // all GetNext threads are blocked.
      mutex_lock parent_l(parent_mu_);
      mutex_lock l(mu_);
      TF_RETURN_IF_ERROR(SaveInput(writer, input_impl_));
      TF_RETURN_IF_ERROR(
          writer->WriteScalar(full_name("buffer_size"), buffer_.size()));
      for (size_t i = 0; i < buffer_.size(); i++) {
        auto& buffer_element = buffer_[i];
        TF_RETURN_IF_ERROR(WriteStatus(writer, i, buffer_element.status));
        if (buffer_element.status.ok()) {
          TF_RETURN_IF_ERROR(writer->WriteScalar(
              full_name(strings::StrCat("buffer[", i, "].size")),
              buffer_element.value.size()));
          for (size_t j = 0; j < buffer_element.value.size(); j++) {
            TF_RETURN_IF_ERROR(writer->WriteTensor(
                full_name(strings::StrCat("buffer[", i, "][", j, "]")),
                buffer_element.value[j]));
          }
        }
      }
      return Status::OK();
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
      mutex_lock parent_l(parent_mu_);
      mutex_lock l(mu_);
      buffer_.clear();
      TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
      size_t buffer_size;
      {
        int64 temp;
        TF_RETURN_IF_ERROR(reader->ReadScalar(full_name("buffer_size"), &temp));
        buffer_size = static_cast<size_t>(temp);
      }
      for (size_t i = 0; i < buffer_size; i++) {
        buffer_.emplace_back();
        auto& buffer_element = buffer_.back();
        TF_RETURN_IF_ERROR(ReadStatus(reader, i, &buffer_element.status));
        if (buffer_element.status.ok()) {
          size_t value_size;
          {
            int64 temp;
            TF_RETURN_IF_ERROR(reader->ReadScalar(
                full_name(strings::StrCat("buffer[", i, "].size")), &temp));
            value_size = static_cast<size_t>(temp);
          }
          buffer_element.value.reserve(value_size);
          for (size_t j = 0; j < value_size; j++) {
            buffer_element.value.emplace_back();
            TF_RETURN_IF_ERROR(reader->ReadTensor(
                full_name(strings::StrCat("buffer[", i, "][", j, "]")),
                &buffer_element.value.back()));
          }
        }
      }
      return Status::OK();
    }

   private:
    // A buffer element comprises a status and (if that status is
    // OK) a vector of tensors, representing an element of the input dataset.
    struct BufferElement {
      // The producer sets `status` if getting the input element fails.
      Status status;
      // The buffered data element.
      std::vector<Tensor> value;
      int64 created_us;
    };

    Status Consume(IteratorContext* ctx, std::vector<Tensor>* out_tensors,
                   bool* end_of_sequence) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      NG_TRACE("Prefetch_Consume", "Prefetch_Consume", "");

      const auto& stats_aggregator = ctx->stats_aggregator();
      if (stats_aggregator) {
        stats_aggregator->AddToHistogram(
            stats_utils::BufferUtilizationHistogramName(dataset()->node_name()),
            {static_cast<float>(buffer_.size()) /
             static_cast<float>(auto_tuner_.buffer_limit())},
            num_elements());
        stats_aggregator->AddScalar(
            stats_utils::BufferSizeScalarName(dataset()->node_name()),
            static_cast<float>(buffer_.size()), num_elements());
        stats_aggregator->AddScalar(
            stats_utils::BufferCapacityScalarName(dataset()->node_name()),
            static_cast<float>(auto_tuner_.buffer_limit()), num_elements());
      }
      // A new element is available. Forward the status from computing it, and
      // (if we successfully got an element) the output values.
      Status s = buffer_.front().status;
      if (s.ok()) {
        if (dataset()->slack_period_ > 0 &&
            (num_elements() + 1) % dataset()->slack_period_ == 0) {
          // TODO(rachelim): Consider doing something more sophisticated
          // to decide how long to sleep for; e.g. using a kalman filter.
          int64 slack_us =
              Env::Default()->NowMicros() - buffer_.front().created_us;
          // Every slack_period_-th element, update the most recent slack time,
          // measured by the duration between when the element is prefetched
          // and when it is consumed. We add kSleepFactor * slack_us_ to the
          // measurement because we slept for that duration before prefetching
          // the element.
          slack_us_ = kSleepFactor * slack_us_ + slack_us;
          VLOG(2) << "Setting slack_us_: " << slack_us_;
        }
        *out_tensors = std::move(buffer_.front().value);
        for (auto& next : *out_tensors) {
          NGRAPH_VLOG(2) << "[PREFETCH] CONSUME: Next Tensor: "
                         << next.DebugString();
        }
        RecordBufferDequeue(ctx, *out_tensors);
      }
      auto_tuner_.RecordConsumption(buffer_.size());
      buffer_.pop_front();
      *end_of_sequence = false;

      // Wake the prefetch thread, in case it has been waiting for space
      // in the buffer. Also wake up threads from other calls to GetNext.
      //
      // TODO(mrry): Consider using different condition variables for
      // GetNext and Prefetch.
      cond_var_.notify_all();

      return s;
    }

    Status EnsurePrefetchThreadStarted(IteratorContext* ctx)
        EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      if (!prefetch_thread_) {
        std::shared_ptr<IteratorContext> new_ctx =
            std::make_shared<IteratorContext>(*ctx);
        prefetch_thread_ = ctx->StartThread(
            "tf_data_prefetch", [this, new_ctx]() { PrefetchThread(new_ctx); });
      }
      return Status::OK();
    }

    // Prefetches elements of the input, storing results in an internal
    // buffer.
    //
    // It owns the iterator context passed to it.
    void PrefetchThread(const std::shared_ptr<IteratorContext>& ctx) {
      RecordStart(ctx.get());
      auto cleanup = gtl::MakeCleanup([this, ctx] { RecordStop(ctx.get()); });
      // Keep track of where we are in an iteration "burst"
      int num_produced = 0;
      while (true) {
        NG_TRACE("Prefetch_Produce", "Prefetch_Produce", "");

        // 1. Wait for a slot in the buffer.
        {
          mutex_lock l(mu_);
          while (!cancelled_ && buffer_.size() >= auto_tuner_.buffer_limit()) {
            RecordStop(ctx.get());
            cond_var_.wait(l);
            RecordStart(ctx.get());
          }

          if (cancelled_) {
            return;
          }
        }

        if (dataset()->slack_period_ > 0 &&
            num_produced % dataset()->slack_period_ == 0) {
          // For the first element in the "burst", sleep for a bit if there is
          // slack.
          VLOG(2) << "Sleeping for: " << slack_us_ * kSleepFactor;
          ctx->env()->SleepForMicroseconds(slack_us_ * kSleepFactor);
        }

        // 2. Read the next element.
        // Acquire the parent lock since we will be reading an element
        // from the input iterator. Note that we do not wish to release
        // this lock till we have added the fetched element to the
        // `buffer_` else there will be local state that may be missed
        // by SaveInternal.
        mutex_lock parent_l(parent_mu_);
        bool end_of_sequence;
        BufferElement buffer_element;
        buffer_element.status = input_impl_->GetNext(
            ctx.get(), &buffer_element.value, &end_of_sequence);
        if (buffer_element.status.ok() && end_of_sequence) {
          mutex_lock l(mu_);
          prefetch_thread_finished_ = true;
          NGRAPH_VLOG(2) << "[PREFETCH] Prefetch thread finished";
          cond_var_.notify_all();
          return;
        }

        // Check if the shared data exist
        ngraph_bridge::NGraphPrefetchSharedResouce* shared_data = nullptr;
        Status s = m_resource_mgr->Lookup(
            ngraph_bridge::NGraphPrefetchSharedResouce::CONTAINER_NAME,
            ngraph_bridge::NGraphPrefetchSharedResouce::RESOURCE_NAME,
            &shared_data);
        if (s.ok()) {
          shared_data->SetBufferDepth(m_buffer_size);

          auto ng_input_tensor_bundle =
              shared_data->GetNextIOTensorBundleForDeviceTransfer();
          auto ng_prefetch_input_indexes_map =
              shared_data->GetPrefetchInputIndexesMap();
          NG_TRACE(
              "Prf Dev Copy: Pipe_Ind_" + to_string(ng_input_tensor_bundle.Id),
              "Copy", "");
          int number_of_buffer_elements = buffer_element.value.size();
          if (number_of_buffer_elements !=
              ng_prefetch_input_indexes_map.size()) {
            throw std::runtime_error(
                "Prefetch buffer elements size " +
                to_string(number_of_buffer_elements) +
                " does not match the number of prefetch inputs expected by "
                "encap " +
                to_string(ng_prefetch_input_indexes_map.size()));
          }
          // Write to these tensors
          for (auto itr : ng_prefetch_input_indexes_map) {
            int ng_index = itr.first;
            int tf_index = itr.second;

            ng::element::Type ng_element_type;
            auto status = ngraph_bridge::TFDataTypeToNGraphElementType(
                buffer_element.value[tf_index].dtype(), &ng_element_type);

            void* current_src_ptr =
                (void*)DMAHelper::base(&buffer_element.value[tf_index]);
            NG_TRACE("H2D_PrefetchInput_" + std::to_string(tf_index), "Copy",
                     "");
            try {
              NGRAPH_VLOG(2)
                  << "[PREFETCH] INPUT tensor being written by Prefetch: "
                  << " Value: " << buffer_element.value[tf_index].DebugString();
              ng_input_tensor_bundle.Inputs[ng_index]->write(
                  current_src_ptr,
                  ng_input_tensor_bundle.Inputs[ng_index]->get_element_count() *
                      ng_element_type.size());
            } catch (const std::exception& exp) {
              throw exp;
            } catch (...) {
              throw std::runtime_error(
                  "Error copying TF tensor to device tensor");
            }
          }

          // Now add them back to the other queue
          shared_data->AddNextIOTensorBundleReadyForDeviceExecution(
              ng_input_tensor_bundle);
          shared_data->Unref();
        }

        // 3. Signal that the element has been produced.
        {
          mutex_lock l(mu_);
          RecordBufferEnqueue(ctx.get(), buffer_element.value);
          buffer_element.created_us = ctx->env()->NowMicros();
          buffer_.push_back(std::move(buffer_element));
          cond_var_.notify_all();
        }
        ++num_produced;
      }
    }

    Status WriteStatus(IteratorStateWriter* writer, size_t index,
                       const Status& status) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      TF_RETURN_IF_ERROR(writer->WriteScalar(
          CodeKey(index), static_cast<int64>(status.code())));
      if (!status.ok()) {
        TF_RETURN_IF_ERROR(writer->WriteScalar(ErrorMessageKey(index),
                                               status.error_message()));
      }
      return Status::OK();
    }

    Status ReadStatus(IteratorStateReader* reader, size_t index, Status* status)
        EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      int64 code_int;
      TF_RETURN_IF_ERROR(reader->ReadScalar(CodeKey(index), &code_int));
      error::Code code = static_cast<error::Code>(code_int);

      if (code != error::Code::OK) {
        string error_message;
        TF_RETURN_IF_ERROR(
            reader->ReadScalar(ErrorMessageKey(index), &error_message));
        *status = Status(code, error_message);
      } else {
        *status = Status::OK();
      }
      return Status::OK();
    }

    string CodeKey(size_t index) {
      return full_name(strings::StrCat("status[", index, "].code"));
    }

    string ErrorMessageKey(size_t index) {
      return full_name(strings::StrCat("status[", index, "].error_message"));
    }

    // This mutex is used to ensure exclusivity between multiple threads
    // reading/writing this iterator's local state.
    mutex mu_;
    // This mutex is used to ensure exclusivity between multiple threads
    // accessing the parent iterator. We keep this separate from `mu_` to
    // allow prefetching to run in parallel with GetNext calls.
    mutex parent_mu_ ACQUIRED_BEFORE(mu_);
    std::unique_ptr<IteratorBase> input_impl_ GUARDED_BY(parent_mu_);
    condition_variable cond_var_;
    PrefetchAutotuner auto_tuner_ GUARDED_BY(mu_);
    std::deque<BufferElement> buffer_ GUARDED_BY(mu_);
    std::unique_ptr<Thread> prefetch_thread_ GUARDED_BY(mu_);
    bool cancelled_ GUARDED_BY(mu_) = false;
    bool prefetch_thread_finished_ GUARDED_BY(mu_) = false;

    std::atomic<int64> slack_us_;
    ResourceMgr* m_resource_mgr{nullptr};
    const int m_buffer_size{0};
  };
  const DatasetBase* const input_;
  const int64 buffer_size_;

  // If non-zero, determines the period between injecting "slack" into the
  // execution.
  const int64 slack_period_;

  // Store the resource manager
  ResourceMgr* m_resource_mgr{nullptr};
};

void NGraphPrefetchDatasetOp::MakeDataset(OpKernelContext* ctx,
                                          DatasetBase* input,
                                          DatasetBase** output) {
  int64 buffer_size = 0;
  OP_REQUIRES_OK(ctx,
                 ParseScalarArgument<int64>(ctx, "buffer_size", &buffer_size));
  OP_REQUIRES(ctx,
              buffer_size >= 0 || buffer_size == PrefetchAutotuner::kAutoTune,
              errors::InvalidArgument("buffer_size must be >= 0 or set "
                                      "buffer_size to be ",
                                      PrefetchAutotuner::kAutoTune,
                                      " for auto-tuning"));

  if (buffer_size == PrefetchAutotuner::kAutoTune) {
    metrics::RecordTFDataAutotune(kDatasetName);
  }

  *output = new Dataset(ctx, input, buffer_size, slack_period_);
}

namespace {
REGISTER_KERNEL_BUILDER(
    Name("NGraphPrefetchDataset").Device(DEVICE_CPU).Priority(1),
    NGraphPrefetchDatasetOp);
}  // namespace

}  // namespace data
}  // namespace tensorflow
