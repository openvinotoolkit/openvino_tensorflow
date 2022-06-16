/*******************************************************************************
 * Copyright (C) 2021-2022 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/

#ifndef OPENVINO_LOG_H_
#define OPENVINO_LOG_H_

#include <string>
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/platform/env_time.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/public/version.h"

class OpenVINOLogMessage : public tensorflow::internal::LogMessage {
 public:
  static tensorflow::int64 MinOpenVINOVLogLevel();
  static std::string GetTimeStampForLogging();
};

#define OVTF_VLOG_IS_ON(lvl) ((lvl) <= OpenVINOLogMessage::MinOpenVINOVLogLevel())

#define OVTF_VLOG(lvl)      \
  if (OVTF_VLOG_IS_ON(lvl)) \
  ::tensorflow::internal::LogMessage(__FILE__, __LINE__, tensorflow::INFO)

#endif  // OPENVINO_LOG_H_
