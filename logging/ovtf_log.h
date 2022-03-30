/*******************************************************************************
 * Copyright (C) 2021-2022 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/

#ifndef NGRAPH_LOG_H_
#define NGRAPH_LOG_H_

#include <string>
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/platform/env_time.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/public/version.h"

class NGraphLogMessage : public tensorflow::internal::LogMessage {
 public:
  static tensorflow::int64 MinNGraphVLogLevel();
  static std::string GetTimeStampForLogging();
};

#define OVTF_VLOG_IS_ON(lvl) ((lvl) <= NGraphLogMessage::MinNGraphVLogLevel())

#define OVTF_VLOG(lvl)      \
  if (OVTF_VLOG_IS_ON(lvl)) \
  ::tensorflow::internal::LogMessage(__FILE__, __LINE__, tensorflow::INFO)

#endif  // NGRAPH_LOG_H_
