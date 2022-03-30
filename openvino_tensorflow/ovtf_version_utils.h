/*******************************************************************************
 * Copyright (C) 2021-2022 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/
#ifndef OPENVINO_TF_BRIDGE_VERSION_UTILS_H_
#define OPENVINO_TF_BRIDGE_VERSION_UTILS_H_

#include "tensorflow/core/public/version.h"

#define TF_VERSION_GEQ(REQ_TF_MAJ_VER, REQ_TF_MIN_VER) \
  ((TF_MAJOR_VERSION > REQ_TF_MAJ_VER) ||              \
   ((TF_MAJOR_VERSION == REQ_TF_MAJ_VER) &&            \
    (TF_MINOR_VERSION >= REQ_TF_MIN_VER)))

#endif  // OPENVINO_TF_BRIDGE_VERSION_UTILS_H_