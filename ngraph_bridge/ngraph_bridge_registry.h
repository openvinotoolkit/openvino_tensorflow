/*******************************************************************************
 * Copyright 2019 Intel Corporation
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
#pragma once

namespace tensorflow {

namespace ngraph_bridge {

// Created a dummy function for bridge specific ops
// so when built static, the linker will include
// all the object files which it would not do normally.
#ifdef NGRAPH_BRIDGE_STATIC_LIB_ENABLE
void register_ngraph_bridge();
#if defined(NGRAPH_TF_ENABLE_VARIABLES_AND_OPTIMIZERS)
void register_ngraph_enable_variable_ops();
#else
void register_ngraph_ops();
#endif
#endif
}
}