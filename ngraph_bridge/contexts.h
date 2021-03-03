#ifndef CONTEXTS_H
#define CONTEXTS_H

#include <inference_engine.hpp>

namespace tensorflow {
namespace ngraph_bridge {

struct GlobalContext {
  InferenceEngine::Core ie_core;
};

} // namespace ngraph_bridge
} // namespace tensorflow

#endif



