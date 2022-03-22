
#include <openvino_tensorflow/ovtf_decoder.h>
#include <ngraph/ngraph.hpp>
#include <openvino/core/type/element_type.hpp>
#include <openvino/frontend/tensorflow/decoder.hpp>
#include <openvino/frontend/tensorflow/frontend.hpp>
#include <string>
#include <vector>

namespace tensorflow {
namespace openvino_tensorflow {

namespace {
const std::map<::tensorflow::DataType, ov::element::Type>& TYPE_MAP() {
  static const std::map<::tensorflow::DataType, ov::element::Type> type_map{
      {::tensorflow::DataType::DT_BOOL, ov::element::boolean},
      {::tensorflow::DataType::DT_INT16, ov::element::i16},
      {::tensorflow::DataType::DT_INT32, ov::element::i32},
      {::tensorflow::DataType::DT_INT64, ov::element::i64},
      {::tensorflow::DataType::DT_HALF, ov::element::f16},
      {::tensorflow::DataType::DT_FLOAT, ov::element::f32},
      {::tensorflow::DataType::DT_DOUBLE, ov::element::f64},
      {::tensorflow::DataType::DT_UINT8, ov::element::u8},
      {::tensorflow::DataType::DT_INT8, ov::element::i8},
      {::tensorflow::DataType::DT_BFLOAT16, ov::element::bf16}};
  return type_map;
}
}  // namespace

// ov::Any OVTFDecoder::get_attribute(const std::string& name,
//                                    const std::type_info& type_info) const {
//   auto attrs = decode_attribute_helper(name);
//   if (attrs.empty()) {
//     return {};
//   }

//   if (type_info == typeid(std::string)) {
//     return attrs[0].s();
//   } else if (type_info == typeid(int64_t)) {
//     return attrs[0].i();
//   } else if (type_info == typeid(std::vector<int64_t>)) {
//     std::vector<int64_t> longs;
//     longs.reserve(attrs[0].list().i_size());
//     for (size_t idx = 0; idx < attrs[0].list().i_size(); ++idx) {
//       longs.push_back(attrs[0].list().i(idx));
//     }
//     return longs;
//   } else if (type_info == typeid(int32_t)) {
//     return static_cast<int32_t>(attrs[0].i());
//   } else if (type_info == typeid(std::vector<int32_t>)) {
//     std::vector<int32_t> ints;
//     ints.reserve(attrs[0].list().i_size());
//     for (size_t idx = 0; idx < attrs[0].list().i_size(); ++idx) {
//       ints.push_back(static_cast<int32_t>(attrs[0].list().i(idx)));
//     }
//     return ints;
//   } else if (type_info == typeid(float)) {
//     return attrs[0].f();
//   } else if (type_info == typeid(std::vector<float>)) {
//     std::vector<float> floats;
//     floats.reserve(attrs[0].list().i_size());
//     for (size_t idx = 0; idx < attrs[0].list().i_size(); ++idx) {
//       floats.push_back(attrs[0].list().f(idx));
//     }
//     return floats;
//   } else if (type_info == typeid(ov::element::Type)) {
//     auto data_type = attrs[0].type();
//     return TYPE_MAP().at(data_type);
//   } else if (type_info == typeid(bool)) {
//     return attrs[0].b();
//   } else if (type_info == typeid(::tensorflow::DataType)) {
//     return attrs[0].type();
//   } else if (type_info == typeid(::tensorflow::TensorProto)) {
//     return attrs[0].tensor();
//   } else if (type_info == typeid(::ov::PartialShape)) {
//     std::vector<ov::Dimension> dims;
//     auto tf_shape = attrs[0].shape();
//     for (int i = 0; i < tf_shape.dim_size(); i++) {
//       dims.push_back(tf_shape.dim(i).size());
//     }
//     auto pshape = ov::PartialShape(dims);
//     return pshape;
//   }

//   // type is not supported by decoder
//   return {};
// }

ov::Any OVTFDecoder::get_attribute(const std::string& name) const {
  auto attrs = decode_attribute_helper(name);
  if (attrs.empty()) {
    return {};
  }

  switch (attrs[0].value_case()) {
    case ::tensorflow::AttrValue::ValueCase::kB:
      return attrs[0].b();
    case ::tensorflow::AttrValue::ValueCase::kF:
      return attrs[0].f();
    case ::tensorflow::AttrValue::ValueCase::kS:
      return attrs[0].s();
    case ::tensorflow::AttrValue::ValueCase::kI:
      return attrs[0].i();
    case ::tensorflow::AttrValue::ValueCase::kShape: {
      std::vector<ov::Dimension> dims;
      const auto& tf_shape = attrs[0].shape();
      for (int i = 0; i < tf_shape.dim_size(); i++) {
        dims.emplace_back(tf_shape.dim(i).size());
      }
      return ov::PartialShape(dims);
    }

    case ::tensorflow::AttrValue::ValueCase::kType:
      return TYPE_MAP().at(attrs[0].type());

    case ::tensorflow::AttrValue::ValueCase::kList: {
      const auto& list = attrs[0].list();
      if (list.i_size())
        return std::vector<int64_t>(list.i().begin(), list.i().end());

      if (list.f_size())
        return std::vector<float>(list.f().begin(), list.f().end());

      if (list.s_size())
        return std::vector<std::string>(list.s().begin(), list.s().end());

      if (list.b_size())
        return std::vector<bool>(list.b().begin(), list.b().end());

      if (list.shape_size()) {
        std::vector<ov::PartialShape> res;
        for (const auto& it : list.shape()) {
          std::vector<ov::Dimension> dims;
          for (int i = 0; i < it.dim_size(); i++) {
            dims.emplace_back(it.dim(i).size());
          }
          res.emplace_back(dims);
        }
      }

      if (list.type_size()) {
        std::vector<ov::element::Type> res;
        for (int idx = 0; idx < list.type_size(); ++idx) {
          res.emplace_back(TYPE_MAP().at(list.type(idx)));
        }
        return res;
      }

      if (list.tensor_size() || list.func_size())
        FRONT_END_GENERAL_CHECK(false,
                                "Conversion from tensorflow data type to "
                                "openvino data type is not supported.");
    }

    case ::tensorflow::AttrValue::ValueCase::kTensor:
      return attrs[0].tensor();
    case ::tensorflow::AttrValue::ValueCase::kPlaceholder:
    case ::tensorflow::AttrValue::ValueCase::kFunc:
    default:
      FRONT_END_GENERAL_CHECK(false,
                              "Conversion from tensorflow data type to "
                              "openvino data type is not supported.");
  }
}

ov::Any OVTFDecoder::get_native_attribute(const std::string& name) const {
  auto attrs = decode_attribute_helper(name);
  if (attrs.empty()) {
    return {};
  }

  switch (attrs[0].value_case()) {
    case ::tensorflow::AttrValue::ValueCase::kTensor:
      return attrs[0].tensor();
    case ::tensorflow::AttrValue::ValueCase::kType:
      return attrs[0].type();
    default:
      FRONT_END_GENERAL_CHECK(false, "Data type is not covered.");
  }
}

size_t OVTFDecoder::get_input_size() const { return m_node_def->input_size(); }

void OVTFDecoder::get_input_node(size_t input_port_idx,
                                 std::string& producer_name,
                                 size_t& producer_output_port_index) const {
  std::string producer_port_name = m_node_def->input(input_port_idx);
  auto delim_pos = producer_port_name.find(':');
  if (delim_pos != std::string::npos) {
    producer_name = producer_port_name.substr(0, delim_pos);
    std::string p_p_idx_str = producer_port_name.substr(delim_pos + 1);
    producer_output_port_index =
        std::stoi(producer_port_name.substr(delim_pos + 1));
    return;
  }
  producer_name = producer_port_name;
  producer_output_port_index = 0;
}

const std::string& OVTFDecoder::get_op_type() const { return m_node_def->op(); }

const std::string& OVTFDecoder::get_op_name() const {
  return m_node_def->name();
}

vector<::tensorflow::AttrValue> OVTFDecoder::decode_attribute_helper(
    const string& name) const {
  auto attr_map = m_node_def->attr();
  if (attr_map.contains(name)) {
    auto value = m_node_def->attr().at(name);
    return {value};
  } else {
    return {};
  }
}

}  // namespace openvino_tensorflow
}  // namespace tensorflow
