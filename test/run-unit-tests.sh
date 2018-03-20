#!/usr/bin/env bash
#
# Purpose:
#   Run all unit tests for ngraph-tensorflow integration. This file should be
#   updated whenever a new unit test is updated.
#
# Usage:
#   NGRAPH_VLOG_LEVEL=2 ./run-unit-tests.sh
#
# Todo:
#   - Reduce repeated code once we have more tests
#   - Consider gtest_filter in exclude mode instead of include
#
# The grand goal:
#   Instead of calling individual test files, we should be simply calling
#   ```
#   bazel test //tensorflow/compiler/xla/tests
#   ```
#   to run all XLA_NGRAPH and XLA_CPU tests, once we support all tests.


# Environment setup
set -u

# Print ngraph vlog level
if [ -z ${NGRAPH_VLOG_LEVEL+x} ]
then
    echo "NGRAPH_VLOG_LEVEL is not set, default to 0"
    NGRAPH_VLOG_LEVEL=0
else
    echo NGRAPH_VLOG_LEVEL=${NGRAPH_VLOG_LEVEL}
fi

# The directory of this script
declare THIS_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# # Convert TF_DIR to an absolute path, using a technique that works on all relevant platforms.
# TF_DIR="$(cd "${TF_DIR_RELATIVE}"; pwd)"
TF_DIR="${THIS_SCRIPT_DIR}"

# Concatenate array as a str
function get_gtest_filter_str {
    # prefix
    echo -n "*"
    # body
    local d=":*.";
    echo -n "$1";
    shift;
    printf "%s" "${@/#/$d}";
    # postfix
    echo -n ""
}

declare -i NUM_FAILED=0
declare -i NUM_PASSED=0

# Fetch all the dependencies using this command
#  bazel fetch //tensorflow/compiler/xla/tests:all

# Build and run test
function build_and_run_tests {
    # parse
    local target=$1
    local name_target="${target##*:}"
    local bin_target="${target/:/\/}"
    local enabled_test_names=("${@:2}")
    echo "Test target:" ${target}
    echo "Test name:" ${name_target}
    echo "Test bin target:" ${bin_target}

    # build?
    echo "XLA_NGRAPH_SKIP_UNIT_TEST_REBUILD = '${XLA_NGRAPH_SKIP_UNIT_TEST_REBUILD:-}'"
    if [[ "${XLA_NGRAPH_SKIP_UNIT_TEST_REBUILD:-}" == "1" ]]; then
        echo "Skipping rebuilding of unit tests."
    else
        echo "Rebuilding unit tests."
        bazel build --fetch=false  \
            //${target} --test_output=all --nocache_test_results
    fi

    # Set the LD_LIBRARY_PATH to ngraph_dist/lib
    export USER_PLUGIN_PATH=${TF_DIR}/libngraph_plugin.so
    export XLA_NGRAPH_BACKEND=INTERPRETER
    export LD_LIBRARY_PATH=${HOME}/ngraph_dist/lib

    UNIT_TEST_PROG="${TF_DIR}/bazel-bin/${bin_target}"
    if [[ ! -f "${UNIT_TEST_PROG}" ]]; then
        printf '\nERROR: unit-test program does not exist: %s\n' "${UNIT_TEST_PROG}" >&2
        exit 1
    fi

    # print full list of tests (before filtering)
    echo "All available tests:"
    "${UNIT_TEST_PROG}" --gtest_list_tests

    # Save the results in xUnit XML format
    export GTEST_OUTPUT="xml:${TF_DIR}/unit_test_results_${name_target}.xml"

    # run
    NGRAPH_VLOG_LEVEL=${NGRAPH_VLOG_LEVEL} \
        "${UNIT_TEST_PROG}" \
            --gtest_filter=$(get_gtest_filter_str ${enabled_test_names[@]})
    # store the resulting exit code
    local TEST_EXIT_CODE=${?}

    if (( TEST_EXIT_CODE == 0 )); then
        ((NUM_PASSED += 1))
    else
        ((NUM_FAILED += 1))
    fi
}

# xla/tests:array_elementwise_ops_test_dynamic_plugin
test_target="tensorflow/compiler/xla/tests:pad_test_dynamic_plugin"
declare -a enabled_tests=(
  "Pad1DS0ToS0Array"
  "Pad1DS0ToS5Array"
  "Pad1DS3Array"
  "Pad4D_2x0x3x2_FloatArray"
  "Pad4DFloat_1x1x3x2_Array"
  "Pad4DFloatArrayWithInteriorPadding"
  #"Pad4DFloatArrayMinorFirstSmall"                     # expect fail for now: bridge does not yet handle non-row-major layout
  #"Pad4DFloatArrayMinorFirstNonTrivialMinorDimensions" # expect fail for now: bridge does not yet handle non-row-major layout
  #"Pad4DU8Array"                                       # expect fail for now: U8 not implemented by bridge
  "Pad4DPredArray"
  "Large2DPad"
  "AllTypes2DPad"
  "High2DPad"
  #"NegativePadding2D"                                  # expect fail for now: negative padding not implemented yet in nG++
  #"NegativeAndInteriorPadding2D"                       # expect fail for now: negative padding not implemented yet in nG++
  "ReducePad"
)
build_and_run_tests ${test_target} ${enabled_tests[@]}

# xla/tests:convolution_dimension_numbers_test_dynamic_plugin
test_target="tensorflow/compiler/xla/tests:convolution_dimension_numbers_test_dynamic_plugin"
declare -a enabled_tests=(
  "InvalidInputDimensionNumbers"
  "InvalidWeightDimensionNumbers"
  "TwoConvsWithDifferentDimensionNumbers"
)
build_and_run_tests ${test_target} ${enabled_tests[@]}

# xla/tests:convolution_test_dynamic_plugin
test_target="tensorflow/compiler/xla/tests:convolution_test_dynamic_plugin"
declare -a enabled_tests=(
  "ForwardPassConvolution_3x3x256_256_OutputZ_Iota"
  "Convolve_1x1x1x2_1x1x1x2_Valid"
  "Convolve_1x1x4x4_1x1x2x2_Valid"
  "Convolve_1x1x4x4_1x1x2x2_Same"
  "Convolve_1x1x4x4_1x1x3x3_Same"
  "Convolve1D_1x2x5_1x2x2_Valid"
  "Convolve3D_1x4x2x3x3_2x2x2x3x3_Valid"
)
build_and_run_tests ${test_target} ${enabled_tests[@]}

# xla/tests:convolution_variants_test_dynamic_plugin
test_target="tensorflow/compiler/xla/tests:convolution_variants_test_dynamic_plugin"
declare -a enabled_tests=(
  "Minimal"
  "MinimalWithBatch"
  "Flat1x1"
  "Deep1x1"
  "Filter1x2in1x2"
  "Filter1x2in1x3"
  "Filter1x2in2x2"
  "Filter2x1in2x2"
  "Filter2x2in2x2"
  "Filter1x2in2x3WithDepthAndBatch"
  "Filter1x1stride1x2in1x4"
  "Filter1x1stride1x2in1x5"
  "Filter1x3stride1x2in1x4"
  "Filter1x3stride1x2in1x5"
  "Filter1x1stride2x2in3x3"
  "Filter3x1in1x1Padded"
  "Filter5x1in3x1Padded"
  "Filter3x3in2x2Padded"
  "Filter1x1in2x1WithPaddingAndDepth"
  "Filter2x2Stride1x1Input3x3"
  "Filter1x2Stride1x1Input1x3"
  "Filter2x1x8x8Input1x1x8x8"
  "Filter1x1x1x1Input16x1x1x1"
  "Filter1x1x2x2Input16x1x2x2"
  "Filter1x1x2x2Input3x1x2x2"
  "Filter1x1x8x8Input16x1x8x8"
  "Filter2x2x8x8Input1x2x8x8"
  "Filter2x2x8x8Input2x2x8x8"
  "Filter2x2x8x8Input32x2x8x8"
  "Filter16x16x1x1Input16x16x1x1"
  "FlatRhsDilation"
  "FlatLhsDilation1D"
  "FlatLhsDilation"
  "NegativePaddingOnBothEnds"
  "NegativePaddingLowAndPositivePaddingHigh"
  "PositivePaddingLowAndNegativePaddingHigh"
  "PositivePaddingAndDilation"
  "NegativePaddingAndDilation"
  "RandomData_Input1x1x2x3_Filter2x1x1x2"
  "RandomData_Input1x16x1x1_Filter1x16x1x1"
  "RandomData_Input16x16x1x1_Filter1x16x1x1"
  "RandomData_Input16x16x1x1_Filter16x16x1x1"
  "RandomData_Input16x16x16x16_Filter16x16x16x16"
  "Filter1x2x1x1Input1x2x3x1GeneralPadding"
  "Filter1x1x1x1Input1x2x3x1GeneralPadding"
  "Filter1x1x1x1Input1x2x3x1NoPadding"
  "Filter1x1x2x3Input1x2x3x2NoPadding"
  "BackwardInputLowPaddingLessThanHighPadding"
  "BackwardInputLowPaddingGreaterThanHighPadding"
  "BackwardInputEvenPadding"
  "BackwardInputWithNegativePaddingHigh"
  "BackwardFilterLowPaddingLessThanHighPadding"
  "BackwardFilterLowPaddingGreaterThanHighPadding"
  "BackwardFilterEvenPadding"
  "BackwardInputEvenPadding1D"
  "BackwardFilterEvenPadding1D"
  "BackwardInputEvenPadding3D"
  "BackwardFilterEvenPadding3D"
)
build_and_run_tests ${test_target} ${enabled_tests[@]}

# xla/tests:array_elementwise_ops_test_dynamic_plugin
test_target="tensorflow/compiler/xla/tests:array_elementwise_ops_test_dynamic_plugin"
declare -a enabled_tests=(
  "CompareEqF32s"
  "CompareEqZeroElementF32s"
  "CompareGeF32s"
  "CompareGtF32s"
  "CompareLeF32s"
  "CompareLtF32s"
  "CompareEqS32s"
  "CompareEqZeroElementS32s"
  "CompareNeF32s"
  "CompareNeS32s"
  "CompareGeS32s"
  "CompareGtS32s"
  "CompareLeS32s"
  "CompareLtS32s"
)
build_and_run_tests ${test_target} ${enabled_tests[@]}

# xla/tests:broadcast_simple_test
test_target="tensorflow/compiler/xla/tests:broadcast_simple_test_dynamic_plugin"
declare -a enabled_tests=(
    "ScalarNoOpBroadcast"
    "ScalarTo2D_2x3"
    "ScalarParamTo2D_2x3"
    "ScalarTo2D_2x0"
    "ScalarTo2D_0x2"
    "1DTo2D"
    # "LogicalAnd2DTo3D_Pred"
    "ZeroElement_1DTo2D"
    "1DToZeroElement2D"
    # "InDimensionAndDegenerateBroadcasting"
    # "Add3DTo3DDegenerate_1_2"
    # "Add3DTo3DDegenerate_0_1"
    # "Add3DTo3DDegenerate_0_2"
    # "Add3DTo3DDegenerate_0"
    # "Add3DTo3DDegenerate_1"
    # "Add3DTo3DDegenerate_2"
    # "Add3DTo3DDegenerate_0_1_2"
    # "Add2DTo2DDegenerate_0"
    # "Add2DTo2DDegenerate_1"
    # "Add1DTo3DInDim0"
    # "Add1DTo3DInDim1"
    # "Add1DTo3DInDim2"
    # "Add1DTo3DInDimAll"
    # "Add1DTo3DInDimAllWithScalarBroadcast"
    "InvalidBinaryAndDegenerateBroadcasting"
    "InvalidInDimensionBroadcasting"
    "InvalidDegenerateBroadcasting"
)
build_and_run_tests ${test_target} ${enabled_tests[@]}

# xla/tests:dot_operation_test
test_target="tensorflow/compiler/xla/tests:dot_operation_test_dynamic_plugin"
declare -a enabled_tests=(
    "ZeroElementVectorDotF32"
    "TrivialMatrixVectorDotF32"
    "OneElementVectorDotF32"
    # "OneElementVectorDotF64"
    "VectorDotF32"
    # "VectorDotF64"
    "Dot_0x2_2x0"  # 0d dot bug
    "Dot_0x2_2x3"  # 0d dot bug
    "Dot_3x2_2x0"  # 0d dot bug
    "Dot_2x0_0x2"  # 0d dot bug
    # "MatrixDotF32_12_117_7_MinorToMajorTF"
    # "MatrixDotF32_12_117_7_MinorToMajorFT"
    "MatrixDotF32_12_117_7_MinorToMajorTT"
    # "MatrixDotF32_12_117_7_MinorToMajorFF"
    "MatrixDotF32_270_270_520_MinorToMajorTT"
    # "MatrixDotF32_270_270_520_MinorToMajorTF"
    # "MatrixDotF32_270_270_520_MinorToMajorFT"
    # "MatrixDotF32_270_270_520_MinorToMajorFF"
    "MatrixDotF32_260_3_520_MinorToMajorTT"
    # "MatrixDotF32_260_3_520_MinorToMajorTF"
    # "MatrixDotF32_260_3_520_MinorToMajorFT"
    # "MatrixDotF32_260_3_520_MinorToMajorFF"
    # "SquareMatrixDotF32MinorToMajorFF"
    # "SquareMatrixDotF32MinorToMajorFT"
    # "SquareMatrixDotF32MinorToMajorTF"
    "SquareMatrixDotF32MinorToMajorTT"
    # "SquareMatrixDotF64"
    # "NonsquareMatrixDotF32MajorToMinorFF"
    # "NonsquareMatrixDotF32MajorToMinorFT"
    # "NonsquareMatrixDotF32MajorToMinorTF"
    "NonsquareMatrixDotF32MajorToMinorTT"
    # "NonsquareMatrixDotF64"
    # "ConcurrentMatMul"
    # "BatchMatMul"
    # "TransposeFolding"
)
build_and_run_tests ${test_target} ${enabled_tests[@]}

# xla/tests:log_test
test_target="tensorflow/compiler/xla/tests:log_test_dynamic_plugin"
declare -a enabled_tests=(
    "LogZeroValues"
    "LogTenValues"
)
build_and_run_tests ${test_target} ${enabled_tests[@]}

# xla/tests:tuple_test
test_target="tensorflow/compiler/xla/tests:tuple_test_dynamic_plugin"
declare -a enabled_tests=(
    "TupleConstant"
    "TupleCreate"
    "TupleCreateWithZeroElementEntry"
    "EmptyTupleCreate"
    "GetTupleElement"
    "GetTupleElementWithZeroElements"
    "AddTupleElements"
    "TupleGTEToTuple"
    # "SelectBetweenPredTuples"
    "TupleGTEToTupleToGTEAdd"
    # "SelectBetweenTuplesOnFalse"
    # "TuplesInAMap"
    # "SelectBetweenTuplesOnTrue"
    # "SelectBetweenTuplesElementResult"
    # "SelectBetweenTuplesCascaded"
    # "SelectBetweenTuplesReuseConstants"
    # "NestedTuples"
    # "GetTupleElementOfNestedTuple"
)
build_and_run_tests ${test_target} ${enabled_tests[@]}

# xla/tests:vector_ops_simple_test_dynamic_plugin
test_target="tensorflow/compiler/xla/tests:vector_ops_simple_test_dynamic_plugin"
declare -a enabled_tests=(
    "ExpTenValues"
    "ExpManyValues"
    "ExpIn4D"
  #"NegateTenFloatValues"
  #"NegateTenInt32Values"
  #"NegateUint32Values
  #"SquareTenValues"
  #"ReciprocalTenValues"
  #"SqrtZeroes"
  #"SqrtSixValues"
  #"InvSqrtSevenValues"
  #"AddTenValuesViaMap"
  #"MaxTenValues"
  #"MaxTenValuesFromParams"
  #"Max15000ValuesFromParams"
  #"MaxTenValuesWithScalar"
  #"MinTenValues"
  #"MinMaxTenValues"
  #"ClampTenValuesConstant"
  #"ClampTwoValuesConstant"
  #"ClampTenValuesConstantNonzeroLower"
  #"MapTenValues"
  #"RemainderTenValuesS32"
  #"VectorPredicateEqual"
  #"VectorPredicateNotEqual"
)
build_and_run_tests ${test_target} ${enabled_tests[@]}

# xla/tests:reduce_test_dynamic_plugin
test_target="tensorflow/compiler/xla/tests:reduce_test_dynamic_plugin"
declare -a enabled_tests=(
  "ReduceR1_1_F32_To_R0"
  "ReduceR1_2_F32_To_R0"
  "ReduceR1_16_F32_To_R0"
  "ReduceR1_128_F32_To_R0"
  "ReduceR1_129_F32_To_R0"
  "ReduceR1_240_F32_To_R0"
  "ReduceR1_256_F32_To_R0"
  "ReduceR1_1024_F32_To_R0"
  "ReduceR1_16K_F32_To_R0"
  "ReduceR1_16KP1_F32_To_R0"
  "ReduceR1_64K_F32_To_R0"
  #"ReduceR1_1M_F32_To_R0"
  #ReduceR1_16M_F32_To_R0
  "ReduceR2_0x2_To_R0"
  "ReduceR2_1x1_To_R0"
  "ReduceR2_2x0_To_R0"
  "ReduceR2_2x2_To_R0"
)
build_and_run_tests ${test_target} ${enabled_tests[@]}

# xla/tests:transpose_test_dynamic_plugin
test_target="tensorflow/compiler/xla/tests:transpose_test_dynamic_plugin"
declare -a enabled_tests=(
  "Transpose0x0"
  "Transpose0x42"
  "Transpose7x0"
  "Transpose2x2"
  "Transpose0x2x3_2x3x0"
  #"Transpose1x2x3_2x3x1"
  #"Transpose1x2x3_3x2x1"
  #"Transpose1x2x3_1x2x3"
  "MultiTranspose3x2"
  "Small_1x1"
  "Small_2x2"

)
build_and_run_tests ${test_target} ${enabled_tests[@]}

# xla/tests:convert_test_dynamic_plugin
test_target="tensorflow/compiler/xla/tests:convert_test_dynamic_plugin"
declare -a enabled_tests=(
  "ConvertR1S32ToR1S32"
  "ConvertR1F32ToR1F32"
  "ConvertR1S32ToR1F32"
  "ConvertR1PREDToR1S32"
  "ConvertR1PREDToR1F32"
  "ConvertR1S0S32ToR1S0F32"
  "ConvertR1F32ToR1S32"
  #"ConvertR1S64ToR1F32"
  #ConvertR1U8ToR1F32
  #ConvertR1U8ToR1S32
  #ConvertR1U8ToR1U32
  #ConvertR1F32ToR1F64
  #ConvertR1F64ToR1F32
  "ConvertS32Extremes"
  #ConvertMapToS32
  #ConvertMapToF32
  "ConvertReshape"
)
build_and_run_tests ${test_target} ${enabled_tests[@]}

# xla/tests:select_test_dynamic_plugin
test_target="tensorflow/compiler/xla/tests:select_test_dynamic_plugin"
declare -a enabled_tests=(
  "SelectScalarF32True"
  "SelectScalarS32True"
  "SelectScalarF32False"
  "SelectR1S0F32WithConstantR1S0PRED"
  "SelectR1F32WithConstantR1PRED"
  "SelectR1S0F32WithCmpR1S0S32s"
  "SelectR1F32WithCmpR1S32s"
  "SelectR1F32WithCmpR1F32s"
  "SelectR1F32WithCmpR1F32sFromParamsSmall"
  "SelectR1F32WithCmpR1F32sFromParamsLarge"
  #"SelectR1F32WithCmpR1S32ToScalar"
  #"SelectR1F32WithCmpR1F32ToScalar"
  #"SelectR1S0F32WithScalarPredicate"
  #"SelectR1F32WithScalarPredicateTrue"
  #"SelectR1F32WithScalarPredicateFalse"
)
build_and_run_tests ${test_target} ${enabled_tests[@]}

# xla/tests:reverse_test_dynamic_plugin
test_target="tensorflow/compiler/xla/tests:reverse_test_dynamic_plugin"
declare -a enabled_tests=(
    "ReverseScalar"
    "Reverse0x0FloatArray"
    "Reverse0x1FloatArray"
    "Reverse1x0FloatArray"
    "Reverse1x1FloatArray"
    "Reverse2x0x4x3FloatArrayDim02"
    "Reverse2x0x4x3FloatArrayDim13"
    # "Reverse4DU8ArrayOnDim23"
    "Reverse4DFloatArrayOnDim01"
)
build_and_run_tests ${test_target} ${enabled_tests[@]}

# xla/tests:select_and_scatter_test_dynamic_plugin
test_target="tensorflow/compiler/xla/tests:select_and_scatter_test_dynamic_plugin"
declare -a enabled_tests=(
  # "R1S0F32" # nGraph zero-size tensor bug
  "R1F32"
  "R1S32"
  "R1S32OverlappingWindow"
  "R2S32"
  "ReshapeR2S32"
  "R2S32OverlappingWindow"
  # "R2S32SamePadding" # has padding
  # "R2S32SamePaddingOverlappingWindow" # has padding
  "R2F32OverlappingR2Source"
  "R4F32Valid"
  "R4F32Overlap"
  "R4F32OverlapSmall"
  "R4F32RefValidFixedSmall"
  #"R4F32RefSameRandom" # has padding
  #"R4F32RefSameRandomFullyPadded" # has padding
  # "R4F32RefValidRandom" # CPU precision bug, works with INTERPRETER
  # "R4F32RefValidRandomSmall" # CPU precision bug, works with INTERPRETER
  "R1F32OverlappingWindowMaxScatter"
  # "R1F32OverlappingWindowMinScatter" #CPU precision bug, works with INTERPRETER
)
build_and_run_tests ${test_target} ${enabled_tests[@]}

echo "TensorFlow Unit test results: PASSED: " ${NUM_PASSED}
echo "TensorFlow Unit test results: FAILED: " ${NUM_FAILED}
if [[ -z ${XLA_NGRAPH_BACKEND+x} ]]
then
    echo "nGraph Backend: Default"
else
    echo "nGraph Backend: ${XLA_NGRAPH_BACKEND}"
fi

if (( NUM_FAILED == 0 )); then
    exit 0
else
    exit 1
fi
