#!/bin/bash
# This script is used to fetch/run models from a curated model-repo for OV-IE integration project
# Invoke locally: .../run_infer_single.sh resnet_50 bike 'mountain bike, all-terrain bike'

MODEL=$1
if [ "${BUILDKITE}" == "true" ]; then
    echo "--- Model: ${MODEL}"
fi
if [ "${MODEL}" == "" ]; then
    echo "Error: model not specified!" && exit 1
fi
IMAGE=$2
if [ "${IMAGE}" == "" ]; then
    echo "Error: image not specified!" && exit 1
elif [[ ! ${IMAGE} =~ *.* ]]; then
    IMAGE="${IMAGE}.jpg"
fi
INFER_PATTERN=$3
if [ "${INFER_PATTERN}" == "" ]; then
    echo "Error: expected pattern not specified!" && exit 1
fi
BENCHMARK=$4 # YES or NO
if [ "${BENCHMARK}" == "" ]; then
    echo "Error: benchmark flag (YES/NO) not specified!" && exit 1
fi

echo MODEL=$MODEL IMAGE=$IMAGE INFER_PATTERN=$INFER_PATTERN BENCHMARK=$BENCHMARK

REPO=https://gitlab.devtools.intel.com/mcavus/tensorflow_openvino_models_public
COMMIT=d12f2d57 # 2021-Jan-06

WORKDIR=`pwd`

if [ "${BUILDKITE}" == "true" ]; then
    LOCALSTORE_PREFIX=/localdisk/buildkite/artifacts
else
    LOCALSTORE_PREFIX=/tmp
fi
LOCALSTORE=${LOCALSTORE_PREFIX}/$(basename $REPO)

function gen_frozen_models {
    script=$1

    initdir=`pwd`
    VENVTMP="$WORKDIR/venv_temp" # to ensure no side-effects of any pip installs
    [ -d $VENVTMP ] && rm -rf $VENVTMP
    virtualenv -p python3 $VENVTMP
    source $VENVTMP/bin/activate
    $script || exit 1
    deactivate
    cd ${initdir}
    rm -rf $VENVTMP
}

function get_model_repo {
    pushd . >/dev/null
    if [ ! -d "${LOCALSTORE}" ]; then
        cd ${LOCALSTORE_PREFIX} || exit 1
        git clone ${REPO}
        # check if successful...
        if [ ! -d "${LOCALSTORE}" ]; then echo "Failed to clone repo!"; exit 1; fi
        # init the models...
        cd ${LOCALSTORE} || exit 1
        git checkout ${COMMIT} || exit 1
        gen_frozen_models ./model_factory/create.all
        echo Downloaded all models; echo
    fi

    cd ${LOCALSTORE} || exit 1
    prev_commit=$(git rev-parse HEAD)
    commit_check=$(git cat-file -t $COMMIT 2>/dev/null)
    [ "$commit_check" == "commit" ] || ( git fetch || exit 1 )
    desired_commit=$(git rev-parse $COMMIT)
    
    if [ -d "temp_build" ]; then rm -rf temp_build; fi

    if [ ! -f "${LOCALSTORE}/frozen/${MODEL}.pb" ] || \
    [ ! -f "${LOCALSTORE}/frozen/${MODEL}.txt" ] || \
    [ "$desired_commit" != "$prev_commit" ]; then
        git checkout ${COMMIT} || exit 1
        gen_frozen_models ./model_factory/create_${MODEL}.sh
        touch "${LOCALSTORE}/frozen/${MODEL}.pb"
        touch "${LOCALSTORE}/frozen/${MODEL}.txt"
        echo Downloaded model ${MODEL}; echo
    fi
    
    if [ "${BENCHMARK}" == "YES" ]; then
        if [ ! -f "${LOCALSTORE}/IR/${MODEL}_batch1.xml" ] || [ ! -f "${LOCALSTORE}/IR/${MODEL}_batch1.bin" ]; then
            opv_root=${INTEL_OPENVINO_DIR:-"/opt/intel/openvino"}
            mo_tf_path="${opv_root}/deployment_tools/model_optimizer/mo_tf.py"
            [ -f "${mo_tf_path}" ] || ( echo "${mo_tf_path} not found!"; exit 1 )
            export INTEL_OPENVINO_DIR="${opv_root}"
            ./model_factory/generate_ir.sh ${MODEL} 1 || exit 1
        fi
    fi
    [ -d "${LOCALSTORE}/demo/outputs" ] || mkdir "${LOCALSTORE}/demo/outputs"
    popd  >/dev/null
}

function print_infer_times {
    NUM_ITER=$1
    WARMUP_ITERS=$2
    TMPFILE=$3
    INFER_TIME_FIRST_ITER="?"
    if (( $NUM_ITER > 1 )); then
        INFER_TIME_FIRST_ITER=$( grep "Inf Execution Time" ${TMPFILE} | head -n 1 | rev | cut -d' ' -f 1 | rev )
        INFER_TIME_FIRST_ITER=$( printf %.04f ${INFER_TIME_FIRST_ITER} )
    fi
    INFER_TIME=$(get_average_infer_time "${WARMUP_ITERS}" "${TMPFILE}")
    echo INFER_TIME Avg of $((NUM_ITER - WARMUP_ITERS)) iters = ${INFER_TIME} seconds, 1st = ${INFER_TIME_FIRST_ITER}
}

function get_average_infer_time {
    num_warmup_iters=$1
    logfile=$2
    count=0
    total=0
    warmup_iters_time=0
    for i in $( grep "Inf Execution Time" "$logfile" | rev | cut -d' ' -f 1 | rev )
    do 
        total=$(echo $total+$i | bc )
        (( count < $num_warmup_iters )) && warmup_iters_time=$(echo $warmup_iters_time+$i | bc )
        ((count++))
    done
    (( count > $num_warmup_iters )) && total=$(echo $total-$warmup_iters_time | bc )
    avg=$(echo "scale=4; $total / $count" | bc)
    avg=$( printf %.04f $avg )
    echo $avg
}

function pip_install {
    pattern_with_ver=$1
    pattern=$(echo $pattern_with_ver | cut -d"=" -f1)
    pip list 2>/dev/null | grep -E "^$pattern " 2>&1 >/dev/null; (($?==0)) || pip install $pattern_with_ver;
}

function run_bench_stocktf {
    pushd . >/dev/null
    cd ${LOCALSTORE}/demo
    TMPFILE=${WORKDIR}/tmp_output$$
    ./run_infer.sh ${MODEL} ${IMGFILE} $NUM_ITER "tf" $device 2>&1 > ${TMPFILE}
    ret_code=$?
    if (( $ret_code == 0 )); then
        echo
        echo "Stock Tensorflow: Checking inference result (warmups=$WARMUP_ITERS) ..."
        ret_code=1
        INFER_PATTERN=$( echo $INFER_PATTERN | sed -e 's/"/\\\\"/g' )
        grep "${INFER_PATTERN}" ${TMPFILE} >/dev/null && echo "TEST PASSED" && ret_code=0
        print_infer_times $NUM_ITER $WARMUP_ITERS "${TMPFILE}"
        INFER_TIME_STOCKTF=$INFER_TIME
    fi
    echo
    rm ${TMPFILE}
    popd >/dev/null
}

function run_bench_stockov {
    pushd . >/dev/null
    VENVTMP="$WORKDIR/venv_temp_stockov" # to ensure no side-effects of any pip installs
    [ -d $VENVTMP ] && rm -rf $VENVTMP
    virtualenv -p python3 $VENVTMP
    source $VENVTMP/bin/activate
    pip_install opencv-python
    pip_install openvino

    cd ${LOCALSTORE}/demo
    TMPFILE=${WORKDIR}/tmp_output$$
    pythonlib=$(echo $(dirname $(which python3))/../lib)
    echo LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$pythonlib ./run_ov_infer.sh ${MODEL} ${IMGFILE} $NUM_ITER $device
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$pythonlib \
        ./run_ov_infer.sh ${MODEL} ${IMGFILE} $NUM_ITER $device 2>&1 > ${TMPFILE}
    ret_code=$?
    if (( $ret_code == 0 )); then
        echo
        echo "Stock OpenVINO: Checking inference result (warmups=$WARMUP_ITERS) ..."
        print_infer_times $NUM_ITER $WARMUP_ITERS "${TMPFILE}"
        INFER_TIME_STOCKOV=$INFER_TIME
    fi
    echo
    deactivate # venv
    rm ${TMPFILE}
    rm -rf $VENVTMP
    popd >/dev/null
}

################################################################################
################################################################################

pip_install Pillow
if [ "${BENCHMARK}" == "YES" ]; then
    # For Stock-OV mo_tf.py called within get_model_repo
    pip_install networkx
    pip_install defusedxml
    pip_install test-generator==0.1.1
fi

cd ${LOCALSTORE_PREFIX} || exit 1
get_model_repo

IMGFILE="${LOCALSTORE}/demo/images/${IMAGE}"
if [ ! -f "${IMGFILE}" ]; then echo "Cannot find image ${IMGFILE} !"; exit 1; fi
device=${NGRAPH_TF_BACKEND:-"CPU"}

if [ "${BENCHMARK}" == "YES" ]; then
    NUM_ITER=105
    WARMUP_ITERS=5
    export NGRAPH_TF_VLOG_LEVEL=-1
else
    NUM_ITER=20
    WARMUP_ITERS=1
    [ -z "$NGRAPH_TF_LOG_PLACEMENT" ] && export NGRAPH_TF_LOG_PLACEMENT=1
    [ -z "$NGRAPH_TF_VLOG_LEVEL" ] && export NGRAPH_TF_VLOG_LEVEL=-1
fi

cd ${LOCALSTORE}/demo
TMPFILE=${WORKDIR}/tmp_output$$
INFER_TIME_TFOV="?"
./run_infer.sh ${MODEL} ${IMGFILE} $NUM_ITER "ngtf" $device 2>&1 > ${TMPFILE}
ret_code=$?
if (( $ret_code == 0 )); then
    echo
    echo "TF-OV-Bridge: Checking inference result (warmups=$WARMUP_ITERS) ..."
    ret_code=1
    INFER_PATTERN=$( echo $INFER_PATTERN | sed -e 's/"/\\\\"/g' )
    grep "${INFER_PATTERN}" ${TMPFILE} >/dev/null && echo "TEST PASSED" && ret_code=0
    print_infer_times $NUM_ITER $WARMUP_ITERS "${TMPFILE}"
    INFER_TIME_TFOV=$INFER_TIME
fi
echo
grep -oP "^NGTF_SUMMARY: (Number|Nodes|Size).*" ${TMPFILE}
rm ${TMPFILE}

if [ "${BENCHMARK}" == "YES" ]; then
    INFER_TIME_STOCKTF="?"; run_bench_stocktf
    INFER_TIME_STOCKOV="?"; run_bench_stockov
fi

if [ "${BUILDKITE}" == "true" ]; then
    if [ "${ret_code}" == "0" ]; then
        if [ "${BENCHMARK}" == "YES" ]; then
            echo -e "--- ... result: \033[33mpassed\033[0m :white_check_mark: Stock-TF ${INFER_TIME_STOCKTF}, Stock-OV ${INFER_TIME_STOCKOV}, TFOV ${INFER_TIME_TFOV}"
        else
            echo -e "--- ... result: \033[33mpassed\033[0m :white_check_mark: ${INFER_TIME_TFOV}"
        fi
    else
        echo -e "--- ... result: \033[33mfailed\033[0m :x:"
    fi
fi

exit $((ret_code))
