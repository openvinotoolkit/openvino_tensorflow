#!/bin/bash
# This script is used by BuildKite CI to fetch/run multiple models from a curated model-repo for OV-IE integration project
# Invoke locally: .../run_infer_multiple.sh [ ./models_cpu.txt ]  [ .../working_dir ]

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# read models & params from manifest
MANIFEST=${SCRIPT_DIR}/models_cpu.txt
[ -z $1 ] || MANIFEST=$1
[ -f "$MANIFEST" ] || ( echo "Manifest not found: $MANIFEST !"; exit 1 )
MANIFEST="$(cd "$(dirname "$MANIFEST")"; pwd)/$(basename "$MANIFEST")" # absolute path

WORKDIR=`pwd`
[ -z $2 ] || WORKDIR=$2
cd ${WORKDIR} || ( echo "Not found: $WORKDIR !"; exit 1 )
echo "Dir: ${WORKDIR}"

failed_models=()
finalretcode=0
while read -r line; do
    line=$( echo $line | sed -e 's/#.*//g' )
    [ -z "$line" ] && continue
    eval args=($line) && declare -p args >/dev/null # params might have quoted strings with spaces
    echo; echo Running model: "${args[@]}" ...
    retcode=1
    "${SCRIPT_DIR}/run_infer_single.sh" "${args[@]}" && retcode=0; finalretcode=$((finalretcode+retcode))
    (( $retcode == 1 )) && failed_models+=("${args[0]}")
done < "$MANIFEST"

if (( $finalretcode > 0 )); then
    echo; echo "$finalretcode model(s) testing failed!"
    echo "${failed_models[@]}"; echo
    exit 1
fi
