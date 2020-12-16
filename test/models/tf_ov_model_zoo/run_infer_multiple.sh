#!/bin/bash
# This script is used by BuildKite CI to fetch/run multiple models from a curated model-repo for OV-IE integration project
# Invoke locally: .../run_infer_multiple.sh [ -m ./models_cpu.txt ]  [ -d .../working_dir ]

usage() { echo "Usage: $0 [-m .../manifest.txt] [-d .../working_dir]" 1>&2; exit 1; }

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
WORKDIR=`pwd`
device=${NGRAPH_TF_BACKEND:-"CPU"}
device="${device,,}" # lowercase
MODELFILENAME=models_${device}.txt
# read models & params from manifest
MANIFEST=${SCRIPT_DIR}/${MODELFILENAME}

while getopts “:m:d:h” opt; do
  case $opt in
    m) MANIFEST=${OPTARG} ;;
    d) WORKDIR=${OPTARG} ;;
    h) usage ;;
    *) usage ;;
  esac
done
shift $((OPTIND-1))

[ -f "$MANIFEST" ] || ( echo "Manifest not found: $MANIFEST !"; exit 1 )
MANIFEST="$(cd "$(dirname "$MANIFEST")"; pwd)/$(basename "$MANIFEST")" # absolute path

cd ${WORKDIR} || ( echo "Not found: $WORKDIR !"; exit 1 )
echo "Dir: ${WORKDIR}"

failed_models=()
finalretcode=0
while read -r line; do
    line=$( echo $line | sed -e 's/#.*//g' )
    [ -z "$line" ] && continue
    envs=$( echo $line | grep '\[' | sed -e 's/^\s*\[\(.*\)\].*$/\1/' )
    eval envs=($envs) && declare -p envs >/dev/null # params might have quoted strings with spaces
    line=$( echo $line | sed -e 's/^.*]\s*//g' )
    [ -z "$line" ] && continue
    eval args=($line) && declare -p args >/dev/null # params might have quoted strings with spaces
    echo; echo Running model: "${args[@]}" ...
    retcode=1
    env "${envs[@]}" "${SCRIPT_DIR}/run_infer_single.sh" "${args[@]}" && retcode=0; finalretcode=$((finalretcode+retcode))
    (( $retcode == 1 )) && failed_models+=("${args[0]}")
done < "$MANIFEST"

if (( $finalretcode > 0 )); then
    echo; echo "$finalretcode model(s) testing failed!"
    echo "${failed_models[@]}"; echo
    exit 1
fi
