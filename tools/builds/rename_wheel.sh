# Copy and rename a wheel to a new project name.
# Usage: copy_to_new_project_name <whl_path> <new_project_name>, for example
# copy_to_new_project_name test_dir/tf_nightly-1.15.0.dev20190813-cp35-cp35m-manylinux2010_x86_64.whl tf_nightly_cpu
# will create a wheel with the same tags, but new project name under the same
# directory at
# test_dir/tf_nightly_cpu-1.15.0.dev20190813-cp35-cp35m-manylinux2010_x86_64.whl
function whl_rename {
  WHL_PATH="$1"
  NEW_PROJECT_NAME="$2"

  ORIGINAL_WHL_NAME=$(basename "${WHL_PATH}")
  ORIGINAL_WHL_DIR=$(realpath "$(dirname "${WHL_PATH}")")
  ORIGINAL_PROJECT_NAME="$(echo "${ORIGINAL_WHL_NAME}" | cut -d '-' -f 1)"
  FULL_TAG="$(echo "${ORIGINAL_WHL_NAME}" | cut -d '-' -f 2-)"
  NEW_WHL_NAME="${NEW_PROJECT_NAME}-${FULL_TAG}"
  VERSION="$(echo "${FULL_TAG}" | cut -d '-' -f 1)"

  TMP_DIR="$(mktemp -d)"
  cp "${WHL_PATH}" "${TMP_DIR}"
  pushd "${TMP_DIR}"
  unzip -q "${ORIGINAL_WHL_NAME}"

  ORIGINAL_WHL_DIR_PREFIX="${ORIGINAL_PROJECT_NAME}-${VERSION}"
  NEW_WHL_DIR_PREFIX="${NEW_PROJECT_NAME}-${VERSION}"
  mv "${ORIGINAL_WHL_DIR_PREFIX}.dist-info" "${NEW_WHL_DIR_PREFIX}.dist-info"
  mv "${ORIGINAL_WHL_DIR_PREFIX}.data" "${NEW_WHL_DIR_PREFIX}.data" || echo
  sed -i.bak "s/${ORIGINAL_PROJECT_NAME}/${NEW_PROJECT_NAME}/g" "${NEW_WHL_DIR_PREFIX}.dist-info/RECORD"

  ORIGINAL_PROJECT_NAME_DASH="${ORIGINAL_PROJECT_NAME//_/-}"
  NEW_PROJECT_NAME_DASH="${NEW_PROJECT_NAME//_/-}"
  sed -i.bak "0,/${ORIGINAL_PROJECT_NAME_DASH}/s//${NEW_PROJECT_NAME_DASH}/" "${NEW_WHL_DIR_PREFIX}.dist-info/METADATA"

  rm "${ORIGINAL_WHL_NAME}"
  zip -rq "${NEW_WHL_NAME}" *
  mv "${NEW_WHL_NAME}" "${ORIGINAL_WHL_DIR}"
  popd
  rm -rf "${TMP_DIR}"
}