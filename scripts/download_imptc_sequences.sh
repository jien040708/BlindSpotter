#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOWNLOAD_DIR="${PROJECT_ROOT}/data/downloads"
SEQUENCE_DIR="${PROJECT_ROOT}/data/imptc_sequences"
RECORD_URL="https://zenodo.org/api/records/14811016/files"

expected_md5() {
  case "$1" in
    imptc_set_01.tar.gz) echo "a1231057d2edac6daebcb1d39bcd5f25" ;;
    imptc_set_02.tar.gz) echo "5601c69c8c965e5d93206ccab04ced6c" ;;
    imptc_set_03.tar.gz) echo "5b0de174a1fd3c9d374b8d1613fd563a" ;;
    imptc_set_04.tar.gz) echo "39029256faa5d5a57b62b43245766d29" ;;
    imptc_set_05.tar.gz) echo "bfdcbae1a2dfb293bd4f6ca3723fe8b5" ;;
    *) return 1 ;;
  esac
}

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 imptc_set_01.tar.gz [imptc_set_02.tar.gz ...]"
  exit 1
fi

mkdir -p "${DOWNLOAD_DIR}" "${SEQUENCE_DIR}"

for file_name in "$@"; do
  md5="$(expected_md5 "${file_name}" || true)"
  if [[ -z "${md5}" ]]; then
    echo "[ERROR] Unknown IMPTC chunk: ${file_name}"
    exit 1
  fi

  archive="${DOWNLOAD_DIR}/${file_name}"
  url="${RECORD_URL}/${file_name}/content"

  if [[ ! -f "${archive}" ]]; then
    echo "[INFO] Downloading ${file_name}..."
    curl -L -C - --retry 5 --retry-delay 10 --fail "${url}" -o "${archive}"
  else
    echo "[INFO] Archive already exists: ${archive}"
  fi

  actual_md5="$(python - "${archive}" <<'PY'
import hashlib
import sys

path = sys.argv[1]
h = hashlib.md5()
with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
        h.update(chunk)
print(h.hexdigest())
PY
)"

  if [[ "${actual_md5}" != "${md5}" ]]; then
    echo "[ERROR] MD5 mismatch for ${archive}"
    echo "Expected: ${md5}"
    echo "Actual:   ${actual_md5}"
    exit 1
  fi

  marker="${SEQUENCE_DIR}/.${file_name}.extracted"
  if [[ -f "${marker}" ]]; then
    echo "[INFO] ${file_name} already extracted."
  else
    echo "[INFO] Extracting ${file_name}..."
    tar -xzf "${archive}" -C "${SEQUENCE_DIR}"
    touch "${marker}"
  fi
done

echo "[OK] IMPTC sequence data is ready at ${SEQUENCE_DIR}"
