#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOWNLOAD_DIR="${PROJECT_ROOT}/data/downloads"
SEQUENCE_DIR="${PROJECT_ROOT}/data/imptc_sequences"
RECORD_URL="https://zenodo.org/api/records/14811016/files"
PARALLEL_PARTS="${IMPTC_PARALLEL_PARTS:-1}"

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
  marker="${SEQUENCE_DIR}/.${file_name}.extracted"

  if [[ -f "${marker}" ]]; then
    echo "[INFO] ${file_name} already extracted."
    continue
  fi

  if [[ ! -f "${archive}" ]]; then
    echo "[INFO] Downloading ${file_name}..."
    if [[ "${PARALLEL_PARTS}" =~ ^[0-9]+$ ]] && [[ "${PARALLEL_PARTS}" -gt 1 ]]; then
      tmp_dir="${DOWNLOAD_DIR}/.${file_name}.parts"
      rm -rf "${tmp_dir}"
      mkdir -p "${tmp_dir}"
      total_size="$(python - "${url}" <<'PY'
from urllib.request import Request, urlopen
import sys

request = Request(sys.argv[1], method="HEAD")
with urlopen(request) as response:
    print(response.headers["Content-Length"])
PY
)"
      for idx in $(seq 0 $((PARALLEL_PARTS - 1))); do
        start=$((idx * total_size / PARALLEL_PARTS))
        end=$((((idx + 1) * total_size / PARALLEL_PARTS) - 1))
        if [[ "${idx}" -eq $((PARALLEL_PARTS - 1)) ]]; then
          end=$((total_size - 1))
        fi
        printf "%s %s %s %s %s\n" "${idx}" "${start}" "${end}" "${url}" "${tmp_dir}"
      done | xargs -n5 -P"${PARALLEL_PARTS}" sh -c \
        'curl -L --fail --retry 5 --retry-delay 10 -r "${2}-${3}" "${4}" -o "${5}/part_${1}"' \
        _
      : > "${archive}"
      for idx in $(seq 0 $((PARALLEL_PARTS - 1))); do
        cat "${tmp_dir}/part_${idx}" >> "${archive}"
      done
      rm -rf "${tmp_dir}"
    else
      curl -L -C - --retry 5 --retry-delay 10 --fail "${url}" -o "${archive}"
    fi
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
    echo "[INFO] Removing incomplete/corrupt archive so it can be downloaded again."
    rm -f "${archive}"
    IMPTC_PARALLEL_PARTS="${PARALLEL_PARTS}" "$0" "${file_name}"
    continue
  fi

  echo "[INFO] Extracting ${file_name}..."
  tar -xzf "${archive}" -C "${SEQUENCE_DIR}"
  touch "${marker}"
done

echo "[OK] IMPTC sequence data is ready at ${SEQUENCE_DIR}"
