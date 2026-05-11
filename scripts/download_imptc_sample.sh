#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOWNLOAD_DIR="${PROJECT_ROOT}/data/downloads"
SAMPLE_DIR="${PROJECT_ROOT}/data/sample"
ARCHIVE="${DOWNLOAD_DIR}/imptc_samples.tar.gz"
URL="https://zenodo.org/api/records/14811016/files/imptc_samples.tar.gz/content"
EXPECTED_MD5="d2bf9579b127b74927150bd3666a034f"

mkdir -p "${DOWNLOAD_DIR}" "${SAMPLE_DIR}"

if [[ ! -f "${ARCHIVE}" ]]; then
  echo "[INFO] Downloading IMPTC sample package..."
  curl -L --retry 5 --retry-delay 10 --fail "${URL}" -o "${ARCHIVE}"
else
  echo "[INFO] Archive already exists: ${ARCHIVE}"
fi

ACTUAL_MD5="$(python - "${ARCHIVE}" <<'PY'
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

if [[ "${ACTUAL_MD5}" != "${EXPECTED_MD5}" ]]; then
  echo "[ERROR] MD5 mismatch for ${ARCHIVE}"
  echo "Expected: ${EXPECTED_MD5}"
  echo "Actual:   ${ACTUAL_MD5}"
  exit 1
fi

if find "${SAMPLE_DIR}" -mindepth 1 -maxdepth 1 -type d | grep -q .; then
  echo "[INFO] Sample data already appears extracted under ${SAMPLE_DIR}"
else
  echo "[INFO] Extracting sample package..."
  tar -xzf "${ARCHIVE}" -C "${SAMPLE_DIR}"
fi

echo "[OK] IMPTC sample is ready at ${SAMPLE_DIR}"
