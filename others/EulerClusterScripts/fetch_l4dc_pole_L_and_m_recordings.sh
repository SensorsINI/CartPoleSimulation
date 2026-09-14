#!/usr/bin/env bash
# Fetch and extract the L4DC 2025 pole-length / pole-mass recordings.
# Prefers a local archive; otherwise downloads the CartPoleSimulation GitHub Release.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
REC="$ROOT/SI_Toolkit_ASF/Experiments/Experiment_16_11_2024_pole_L_and_m_informed/Recordings"
NAME="Experiment_16_11_2024_pole_L_and_m_informed_recordings.tar.zst"
DEFAULT_LOCAL="$ROOT/$NAME"
RELEASE_TAG="l4dc25-pole-L-and-m-data"
RELEASE_URL="https://github.com/SensorsINI/CartPoleSimulation/releases/download/${RELEASE_TAG}/${NAME}"
EXPECTED_SHA256="c6613c058a0fbe6d7f111d691113861f2fa7f5813ba9dc79b6d79b8e20ea5787"

usage() {
  echo "Usage:"
  echo "  $0                  # use ./$NAME if present, else download the Release"
  echo "  $0 /path/to/$NAME   # extract this file"
  exit 1
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
fi

ARCHIVE="${1:-}"
if [[ -z "$ARCHIVE" ]]; then
  if [[ -f "$DEFAULT_LOCAL" ]]; then
    ARCHIVE="$DEFAULT_LOCAL"
  else
    ARCHIVE="$DEFAULT_LOCAL"
    echo "Downloading $RELEASE_URL"
    curl -fL --retry 3 --retry-delay 2 -o "$ARCHIVE" "$RELEASE_URL"
  fi
fi
[[ -f "$ARCHIVE" ]] || { echo "Archive not found: $ARCHIVE"; exit 1; }

if command -v shasum >/dev/null 2>&1; then
  GOT="$(shasum -a 256 "$ARCHIVE" | awk '{print $1}')"
elif command -v sha256sum >/dev/null 2>&1; then
  GOT="$(sha256sum "$ARCHIVE" | awk '{print $1}')"
else
  GOT=""
fi
if [[ -n "$GOT" && "$GOT" != "$EXPECTED_SHA256" ]]; then
  echo "SHA256 mismatch for $ARCHIVE"
  echo "  expected $EXPECTED_SHA256"
  echo "  got      $GOT"
  exit 1
fi

mkdir -p "$REC"
if command -v zstd >/dev/null 2>&1; then
  tar --zstd -xf "$ARCHIVE" -C "$(dirname "$REC")"
else
  tar -xf "$ARCHIVE" -C "$(dirname "$REC")"
fi

train_n="$(find "$REC/Train" -name '*.csv' | wc -l | tr -d ' ')"
val_n="$(find "$REC/Validate" -name '*.csv' | wc -l | tr -d ' ')"
test_n="$(find "$REC/Test" -name '*.csv' | wc -l | tr -d ' ')"
echo "Extracted into $REC"
echo "CSV counts: Train=$train_n Validate=$val_n Test=$test_n (expect 172 / 10 / 10)"
if [[ "$train_n" != "172" || "$val_n" != "10" || "$test_n" != "10" ]]; then
  echo "Unexpected file counts." >&2
  exit 1
fi
