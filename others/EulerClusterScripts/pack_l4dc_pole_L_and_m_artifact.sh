#!/usr/bin/env bash
# Pack or extract the L4DC 2025 pole-length / pole-mass recordings as a Release artifact.
# Does not include the GRU (that stays in git).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
REC="$ROOT/SI_Toolkit_ASF/Experiments/Experiment_16_11_2024_pole_L_and_m_informed/Recordings"
OUT="${2:-$ROOT/Experiment_16_11_2024_pole_L_and_m_informed_recordings.tar.zst}"

usage() {
  echo "Usage:"
  echo "  $0                  # pack Recordings/ -> tar.zst in repo root"
  echo "  $0 --extract FILE   # unpack FILE into Recordings/"
  exit 1
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
fi

if [[ "${1:-}" == "--extract" ]]; then
  ARCHIVE="${2:-}"
  [[ -n "$ARCHIVE" && -f "$ARCHIVE" ]] || usage
  mkdir -p "$REC"
  if command -v zstd >/dev/null 2>&1; then
    tar --zstd -xf "$ARCHIVE" -C "$(dirname "$REC")"
  else
    # GitHub may serve .tar.zst; fall back if the file is actually gzip/plain tar.
    tar -xf "$ARCHIVE" -C "$(dirname "$REC")"
  fi
  echo "Extracted into $REC"
  exit 0
fi

[[ -d "$REC/Train" ]] || { echo "Missing $REC/Train — nothing to pack."; exit 1; }

if command -v zstd >/dev/null 2>&1; then
  COPYFILE_DISABLE=1 tar --zstd --exclude='.DS_Store' -cf "$OUT" -C "$(dirname "$REC")" Recordings
else
  OUT="${OUT%.tar.zst}.tar.gz"
  COPYFILE_DISABLE=1 tar --exclude='.DS_Store' -czf "$OUT" -C "$(dirname "$REC")" Recordings
fi
echo "Wrote $OUT"
ls -lh "$OUT"
