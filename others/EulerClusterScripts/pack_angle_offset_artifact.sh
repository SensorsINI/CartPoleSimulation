#!/usr/bin/env bash
# Pack Experiment_angle_offset_GRU_8IN/Recordings (Train/Validate/Test CSVs only).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
REC="$ROOT/SI_Toolkit_ASF/Experiments/Experiment_angle_offset_GRU_8IN/Recordings"
OUT="${2:-$ROOT/Experiment_angle_offset_GRU_8IN_recordings.tar.zst}"
CHECKSUMS="$ROOT/SI_Toolkit_ASF/Experiments/Experiment_angle_offset_GRU_8IN/CHECKSUMS.sha256"

usage() {
  echo "Usage:"
  echo "  $0                  # pack Recordings/ to tar.zst"
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
    tar -xf "$ARCHIVE" -C "$(dirname "$REC")"
  fi
  echo "Extracted into $REC"
  exit 0
fi

[[ -d "$REC/Train" ]] || { echo "Missing $REC/Train — nothing to pack."; exit 1; }

# Refuse unexpected files (private leftovers, plots, zips).
bad="$(find "$REC" -type f ! -name 'Experiment-*.csv' | head)"
if [[ -n "$bad" ]]; then
  echo "Refusing to pack unexpected files:" >&2
  echo "$bad" >&2
  exit 1
fi

if command -v zstd >/dev/null 2>&1; then
  COPYFILE_DISABLE=1 tar --zstd --exclude='.DS_Store' -cf "$OUT" -C "$(dirname "$REC")" Recordings
else
  OUT="${OUT%.tar.zst}.tar.gz"
  COPYFILE_DISABLE=1 tar --exclude='.DS_Store' -czf "$OUT" -C "$(dirname "$REC")" Recordings
fi
echo "Wrote $OUT"
ls -lh "$OUT"

if command -v shasum >/dev/null 2>&1; then
  HASH="$(shasum -a 256 "$OUT" | awk '{print $1}')"
elif command -v sha256sum >/dev/null 2>&1; then
  HASH="$(sha256sum "$OUT" | awk '{print $1}')"
else
  HASH=""
fi
if [[ -n "$HASH" ]]; then
  mkdir -p "$(dirname "$CHECKSUMS")"
  tmp="$(mktemp)"
  if [[ -f "$CHECKSUMS" ]]; then
    grep -v 'recordings.tar' "$CHECKSUMS" > "$tmp" || true
  else
    : > "$tmp"
  fi
  echo "$HASH  $(basename "$OUT")" >> "$tmp"
  mv "$tmp" "$CHECKSUMS"
  echo "Updated $CHECKSUMS"
fi
