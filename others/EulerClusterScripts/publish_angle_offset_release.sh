#!/usr/bin/env bash
# Publish the GRU-8IN-64-0 angle-offset recordings.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ARCHIVE="$ROOT/Experiment_angle_offset_GRU_8IN_recordings.tar.zst"
TAG="angle-offset-gru-8in-data"
REPO="SensorsINI/CartPoleSimulation"
BRANCH="angle-adaptive-gru-recreated"
CHECKSUMS="$ROOT/SI_Toolkit_ASF/Experiments/Experiment_angle_offset_GRU_8IN/CHECKSUMS.sha256"

[[ -f "$ARCHIVE" ]] || { echo "Missing $ARCHIVE — pack first."; exit 1; }

HASH=""
if [[ -f "$CHECKSUMS" ]]; then
  HASH="$(awk '/recordings\.tar/{print $1; exit}' "$CHECKSUMS" || true)"
fi

gh auth status -h github.com >/dev/null

if gh release view "$TAG" -R "$REPO" >/dev/null 2>&1; then
  echo "Release $TAG already exists. Uploading asset (clobber)."
  gh release upload "$TAG" "$ARCHIVE" -R "$REPO" --clobber
else
  gh release create "$TAG" "$ARCHIVE" \
    -R "$REPO" \
    --target "$BRANCH" \
    --title "Angle-offset A-NC — GRU-8IN-64-0 recordings" \
    --notes "$(cat <<EOF
Train / Validate / Test CSVs for \`Experiment_angle_offset_GRU_8IN\`.

This is the **vertical-angle offset** dataset for \`GRU-8IN-64H1-64H2-1OUT-0\`,
**not** the L4DC pole-length / pole-mass experiment.

- SHA256: \`${HASH:-run pack_angle_offset_artifact.sh}\`
- Files: Train 172, Validate 9, Test 11 (simulation, swing teacher)

\`\`\`bash
./others/EulerClusterScripts/fetch_angle_offset_recordings.sh
\`\`\`
EOF
)"
fi

echo "Release: https://github.com/${REPO}/releases/tag/${TAG}"
