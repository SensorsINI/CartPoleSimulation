#!/usr/bin/env bash
# Publish the recordings archive as CartPoleSimulation Release l4dc25-pole-L-and-m-data.
# Requires: gh auth login, and the branch already pushed.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ARCHIVE="$ROOT/Experiment_16_11_2024_pole_L_and_m_informed_recordings.tar.zst"
TAG="l4dc25-pole-L-and-m-data"
REPO="SensorsINI/CartPoleSimulation"
BRANCH="L4DC25-adaptive-neural-control-recreated"

[[ -f "$ARCHIVE" ]] || { echo "Missing $ARCHIVE — pack first."; exit 1; }

gh auth status -h github.com >/dev/null

if gh release view "$TAG" -R "$REPO" >/dev/null 2>&1; then
  echo "Release $TAG already exists. Uploading asset (clobber)."
  gh release upload "$TAG" "$ARCHIVE" -R "$REPO" --clobber
else
  gh release create "$TAG" "$ARCHIVE" \
    -R "$REPO" \
    --target "$BRANCH" \
    --title "L4DC 2025 A-NC pole length and pole mass — recordings" \
    --notes "$(cat <<'EOF'
Train / Validate / Test CSVs for `Experiment_16_11_2024_pole_L_and_m_informed`.

This is the **pole length and pole mass** A-NC dataset (L4DC 2025, PMLR 283:987–998), **not** the angle-offset experiment.

- Compressed: ~301 MB (`tar.zst`)
- SHA256: `c6613c058a0fbe6d7f111d691113861f2fa7f5813ba9dc79b6d79b8e20ea5787`
- Files: Train 172 (3.096 M rows), Validate 10 (180 k), Test 10 (180 k) at 50 Hz

Extract from a CartPoleSimulation checkout of `L4DC25-adaptive-neural-control-recreated`:

```bash
./others/EulerClusterScripts/fetch_l4dc_pole_L_and_m_recordings.sh
```
EOF
)"
fi

echo "Release: https://github.com/${REPO}/releases/tag/${TAG}"
