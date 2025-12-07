#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Publish all Artificial crates to crates.io in the correct order.
#
# 1. artificial-core    – no intra-workspace dependencies
# 2. artificial-prompt  – depends on artificial-core
# 3. artificial-types   – depends on artificial-core, artificial-prompt
# 4. artificial-openai  – depends on artificial-core
# 5. artificial         – umbrella crate re-exporting everything else
#
# The script:
#   ● aborts on any error
#   ● uses `--locked` so the Cargo.lock must match Cargo.toml
#   ● adds a 20-second delay after each crate to let the index propagate
#
# Usage:
#   ./publish.sh           # live run
#   DRY_RUN=1 ./publish.sh # "cargo publish --dry-run" for all crates
#
###############################################################################

CRATES=(
  "crates/artificial-core"
  "crates/artificial-prompt"
  "crates/artificial-types"
  "crates/artificial-openai"
  "crates/artificial"
)

# Allow a dry-run mode: DRY_RUN=1 ./publish.sh
PUBLISH_CMD="cargo publish --locked"
if [[ "${DRY_RUN:-}" == "1" ]]; then
  PUBLISH_CMD+=" --dry-run"
  echo "🔍 Dry-run enabled – no crate will actually be published."
fi

for crate in "${CRATES[@]}"; do
  echo "─────────────────────────────────────────────────────────────"
  echo "📦 Publishing $crate"
  echo "─────────────────────────────────────────────────────────────"

  pushd "$crate" > /dev/null
  # Ensure version tag exists and working tree is clean.
  echo "→ Checking git status…"
  if [[ -n $(git status --porcelain) ]]; then
    echo "⚠️  Working directory not clean in $crate – aborting."
    exit 1
  fi

  # Perform the (dry-)publish
  $PUBLISH_CMD
  popd > /dev/null
  echo "⏳ Waiting for crates.io index to update…"
  sleep 20
done

echo "✅ All done!"
