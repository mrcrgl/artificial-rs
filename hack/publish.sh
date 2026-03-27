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
# Optional env:
#   EXPECT_TAG=v0.6.0      # require this git tag to exist
#
###############################################################################

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CRATES=(
  "crates/artificial-core"
  "crates/artificial-prompt"
  "crates/artificial-types"
  "crates/artificial-openai"
  "crates/artificial"
)

CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [[ "$CURRENT_BRANCH" != "master" && "$CURRENT_BRANCH" != "main" ]]; then
  echo "❌ Refusing to publish from branch '$CURRENT_BRANCH'. Use 'master' or 'main'."
  exit 1
fi

if [[ -n "${EXPECT_TAG:-}" ]]; then
  if ! git rev-parse --verify --quiet "refs/tags/$EXPECT_TAG" >/dev/null; then
    echo "❌ Expected tag '$EXPECT_TAG' does not exist."
    exit 1
  fi
fi

# Ensure repository is clean before publishing any crate.
if [[ -n "$(git status --porcelain)" ]]; then
  echo "❌ Working tree is not clean – aborting."
  exit 1
fi

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
  echo "→ Packaging check…"
  cargo package --locked

  # Perform the (dry-)publish
  $PUBLISH_CMD
  popd > /dev/null
  echo "⏳ Waiting for crates.io index to update…"
  sleep 20
done

echo "✅ All done!"
