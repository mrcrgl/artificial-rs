#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Bump **all** Artificial workspace crates to the supplied version and make
# sure internal path-dependencies carry the same `version = "…"`.
#
# Usage:
#   ./hack/version.sh 0.2.1
#
# After running, commit & tag:
#   git commit -am "chore: release v0.2.1"
#   git tag v0.2.1
###############################################################################

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <new-version>" >&2
  exit 1
fi
NEW_VER="$1"

# Absolute repo root (folder where this script lives -> ../../)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# All crate Cargo.toml paths (determined statically)
CRATES=(
  "crates/artificial-core/Cargo.toml"
  "crates/artificial-prompt/Cargo.toml"
  "crates/artificial-types/Cargo.toml"
  "crates/artificial-openai/Cargo.toml"
  "crates/artificial/Cargo.toml"
)

echo "🔄  Updating package versions to $NEW_VER …"
for file in "${CRATES[@]}"; do
  toml="$ROOT_DIR/$file"

  # Replace top-level package version.
  sed -Ei "0,/^version = \".*\"/s//version = \"$NEW_VER\"/" "$toml"

  # Add or update version for in-workspace dependencies.
  for dep in artificial-core artificial-prompt artificial-types artificial-openai artificial; do
    # Skip self-dependencies.
    [[ $file == *"$dep"* ]] && continue

    # If dependency is present, patch/add the version key.
    if grep -qE "^\s*$dep\s*=" "$toml"; then
      # Case 1: already has a version key – replace it.
      if grep -qE "^\s*$dep\s*=\s*\{[^}]*version" "$toml"; then
        sed -Ei "s/^(\s*$dep\s*=\s*\{[^}]*)version\s*=\s*\"[^\"]*\"/\1version = \"$NEW_VER\"/" "$toml"
      # Case 2: only path key – append version.
      else
        sed -Ei "s/^(\s*$dep\s*=\s*\{[^}]*)\}/\1, version = \"$NEW_VER\"}/" "$toml"
      fi
    fi
  done
done

echo "✅  All crate versions bumped to $NEW_VER"
echo "   Next steps:"
echo "   1. Verify changelogs."
echo "   2. Run cargo test."
echo "   3. Commit, tag, and publish."
