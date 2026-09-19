#!/usr/bin/env bash
# Clone the OpenAD reference code (MIT, (c) 2023 Toan Nguyen) at a pinned commit.
# We reference OpenAD; we do not vendor it into this repo. Our code in src/ imports
# from the cloned checkout at ./_ref_openad.
set -euo pipefail

REPO="https://github.com/Fsoft-AIC/Open-Vocabulary-Affordance-Detection-in-3D-Point-Clouds"
PINNED_COMMIT="b082265ed085455a3fdc80b0d8d95dc51351e7b4"
DEST="_ref_openad"

if [ -d "$DEST/.git" ]; then
  echo "'$DEST' already exists; leaving it untouched."
  exit 0
fi

echo "Cloning OpenAD into '$DEST' ..."
git clone "$REPO" "$DEST"
git -C "$DEST" checkout "$PINNED_COMMIT"
echo "Done. OpenAD checked out at $PINNED_COMMIT."
