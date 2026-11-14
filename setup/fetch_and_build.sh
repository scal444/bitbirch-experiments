#!/bin/bash
# Fetch and build the implementations (see setup/SETUP.md): nvMolKit from upstream main, bblean at pinned commits.
#   setup/fetch_and_build.sh [nvmolkit|bblean_tree|bblean_batched|all]
# Variables (defaults in brackets):
#   BUILD       folder for the checkouts, build folders and the nvMolKit Python overlay [$HOME/bitbirch-build]
#   CONDA_ROOT  Miniforge install                                                     [$HOME/miniforge3]
#   ENV_NAME    conda environment made by setup/install_env.sh                        [bitbirch]
# nvMolKit needs the CUDA toolkit (nvcc) and an NVIDIA GPU; the bblean builds need only a C++ compiler.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
BUILD=${BUILD:-$HOME/bitbirch-build}
CONDA_ROOT=${CONDA_ROOT:-$HOME/miniforge3}
ENV_NAME=${ENV_NAME:-bitbirch}
NVMOLKIT_REMOTE=https://github.com/NVIDIA-BioNeMo/nvMolKit
NVMOLKIT_BRANCH=main           # latest commit of this branch (the example results used 763b923)
BBLEAN_REMOTE=https://github.com/scal444/bblean.git   # fork of mqcomplab/bblean
BBLEAN_TREE_COMMIT=a2deb39     # branch cpp-tree (on upstream e83709a = bblean 0.10.3)
BBLEAN_BATCHED_COMMIT=bdad1b3  # branch cpp-tree-batched (one commit on a2deb39)
mkdir -p "$BUILD"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

checkout() {  # dir remote commit-or-ref [branch to fetch]
  if [ ! -d "$1/.git" ] && [ ! -f "$1/.git" ]; then git clone "$2" "$1"; fi
  git -C "$1" remote set-url origin "$2"   # an older checkout may point elsewhere
  git -C "$1" fetch origin ${4:-} || true
  git -C "$1" checkout --detach "$3"
}

nvmolkit() {
  checkout "$BUILD/nvmolkit-main" $NVMOLKIT_REMOTE "origin/$NVMOLKIT_BRANCH" $NVMOLKIT_BRANCH
  echo "nvMolKit $NVMOLKIT_BRANCH at $(git -C "$BUILD/nvmolkit-main" rev-parse --short HEAD)"
  "$HERE/build_nvmolkit.sh" "$BUILD/nvmolkit-main" "$BUILD/build-main" "$BUILD/py-main"
}
bblean_build() {  # dir commit branch
  checkout "$1" $BBLEAN_REMOTE "$2" "$3"
  (cd "$1" && BITBIRCH_BUILD_CPP=1 python setup.py build_ext --inplace -j 16)
}
bblean_tree()    { bblean_build "$BUILD/bblean-cpp-tree" $BBLEAN_TREE_COMMIT cpp-tree; }
bblean_batched() { bblean_build "$BUILD/bblean-tree-batched" $BBLEAN_BATCHED_COMMIT cpp-tree-batched; }

case "${1:-all}" in
  all) nvmolkit; bblean_tree; bblean_batched ;;
  *) "$1" ;;
esac
