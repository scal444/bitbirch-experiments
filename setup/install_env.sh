#!/bin/bash
# Create the conda environment used for every script in this repository (Linux, aarch64 or x86_64).
#   setup/install_env.sh
# Variables (defaults in brackets):
#   CONDA_ROOT  Miniforge install to use or create            [$HOME/miniforge3]
#   ENV_NAME    name of the environment                       [bitbirch]
#   WORK        folder for the installer and bblean source    [$HOME/bitbirch-setup]
#   TORCH_INDEX PyTorch wheel index; CUDA 13 wheels are needed for compute capability 10.x GPUs
#               [https://download.pytorch.org/whl/cu130]
# Resumable: a rerun skips the steps already done.
set -euxo pipefail
CONDA_ROOT=${CONDA_ROOT:-$HOME/miniforge3}
ENV_NAME=${ENV_NAME:-bitbirch}
WORK=${WORK:-$HOME/bitbirch-setup}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}
mkdir -p "$WORK"
if [ ! -x "$CONDA_ROOT/bin/conda" ]; then
  INSTALLER=Miniforge3-Linux-$(uname -m).sh
  curl -fsSL -o "$WORK/$INSTALLER" https://github.com/conda-forge/miniforge/releases/latest/download/$INSTALLER
  bash "$WORK/$INSTALLER" -b -p "$CONDA_ROOT"
fi
source "$CONDA_ROOT/etc/profile.d/conda.sh"
if [ ! -d "$CONDA_ROOT/envs/$ENV_NAME" ]; then
  conda create -y -n "$ENV_NAME" -c conda-forge python=3.13 \
    rdkit=2025.09.6 librdkit-dev=2025.09.6 libboost-devel libboost-python-devel eigen \
    cmake ninja numpy=2.2.6 scipy scikit-learn pandas matplotlib pytest ipython
fi
conda activate "$ENV_NAME"
pip install --index-url "$TORCH_INDEX" torch
# bblean 0.10.3 from source with its C++ extension. PyPI has no aarch64 wheel, and the source package builds a
# pure-Python bblean unless BITBIRCH_BUILD_CPP=1 is set (it still imports, but runs much slower).
pip install pybind11 setuptools-scm
pip download bblean==0.10.3 --no-deps --no-binary :all: -d "$WORK/bblean_src"
tar xzf "$WORK/bblean_src/bblean-0.10.3.tar.gz" -C "$WORK/bblean_src"
BITBIRCH_BUILD_CPP=1 pip install --no-deps --no-build-isolation "$WORK/bblean_src/bblean-0.10.3"
pip install bblean==0.10.3  # remaining dependencies
python -W error -c "import bblean._cpp_similarity"  # fails if the C++ extension was not built
python -c "import torch, rdkit, bblean, numpy; print(torch.__version__, rdkit.__version__, bblean.__version__, numpy.__version__)"
