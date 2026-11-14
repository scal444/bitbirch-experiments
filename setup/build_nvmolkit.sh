#!/bin/bash
# Build nvMolKit (any checkout) for the local GPU and install a Python overlay.
#   setup/build_nvmolkit.sh <checkout> <build_dir> <py_overlay_dir>
# The overlay folder gets the nvmolkit/ package with the built .so files; put it on PYTHONPATH (config key
# BB_NVMOLKIT_PY). All Python modules are built (types.py imports _embedMolecules).
# Variables (defaults in brackets):
#   CONDA_ROOT  Miniforge install                         [$HOME/miniforge3]
#   ENV_NAME    conda environment from setup/install_env.sh [bitbirch]
#   CUDACXX     CUDA compiler                             [/usr/local/cuda/bin/nvcc]
#   CUDA_ARCH   CMAKE_CUDA_ARCHITECTURES                  [compute capability of GPU 0 from nvidia-smi, e.g. 103]
#   NJOBS       parallel compile jobs                     [24]
set -euxo pipefail
SRC=$(realpath "$1"); BUILD=$(realpath -m "$2"); PY=$(realpath -m "$3")
CONDA_ROOT=${CONDA_ROOT:-$HOME/miniforge3}
ENV_NAME=${ENV_NAME:-bitbirch}
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
ARCH=${CUDA_ARCH:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)}
BOOST_DIR=$(ls -d $CONDA_PREFIX/lib/cmake/Boost-* | head -1)
export CUDACXX=${CUDACXX:-/usr/local/cuda/bin/nvcc}
cmake -S "$SRC" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CUDA_ARCHITECTURES=$ARCH -DNVMOLKIT_BUILD_TESTS=OFF \
  -DNVMOLKIT_BUILD_PYTHON_BINDINGS=ON -DNVMOLKIT_BUILD_BENCHMARKS=OFF \
  -DRDKit_DIR=$CONDA_PREFIX/lib/cmake/rdkit -DBoost_DIR=$BOOST_DIR \
  -DPython_EXECUTABLE=$CONDA_PREFIX/bin/python3.13
cmake --build "$BUILD" -j ${NJOBS:-24}
rm -rf "$PY" && mkdir -p "$PY"
cp -r "$SRC/nvmolkit" "$PY/"
cp "$BUILD"/nvmolkit/*.so "$PY/nvmolkit/"
cd "$PY/.." && PYTHONPATH="$PY" python -c "import nvmolkit.clustering as c; print([n for n in dir(c) if 'birch' in n.lower()])"
