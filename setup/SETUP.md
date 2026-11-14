# Setup

Builds the four implementations at pinned commits. You need Linux (aarch64 or x86_64), an NVIDIA GPU with
driver and CUDA toolkit (`nvcc`), a C++ compiler, git and curl. `install_env.sh` installs Miniforge if it's missing.
Builds take a few GB of disk; ZINC20 101M needs about 30 GB of RAM per copy.

| name | label folder | source | branch | commit | called as |
|---|---|---|---|---|---|
| bblean serial | `bblean_tree` | https://github.com/scal444/bblean (from [mqcomplab/bblean](https://github.com/mqcomplab/bblean)) | `cpp-tree` | **a2deb39** (on upstream e83709a = bblean 0.10.3) | `bblean.BitBirch(threshold=t, branching_factor=1024, merge_criterion="diameter").fit(fps)`, one thread |
| bblean split-merge | `bblean_mr<P>` | same build as serial | `cpp-tree` | a2deb39 | `bb multiround <folder of P .npy parts> --ps P -b 1024 -t t`, defaults otherwise |
| bblean batched | `bblean_batched` | https://github.com/scal444/bblean | `cpp-tree-batched` | **bdad1b3** (one commit on a2deb39) | `bblean.BitBirch(..., batch_size=4096, n_threads=T).fit(fps)` or `bb run --batch-size 4096 --threads T` |
| nvMolKit batched | `nvmolkit_main` | https://github.com/scal444/nvMolKit (from [NVIDIA-BioNeMo/nvMolKit](https://github.com/NVIDIA-BioNeMo/nvMolKit)) | `main` | **763b923** (BitBIRCH merged as 0af6a71) | `nvmolkit.clustering.bitbirch(x_gpu, t, branching_factor=1024, batch_size=4096)` |

`cpp-tree` swaps bblean's Python tree for a C++ one and gives the same clusters as bblean 0.10.3. bblean batched
is nvMolKit's batched insertion added to bblean as a multithreaded CPU fit; it gives the same clusters as nvMolKit
batched.

## 1. Environment

```bash
setup/install_env.sh                          # variables: CONDA_ROOT, ENV_NAME (default bitbirch), WORK, TORCH_INDEX
source "$HOME/miniforge3/etc/profile.d/conda.sh" && conda activate bitbirch
```

It pins Python 3.13, RDKit 2025.09.6 and NumPy 2.2.6 (other RDKit versions can change a few fingerprints) and
installs bblean 0.10.3 for thresholds and as a reference; the runs use the builds below.

- **bblean on aarch64:** there's no aarch64 wheel, and the source package quietly builds a slow pure-Python bblean
  unless `BITBIRCH_BUILD_CPP=1` is set. `install_env.sh` sets it and checks that `bblean._cpp_similarity` imports.
- **PyTorch:** compute capability 10.x GPUs need CUDA 13 wheels (the default); on older GPUs any CUDA build works
  (set `TORCH_INDEX`).

## 2. Builds

```bash
setup/fetch_and_build.sh all      # or: nvmolkit | bblean_tree | bblean_batched
```

Everything goes into `BUILD` (default `$HOME/bitbirch-build`). Put these paths in `config/paths.env`:

| folder | what | config key |
|---|---|---|
| `nvmolkit-main/`, `build-main/`, `py-main/` | nvMolKit 763b923, its CMake build, and a Python overlay with the built `.so` files | `BB_NVMOLKIT_PY` = `BUILD/py-main` |
| `bblean-cpp-tree/` | bblean `cpp-tree` a2deb39, built in place | `BB_BBLEAN_TREE` = `BUILD/bblean-cpp-tree` |
| `bblean-tree-batched/` | bblean `cpp-tree-batched` bdad1b3, built in place | `BB_BBLEAN_BATCHED` = `BUILD/bblean-tree-batched` |

The scripts put the right build first on `PYTHONPATH`, so nothing gets installed into the environment.

- nvMolKit: CMake + Ninja, `RelWithDebInfo`, `CMAKE_CUDA_ARCHITECTURES` from `nvidia-smi` (override with
  `CUDA_ARCH`), Python bindings on. About a minute.
- bblean: `BITBIRCH_BUILD_CPP=1 python setup.py build_ext --inplace -j 16`, default compiler flags. A native build
  (`BITBIRCH_BUILD_NATIVE=1`) gives the same clusters and is a few per cent faster; state it if you time with it.

## About split-merge

bblean has no threaded or multi-process option for one `BitBirch.fit`; its only parallel clustering is
`bb multiround`. It runs three rounds:

1. Each input file is clustered by its own process (`--ps P`), diameter merge, threshold t; then the largest cluster
   of each file's tree is broken up and its leaf summaries reinserted with `tolerance-diameter` (tolerance 0.05).
2. The per-file summaries are grouped 10 files to a bin and each bin is reclustered with `tolerance-diameter`.
3. One process merges everything left with `tolerance-diameter`. This round dominates the run time.

`scripts/run/run_bblean_multiround.py` shuffles the input, cuts it into P equal files and runs
`bb multiround <folder> --ps P -b 1024 -t t` with bblean's defaults otherwise. On the `cpp-tree` build its clusters
match bblean 0.10.3's. The looser merge test in the later rounds gives a coarser clustering than serial.
