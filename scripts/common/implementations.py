"""The four implementations: one name each, used in every table, figure script and command line.

  bblean_serial       bblean serial (the `cpp-tree` build)
  bblean_split_merge  bblean split-merge (`bb multiround`); its process count P goes in a separate `processes` column
  bblean_batched      bblean batched, multithreaded CPU
  nvmolkit_batched    nvMolKit batched, GPU

Labels live under BB_OUT/labels/<library>/<folder>/k<k>/s<order>.npy. The folder names (`bblean_tree`,
`bblean_mr<P>`, `bblean_batched`, `nvmolkit_main`) are internal; this module is the only place that maps them.
"""
import re

IMPLEMENTATIONS = ("bblean_serial", "bblean_split_merge", "bblean_batched", "nvmolkit_batched")
MAIN_PROCESSES = 36  # processes (= input parts) of the bblean split-merge runs in the comparison
_FOLDER = {"bblean_serial": "bblean_tree", "bblean_batched": "bblean_batched", "nvmolkit_batched": "nvmolkit_main"}


def label_folder(impl, processes=MAIN_PROCESSES):
    """Label folder under BB_OUT/labels/<library>/ of an implementation (split-merge: of its process count)."""
    if impl == "bblean_split_merge":
        return f"bblean_mr{processes}"
    return _FOLDER[impl]


def from_folder(name):
    """(implementation, processes) of a label folder; processes is None except for split-merge. None for folders
    that are not one of the four implementations at the standard settings."""
    for impl, folder in _FOLDER.items():
        if name == folder:
            return impl, None
    m = re.fullmatch(r"bblean_mr(\d+)", name)
    return ("bblean_split_merge", int(m.group(1))) if m else None
