"""GPU Calinski-Harabasz, Davies-Bouldin and Dunn indices, matching bblean.metrics definitions.

Follows the BitBIRCH paper protocol: each index is computed after keeping only clusters with
>= min_size members, for min_size in (1, 2, 3, 5, 10). Dunn needs iSIM, which is undefined for
singletons, so it always uses clusters with >= max(min_size, 2) members. Used by scripts/metrics/moleculeace_metrics.py.

usage:
  paper_indices.py validate      (compare with bblean.metrics on one MoleculeACE set)
  paper_indices.py FINGERPRINTS.npy OUTPUT.csv ARCHIVE.npz [ARCHIVE.npz ...] [--skip REGEX]
"""

import sys
from pathlib import Path

import numpy as np
import torch

MIN_SIZES = (1, 2, 3, 5, 10)
BITS = 2048
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = False
SHIFTS = torch.arange(8, device=DEVICE, dtype=torch.uint8)


def unpack(packed_gpu):
    """(m, 256) uint8 -> (m, 2048) uint8 bits (bit order irrelevant: all measures are bit-permutation invariant)."""
    return ((packed_gpu[:, :, None] >> SHIFTS) & 1).reshape(len(packed_gpu), -1)


def pack_bits(bits_gpu):
    weights = (1 << SHIFTS).to(torch.uint8)
    return (bits_gpu.reshape(len(bits_gpu), -1, 8) * weights).sum(-1, dtype=torch.uint8)


def cluster_summaries(fps, labels):
    """Per-cluster linear sums (host int32 K x 2048), sizes, packed majority centroids, and per-cluster
    sums of member distances d and d^2 to their own centroid (d = 1 - Tanimoto)."""
    order = np.argsort(labels, kind="stable")
    sizes = np.bincount(labels)
    k = len(sizes)
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    sums = np.empty((k, BITS), dtype=np.int32)
    centroids = np.empty((k, BITS // 8), dtype=np.uint8)
    d_sum = np.zeros(k)
    d_sq = np.zeros(k)
    begin = 0
    while begin < k:
        end = int(np.searchsorted(offsets, offsets[begin] + 65_536, side="right")) - 1
        end = max(end, begin + 1)
        members = order[offsets[begin] : offsets[end]]
        bits = unpack(torch.from_numpy(fps[members]).to(DEVICE))
        local = torch.from_numpy(np.repeat(np.arange(end - begin), sizes[begin:end])).to(DEVICE)
        block = torch.zeros((end - begin, BITS), dtype=torch.int32, device=DEVICE)
        block.index_add_(0, local, bits.to(torch.int32))
        n = torch.from_numpy(sizes[begin:end]).to(DEVICE)
        cbits = torch.where(n[:, None] <= 1, block.clamp(max=1), (2 * block >= n[:, None]).to(torch.int32)).to(torch.uint8)
        own = cbits[local]
        inter = (bits & own).sum(1, dtype=torch.float32)
        union = (bits | own).sum(1, dtype=torch.float32)
        d = 1 - torch.where(union > 0, inter / union.clamp(min=1), torch.ones_like(inter))
        d64 = d.double()
        ds = torch.zeros(end - begin, dtype=torch.float64, device=DEVICE).index_add_(0, local, d64)
        dq = torch.zeros(end - begin, dtype=torch.float64, device=DEVICE).index_add_(0, local, d64 * d64)
        sums[begin:end] = block.cpu().numpy()
        centroids[begin:end] = pack_bits(cbits).cpu().numpy()
        d_sum[begin:end] = ds.cpu().numpy()
        d_sq[begin:end] = dq.cpu().numpy()
        begin = end
    return sums, sizes, centroids, d_sum, d_sq


def tanimoto_rows(a_bits, b_bits):
    a = a_bits.half()
    b = b_bits.half()
    inter = (a @ b.T).float()
    union = a.float().sum(1)[:, None] + b.float().sum(1)[None, :] - inter
    return torch.where(union > 0, inter / union.clamp(min=1), torch.ones_like(inter))


def chi(sums, sizes, centroids, d_sq, keep):
    n_total = sizes[keep].sum()
    k = int(keep.sum())
    if k <= 1:
        return 0.0
    total = torch.from_numpy(sums[keep].sum(0, dtype=np.int64)).to(DEVICE)
    g = (2 * total >= n_total).to(torch.uint8)[None, :]
    bcss = 0.0
    kept = np.flatnonzero(keep)
    for start in range(0, k, 65_536):
        rows = kept[start : start + 65_536]
        t = tanimoto_rows(unpack(torch.from_numpy(centroids[rows]).to(DEVICE)), g)[:, 0].double()
        bcss += float((torch.from_numpy(sizes[rows]).to(DEVICE) * (1 - t) ** 2).sum())
    wcss = d_sq[keep].sum()
    return bcss * (n_total - k) / (wcss * (k - 1))


def dbi(sizes, centroids, d_sum, keep, block=8192):
    kept = np.flatnonzero(keep)
    k = len(kept)
    s = torch.from_numpy(d_sum[kept] / sizes[kept]).float().to(DEVICE)
    row_max = torch.zeros(k, device=DEVICE)
    zero_pairs = 0
    for i in range(0, k, block):
        a = unpack(torch.from_numpy(centroids[kept[i : i + block]]).to(DEVICE))
        for j in range(i, k, block):
            b = a if j == i else unpack(torch.from_numpy(centroids[kept[j : j + block]]).to(DEVICE))
            m = 1 - tanimoto_rows(a, b)
            if j == i:
                m.fill_diagonal_(float("inf"))
            zero_pairs += int((m == 0).sum()) * (1 if j == i else 2)
            ratio = (s[i : i + block, None] + s[None, j : j + block]) / m.clamp(min=1e-9)
            row_max[i : i + block] = torch.maximum(row_max[i : i + block], ratio.max(1).values)
            if j != i:
                row_max[j : j + block] = torch.maximum(row_max[j : j + block], ratio.max(0).values)
    return float(row_max.double().sum()) / sizes[kept].sum(), zero_pairs // 2


def isim_from_parts(sum_kq, sum_kqsq, n):
    a = (sum_kqsq - sum_kq) / 2
    value = a / (a + n * sum_kq - sum_kqsq)
    return torch.where(sum_kq == 0, torch.ones_like(value), value)


def dunn(sums, sizes, keep, block=4096):
    kept = np.flatnonzero(keep & (sizes >= 2))
    k = len(kept)
    if k < 2:
        return float("nan")
    n = torch.from_numpy(sizes[kept]).double().to(DEVICE)
    linear = torch.empty(k, dtype=torch.float64, device=DEVICE)
    square = torch.empty(k, dtype=torch.float64, device=DEVICE)
    for start in range(0, k, 65_536):
        x = torch.from_numpy(sums[kept[start : start + 65_536]]).to(DEVICE).double()
        linear[start : start + 65_536] = x.sum(1)
        square[start : start + 65_536] = (x * x).sum(1)
    own = isim_from_parts(linear, square, n)
    max_d = float(own.max())
    best = -1.0  # max iSIM over merged pairs == 1 - min merged diameter
    for i in range(0, k, block):
        a = torch.from_numpy(sums[kept[i : i + block]]).to(DEVICE).float()
        for j in range(i, k, block):
            b = a if j == i else torch.from_numpy(sums[kept[j : j + block]]).to(DEVICE).float()
            cross = (a @ b.T).double()
            merged = isim_from_parts(
                linear[i : i + block, None] + linear[None, j : j + block],
                square[i : i + block, None] + square[None, j : j + block] + 2 * cross,
                n[i : i + block, None] + n[None, j : j + block],
            )
            if j == i:
                merged.fill_diagonal_(-1.0)
            best = max(best, float(merged.max()))
    return (1 - best) / max_d


def indices(fps, labels):
    sums, sizes, centroids, d_sum, d_sq = cluster_summaries(fps, labels)
    rows = []
    dunn_by_floor = {}  # Dunn ignores singletons, so min_size 1 and 2 give the same value
    for min_size in MIN_SIZES:
        keep = sizes >= min_size
        db, zero_pairs = dbi(sizes, centroids, d_sum, keep)
        floor = max(min_size, 2)
        if floor not in dunn_by_floor:
            dunn_by_floor[floor] = dunn(sums, sizes, keep)
        rows.append(
            {
                "min_size": min_size,
                "chi": chi(sums, sizes, centroids, d_sq, keep),
                "dbi": db,
                "dunn": dunn_by_floor[floor],
                "dbi_zero_distance_pairs": zero_pairs,
                "clusters_kept": int(keep.sum()),
            }
        )
        torch.cuda.empty_cache()
    return rows


def validate(set_name="CHEMBL233_Ki", threshold=0.55, bf=50):
    """Compare with bblean.metrics (jt_isim_chi, jt_dbi, jt_isim_dunn), the BitBIRCH paper's own definitions."""
    import os
    import warnings
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.common import paths as P
    os.environ.setdefault("BITBIRCH_BUILD_CPP", "1")
    sys.path.insert(0, str(P.path("BB_BBLEAN_TREE")))
    import bblean
    from bblean.metrics import jt_dbi, jt_isim_chi, jt_isim_dunn

    warnings.simplefilter("ignore")
    fps = np.load(P.DATA / "moleculeace" / set_name / "rdkit_2048.npy")
    tree = bblean.BitBirch(threshold=threshold, branching_factor=bf, merge_criterion="diameter")
    tree.fit(fps)
    labels = np.empty(len(fps), dtype=np.int64)
    for cluster, members in enumerate(tree.get_cluster_mol_ids()):
        labels[np.asarray(members)] = cluster
    gpu = {r["min_size"]: r for r in indices(fps, labels)}
    out = []
    for min_size in MIN_SIZES:
        clusters = [fps[labels == c] for c in range(labels.max() + 1) if (labels == c).sum() >= min_size]
        multi = [c for c in clusters if len(c) >= 2]
        if len(multi) < 2:
            continue
        reference = {
            "chi": jt_isim_chi(clusters, n_features=BITS),
            "dbi": jt_dbi(clusters, n_features=BITS),
            "dunn": jt_isim_dunn(multi, n_features=BITS),
        }
        for k, v in reference.items():
            out.append(dict(set=set_name, threshold=threshold, bf=bf, min_size=min_size, clusters_kept=len(clusters),
                            index=k, bblean=v, gpu=gpu[min_size][k], rel_diff=abs(v - gpu[min_size][k]) / abs(v)))
    return out


if __name__ == "__main__":
    import pandas as pd
    rows = []
    for t in (0.45, 0.55, 0.65):
        rows += validate(threshold=t)
    df = pd.DataFrame(rows)
    print(df.to_string())
    print("max relative difference:", df.rel_diff.max())
