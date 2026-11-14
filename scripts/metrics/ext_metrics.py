"""Fast external clustering-agreement metrics (ARI, homogeneity/completeness/V-measure, AMI).

Matches sklearn 1.9 formulas (labels_true = ref, labels_pred = labels; AMI uses the
arithmetic-mean normaliser). Everything runs in torch: on the GPU when one is available,
otherwise on multithreaded CPU torch.

The expected mutual information (EMI) depends only on the cluster sizes, so it is summed
over unique (row size, column size) pairs weighted by how many (row, column) pairs share
those sizes, instead of over all R x C pairs as sklearn does.
"""
import math
import time

import numpy as np
import torch
from scipy.special import gammaln

_EPS = float(np.finfo(np.float64).eps)


def _device(device):
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def _to_codes(x, dev):
    """Labels -> contiguous int64 codes 0..K-1 on `dev`, plus K."""
    x = np.asarray(x)
    if x.dtype.kind in "iub":
        t = torch.from_numpy(np.ascontiguousarray(x).astype(np.int64, copy=False)).to(dev)
        uniq, inv = torch.unique(t, sorted=True, return_inverse=True)
        return inv, int(uniq.numel())
    _, inv = np.unique(x, return_inverse=True)  # non-integer labels: factorise on CPU
    inv = torch.from_numpy(inv.astype(np.int64).ravel()).to(dev)
    return inv, int(inv.max().item()) + 1 if inv.numel() else 0


def _entropy_from_counts(pi):
    """sklearn _entropy, given the (positive) label counts as float64 tensor."""
    if pi.numel() == 1:
        return 0.0
    pi_sum = pi.sum()
    return float(-torch.sum((pi / pi_sum) * (torch.log(pi) - math.log(float(pi_sum)))))


def _emi(a, b, n, dev, chunk=1 << 25):
    """Expected mutual information (sklearn expected_mutual_information).

    a, b: int64 row / column sums (on dev). Work is grouped by unique (a_i, b_j) values.
    """
    if a.numel() == 1 or b.numel() == 1:
        return 0.0
    ua, ca = torch.unique(a, return_counts=True)
    ub, cb = torch.unique(b, return_counts=True)
    ua_np, ub_np = ua.cpu().numpy(), ub.cpu().numpy()
    f64 = torch.float64

    # Per-size tables, computed exactly the way sklearn does (scipy gammaln, numpy log).
    def tbl(v):
        return torch.from_numpy(np.ascontiguousarray(v, dtype=np.float64)).to(dev)
    gln_a, gln_b = tbl(gammaln(ua_np + 1)), tbl(gammaln(ub_np + 1))
    gln_Na, gln_Nb = tbl(gammaln(n - ua_np + 1)), tbl(gammaln(n - ub_np + 1))
    log_a, log_b = tbl(np.log(ua_np)), tbl(np.log(ub_np))
    maxn = int(max(ua_np.max(), ub_np.max()))
    nijs = np.arange(0, maxn + 1, dtype=np.float64)
    nijs[0] = 1
    term1 = tbl(nijs / n)
    log_Nnij = tbl(np.log(n) + np.log(nijs))
    gln_Nnij = tbl(gammaln(nijs + 1) + gammaln(n + 1))
    # lgamma(k + 1) for k = 0..n (sklearn uses libc lgamma; torch CPU lgamma is libc too)
    lg = torch.lgamma(torch.arange(n + 1, dtype=f64) + 1).to(dev)

    # All unique size pairs.
    na, nb = ua.numel(), ub.numel()
    ia = torch.arange(na, device=dev).repeat_interleave(nb)
    ib = torch.arange(nb, device=dev).repeat(na)
    pa, pb = ua[ia], ub[ib]
    start = torch.clamp(pa + pb - n, min=1)
    end = torch.minimum(pa, pb)  # inclusive
    length = torch.clamp(end - start + 1, min=0)
    keep = length > 0
    ia, ib, pa, pb, start, length = ia[keep], ib[keep], pa[keep], pb[keep], start[keep], length[keep]
    weight = (ca[ia] * cb[ib]).to(f64)
    # sklearn summation order: (((gln_a + gln_b) + gln_Na) + gln_Nb)
    base = ((gln_a[ia] + gln_b[ib]) + gln_Na[ia]) + gln_Nb[ib]
    la, lb = log_a[ia], log_b[ib]
    cum = torch.cumsum(length, 0)
    offs = cum - length
    total = int(cum[-1].item()) if cum.numel() else 0

    emi = torch.zeros((), dtype=f64, device=dev)
    for k0 in range(0, total, chunk):
        k = torch.arange(k0, min(k0 + chunk, total), device=dev, dtype=torch.int64)
        p = torch.searchsorted(cum, k, right=True)
        nij = start[p] + (k - offs[p])
        A, B = pa[p], pb[p]
        term2 = (log_Nnij[nij] - la[p]) - lb[p]
        gln = (((base[p] - gln_Nnij[nij]) - lg[A - nij]) - lg[B - nij]) - lg[n - A - B + nij]
        emi += torch.sum(term1[nij] * term2 * torch.exp(gln) * weight[p])
    return float(emi)


def external_metrics(labels, ref, subset=None, device=None, timings=False):
    """Agreement between a clustering `labels` and a reference labelling `ref`.

    subset: optional boolean mask or index array; both labellings are restricted to it.
    Returns dict with ari, homogeneity, completeness, v_measure, ami, mi, emi, n,
    n_clusters, n_ref (and t_* seconds if timings=True).
    """
    labels, ref = np.asarray(labels).ravel(), np.asarray(ref).ravel()
    if labels.shape != ref.shape:
        raise ValueError("labels and ref must have the same length")
    if subset is not None:
        subset = np.asarray(subset)
        if subset.dtype == bool:
            if subset.shape != labels.shape:
                raise ValueError("boolean subset must match label length")
        labels, ref = labels[subset], ref[subset]
    dev = _device(device)
    n = int(labels.shape[0])
    t = {}
    tic = time.perf_counter()
    if n == 0:
        out = dict(ari=1.0, homogeneity=1.0, completeness=1.0, v_measure=1.0, ami=1.0,
                   mi=0.0, emi=0.0, n=0, n_clusters=0, n_ref=0)
        return out

    # --- contingency table (rows = ref classes, columns = clusters) ---
    ci, R = _to_codes(ref, dev)
    cj, C = _to_codes(labels, dev)
    key = ci * C + cj
    ukey, nij = torch.unique(key, sorted=True, return_counts=True)
    ri, cjj = ukey // C, ukey % C
    a = torch.bincount(ci, minlength=R)  # ref class sizes
    b = torch.bincount(cj, minlength=C)  # cluster sizes
    _sync(dev)
    t["t_contingency"] = time.perf_counter() - tic

    # --- ARI (sklearn pair_confusion_matrix, exact integers) ---
    tic = time.perf_counter()
    ss = int(torch.sum(nij * nij))
    s_b = int(torch.sum(nij * b[cjj]))  # contingency.dot(n_k).sum()
    s_a = int(torch.sum(nij * a[ri]))
    tp = ss - n
    fp = s_b - ss
    fn = s_a - ss
    tn = n * n - fp - fn - ss
    if fn == 0 and fp == 0:
        ari = 1.0
    else:
        ari = 2.0 * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
    _sync(dev)
    t["t_ari"] = time.perf_counter() - tic

    # --- entropies, MI, homogeneity / completeness / V-measure ---
    tic = time.perf_counter()
    h_true = _entropy_from_counts(a.to(torch.float64))
    h_pred = _entropy_from_counts(b.to(torch.float64))
    if R == 1 or C == 1:
        mi = 0.0
    else:
        nz = nij.to(torch.float64)
        cnm = nz / n
        outer = (a[ri] * b[cjj]).to(torch.float64)
        log_outer = (-torch.log(outer) + math.log(n)) + math.log(n)
        mi_t = cnm * (torch.log(nz) - math.log(n)) + cnm * log_outer
        mi_t = torch.where(mi_t.abs() < _EPS, torch.zeros_like(mi_t), mi_t)
        mi = max(float(mi_t.sum()), 0.0)
    hom = mi / h_true if h_true else 1.0
    com = mi / h_pred if h_pred else 1.0
    v = 0.0 if hom + com == 0.0 else 2.0 * hom * com / (hom + com)
    _sync(dev)
    t["t_vmeasure"] = time.perf_counter() - tic

    # --- AMI ---
    tic = time.perf_counter()
    if R == C == 1:
        ami, emi = 1.0, 0.0
    elif R == 1 or C == 1:
        ami, emi = 0.0, 0.0
    else:
        emi = _emi(a, b, n, dev)
        den = float(np.mean([h_true, h_pred])) - emi
        den = min(den, -_EPS) if den < 0 else max(den, _EPS)
        num = mi - emi
        num = min(num, -_EPS) if num < 0 else max(num, _EPS)
        ami = num / den
    _sync(dev)
    t["t_ami"] = time.perf_counter() - tic

    out = dict(ari=float(ari), homogeneity=float(hom), completeness=float(com),
               v_measure=float(v), ami=float(ami), mi=float(mi), emi=float(emi),
               n=n, n_clusters=C, n_ref=R)
    if timings:
        out.update(t)
    return out
