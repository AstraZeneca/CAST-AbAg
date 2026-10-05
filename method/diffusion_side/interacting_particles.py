"""Family ``interaction`` -- couple particles to each other, or to earlier structures.

  * ``af3_red``  AF3-ReD (OkazakiLab, doi 10.64898/2025.12.17.693105): a Gaussian repulsive bias
                 V = w * sum_b exp(-MSD / 2(sigma+noise)^2) against PREVIOUSLY COMPLETED structures,
                 added to the Euler update. The coupling is to earlier structures rather than to
                 co-evolving particles, but the force still depends on what else has been generated.
  * ``svgd``     STALE and unsupported, see the note at DIFFUSION_FAMILIES in ``diffusion_search``.

Neither method has a Feynman-Kac potential: a pairwise-RMSD repulsion is population-relative, so
there is no fixed target for it to reweight toward. Both therefore inject TRANSPORT (a coordinate
displacement) rather than a weight, and neither resamples. `af3_red` also takes no reward at all.

Plain numpy and backend-agnostic; ``models/protenix_hooks`` adapts it to the backend's tensors.
"""
from __future__ import annotations


from typing import Optional, Tuple

import numpy as np

# bias_restrict_mask compares against history via the shared superposition helper
from method.diffusion_side.base_search import _pairwise_kabsch, _pairwise_kabsch_site



def bias_restrict_mask(hist: np.ndarray, ref: np.ndarray, subset_idx: np.ndarray,
                       cut_rmsd: float) -> np.ndarray:
    """AF3-ReD ``--bias_restrict``: which previously-completed structures may still push."""
    h = np.asarray(hist, dtype=np.float64)
    r = np.asarray(ref, dtype=np.float64)
    if h.size == 0 or r.size == 0 or float(cut_rmsd) <= 0.0:
        return np.ones(len(h), dtype=bool)
    rho, _ = _pairwise_kabsch(np.concatenate([h, r]), subset_idx)
    return rho[:len(h), len(h):].min(1) <= float(cut_rmsd)


def af3_red_smooth(diffs: np.ndarray, chain_id: np.ndarray, n_smooth: int) -> np.ndarray:
    """Average per-residue difference vectors over +-n_smooth neighbours, never across a chain break."""
    d = np.asarray(diffs, dtype=np.float64)
    nb, n_res, _ = d.shape
    cid = np.asarray(chain_id).reshape(-1)
    n = int(n_smooth)

    # contiguous-segment bounds for every residue (reference gates on chain_id != 0 too)
    valid = cid != 0
    brk = np.flatnonzero(np.r_[True, (cid[1:] != cid[:-1])])
    seg_start = np.zeros(n_res, dtype=int)
    seg_start[brk] = brk
    seg_start = np.maximum.accumulate(seg_start)
    ends = np.r_[brk[1:], n_res]
    seg_end = np.repeat(ends, np.diff(np.r_[brk, n_res]))    # exclusive

    lo = np.maximum(seg_start, np.arange(n_res) - n)
    hi = np.minimum(seg_end, np.arange(n_res) + n + 1)
    cnt = np.where(valid, hi - lo, 0)

    c = np.zeros((nb, n_res + 1, 3), dtype=np.float64)
    np.cumsum(d * valid[None, :, None], axis=1, out=c[:, 1:, :])
    out = c[:, hi, :] - c[:, lo, :]
    return out / np.maximum(cnt, 1)[None, :, None] * valid[None, :, None]


def af3_red_bias_grad_batch(res_pool: np.ndarray, res_bias: np.ndarray, chain_id: np.ndarray,
                            noise_level: float, n_smooth: int = 10, sigma: float = 2.0,
                            weight: float = 100.0, bias_tmin: float = 2.0,
                            subset_idx: np.ndarray = None) -> np.ndarray:
    """Same as :func:`af3_red_bias_grad` for a whole PARTICLE POOL at once, (P, n_res, 3)."""
    X = np.asarray(res_pool, dtype=np.float64)
    B = np.asarray(res_bias, dtype=np.float64)
    P, n_res = X.shape[0], X.shape[1]
    if B is None or len(B) == 0 or float(noise_level) <= float(bias_tmin):
        return np.zeros_like(X)
    nb = len(B)
    Xc = X - X.mean(1, keepdims=True)                       # (P, n_res, 3)
    Bc = B - B.mean(1, keepdims=True)                       # (nb, n_res, 3)
    cov = np.einsum("pia,bic->pbac", Xc, Bc).reshape(P * nb, 3, 3)
    U, _, Vt = np.linalg.svd(cov)
    R = np.einsum("nij,njk->nik", Vt.transpose(0, 2, 1), U.transpose(0, 2, 1))
    fitted = np.einsum("pbia,pbac->pbic", np.broadcast_to(Bc, (P, nb, n_res, 3)),
                       R.reshape(P, nb, 3, 3))
    diffs = Xc[:, None, :, :] - fitted                      # (P, nb, n_res, 3)
    if subset_idx is not None:
        m = np.zeros(n_res, dtype=bool); m[np.asarray(subset_idx, dtype=int)] = True
        diffs = diffs * m[None, None, :, None]
        n_eff = max(int(m.sum()), 1)
    else:
        n_eff = max(n_res, 1)
    msd = (diffs ** 2).sum((2, 3)) / n_eff                  # (P, nb)
    s = float(sigma) + float(noise_level)
    bias_exp = np.exp(-msd / (2.0 * s * s))
    sm = np.stack([af3_red_smooth(diffs[i], chain_id, n_smooth) for i in range(P)])
    return -float(weight) * (bias_exp[:, :, None, None] * sm).sum(1) / (n_eff * s * s)


def af3_red_bias_grad(res_now: np.ndarray, res_bias: np.ndarray, chain_id: np.ndarray,
                      noise_level: float, n_smooth: int = 10, sigma: float = 2.0,
                      weight: float = 100.0, bias_tmin: float = 2.0,
                      subset_idx: np.ndarray = None) -> np.ndarray:
    """Single-structure AF3-ReD bias gradient, (n_res, 3). Thin wrapper over
    :func:`af3_red_bias_grad_batch` so there is exactly ONE implementation of the Kabsch superposition,
    the Gaussian potential and the chain-aware smoothing. See that function for the maths and the
    sign convention."""
    return af3_red_bias_grad_batch(np.asarray(res_now)[None], res_bias, chain_id, noise_level,
                                   n_smooth, sigma, weight, bias_tmin, subset_idx)[0]


# ---- STALE FROM HERE. `svgd_bandwidth`, `svgd_phi`, `AdaGrad`, `site_repulsion` and
# `_archive_repulsion` serve algorithm=svgd only, which is unsupported and is selected by nothing
# that ships. See the note at DIFFUSION_FAMILIES in diffusion_search.py. `af3_red` above is the
# only `interaction` algorithm in use.
def svgd_bandwidth(rho: np.ndarray, target_scale: float = 0.0) -> float:
    """RBF bandwidth by the SVGD median heuristic: h = median(rho^2)/log(P+1)."""
    if target_scale and target_scale > 0:
        return max(float(target_scale) ** 2, 1e-8)
    P = rho.shape[0]
    off = rho[~np.eye(P, dtype=bool)]
    med2 = float(np.median(off ** 2)) if off.size else 1.0
    return max(med2 / max(np.log(P + 1.0), 1e-6), 1e-8)


def svgd_phi(res_pool: np.ndarray, score_res: np.ndarray, paratope_idx: np.ndarray,
             bandwidth: float = 0.0, repulsion_only: bool = False,
             drive_only: bool = False, drive_self_norm: bool = False,
             recv_mask: np.ndarray = None) -> np.ndarray:
    """The SVGD update direction phi*(x_i), shape (P, n_res, 3)."""
    # _pairwise_kabsch is shared geometry (base_search), svgd_bandwidth is SVGD-local and now lives
    # in this module, so importing it from base_search would fail.
    idx = np.asarray(paratope_idx).astype(int)
    rho, diff = _pairwise_kabsch(res_pool, idx)          # diff[i,j] = A_i - A_j R*^T
    h = svgd_bandwidth(rho, bandwidth)      # bandwidth>0 = fixed target scale (A), 0 = median
    P, M = rho.shape[0], diff.shape[2]
    K = np.exp(-(rho ** 2) / h)                          # K_ij, reference kernel in this convention

    drive = (np.zeros_like(np.asarray(res_pool, dtype=np.float64)) if repulsion_only
             else np.einsum("ij,jma->ima", K, np.asarray(score_res, dtype=np.float64)))
    if drive_self_norm and not repulsion_only:
        mass = K.sum(axis=1).reshape(-1, 1, 1)                # sum_j K_ij, >= K_ii = 1
        drive = drive * (P / np.maximum(mass, 1e-12))         # cancels the 1/P applied below
    rep = np.zeros_like(drive)
    if not drive_only:
        K_off = K.copy()
        np.fill_diagonal(K_off, 0.0)
        rep[:, idx, :] = (2.0 / (h * M)) * np.einsum("ij,ijma->ima", K_off, diff)
    if recv_mask is not None:
        rep = rep * np.asarray(recv_mask, dtype=bool).reshape(-1, 1, 1)
    return (drive + rep) / max(P, 1)                     # the 1/P of the reference


class AdaGrad:
    """AdaGrad with momentum, exactly as the SVGD reference drives its updates."""

    def __init__(self, stepsize: float = 1e-3, alpha: float = 0.9, fudge: float = 1e-6):
        self.stepsize, self.alpha, self.fudge = float(stepsize), float(alpha), float(fudge)
        self.historical = None      # type: ignore[assignment]

    def step(self, grad: np.ndarray) -> np.ndarray:
        g = np.asarray(grad, dtype=np.float64)
        sq = g ** 2
        if self.historical is None:
            self.historical = sq
        else:
            if self.historical.shape != sq.shape:       # pool was resized: restart the accumulator
                self.historical = sq
            else:
                self.historical = self.alpha * self.historical + (1.0 - self.alpha) * sq
        return self.stepsize * g / (self.fudge + np.sqrt(self.historical))


def site_repulsion(res_pool: np.ndarray, paratope_idx: np.ndarray,
                   antigen_idx: Optional[np.ndarray] = None,
                   bandwidth: float = 0.0,
                   archive: Optional[np.ndarray] = None,
                   archive_weight: float = 1.0,
                   archive_scale_a: float = 15.0) -> Tuple[np.ndarray, np.ndarray]:
    """Kernel repulsion on the SITE axis. Returns ``(phi, rho)``, phi (P, n_res, 3), rho (P, P)."""
    X = np.asarray(res_pool, dtype=np.float64)
    pi = np.asarray(paratope_idx).astype(int)
    if antigen_idx is None or len(np.asarray(antigen_idx)) == 0:
        rho, diff = _pairwise_kabsch(X, pi)
    else:
        rho, diff = _pairwise_kabsch_site(X, pi, np.asarray(antigen_idx).astype(int))
    P, M = rho.shape[0], diff.shape[2]
    h = svgd_bandwidth(rho, bandwidth)
    K = np.exp(-(rho ** 2) / h)
    # grad_{x_i} k(x_i,x_j) = K_ij * (-2 rho_ij / h) * d rho_ij/dx_i, and d rho_ij/dx_i = diff/(M rho),
    # so the rho cancels and nothing blows up when two particles coincide.
    coef = (2.0 / h) * K                                   # (P,P)
    upd = np.einsum("ij,ijma->ima", coef, diff) / (float(P) * M)
    if archive is not None and len(archive):
        arc = _archive_repulsion(X, pi, antigen_idx, archive, h, M,
                                 archive_scale_a=float(archive_scale_a))
        w = float(np.clip(archive_weight, 0.0, 1.0))
        upd = (1.0 - w) * _unit_rms(upd) + w * _unit_rms(arc)
    phi = np.zeros_like(X)
    phi[:, pi, :] = upd
    return phi, rho


def _unit_rms(v: np.ndarray) -> np.ndarray:
    """``v`` rescaled to unit per-atom RMS norm, or left at zero if it is identically zero."""
    r = float(np.sqrt(np.mean((v ** 2).sum(-1)))) if v.size else 0.0
    return v / r if r > 1e-12 else v



def _antigen_frame(ag: np.ndarray) -> np.ndarray:
    """A CANONICAL rotation built from the antigen alone: its principal axes, sign-fixed."""
    c = ag.mean(0)
    X = ag - c
    _w, V = np.linalg.eigh(X.T @ X)          # ascending, columns are the principal axes
    V = V[:, ::-1]                            # major axis first
    ref = X[np.argmax((X ** 2).sum(1))]       # deterministic reference direction
    for k in (0, 1):
        if float(ref @ V[:, k]) < 0:
            V[:, k] = -V[:, k]
    V[:, 2] = np.cross(V[:, 0], V[:, 1])      # right-handed, so the frame is a rotation not a flip
    return V


def paratope_in_antigen_frame(res_pool: np.ndarray, paratope_idx, antigen_idx) -> np.ndarray:
    """Paratope coordinates in each particle's own ANTIGEN principal frame. (P, M, 3)."""
    X = np.asarray(res_pool, dtype=np.float64)
    pi = np.asarray(paratope_idx).astype(int)
    gi = np.asarray(antigen_idx).astype(int)
    out = np.empty((X.shape[0], len(pi), 3))
    for p in range(X.shape[0]):
        ag = X[p, gi, :]
        V = _antigen_frame(ag)
        out[p] = (X[p, pi, :] - ag.mean(0)) @ V
    return out


def _archive_repulsion(X, pi, antigen_idx, archive, h, M, eps: float = 1e-3,
                       archive_scale_a: float = 15.0):
    """Repulsion of each particle from a fixed archive of previously visited paratope placements."""
    Arc = np.asarray(archive, dtype=np.float64)            # (K, M, 3) in the archive's own frame
    if antigen_idx is None or len(np.asarray(antigen_idx)) == 0:
        A = X[:, pi, :] - X[:, pi, :].mean(1, keepdims=True)
    else:
        gi = np.asarray(antigen_idx).astype(int)
        A = X[:, pi, :] - X[:, gi, :].mean(1, keepdims=True)
    if antigen_idx is not None and len(np.asarray(antigen_idx)):
        A = paratope_in_antigen_frame(X, pi, antigen_idx)   # canonical, no alignment needed
    d = A[:, None, :, :] - Arc[None, :, :, :]              # (P,K,M,3)
    rho = np.sqrt(np.maximum((d ** 2).sum((-1, -2)) / M, 0.0))
    h_arc = 2.0 * float(archive_scale_a) ** 2
    K_ = np.exp(-(rho ** 2) / h_arc)                       # weight: 1 at coincidence, decaying
    unit = d / np.maximum(rho, eps)[:, :, None, None]      # direction only
    out = np.einsum("ik,ikma->ima", K_, unit) / float(Arc.shape[0])
    if float(np.abs(out).max()) < 1e-12:
        # silently returning zero here is how an inert term looks like a working one
        raise RuntimeError(
            f"archive repulsion underflowed to zero at scale {archive_scale_a} A: the nearest "
            f"archived placement is {float(rho.min()):.1f} A away, far outside the kernel")
    return out


