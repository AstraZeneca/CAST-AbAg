"""Family ``guidance`` -- steer each particle by a per-particle reward.

  * ``fk``   FK steering (Singhal et al. 2501.06848): reward + importance weights + resampling,
             NO gradient. Carries the Feynman-Kac machinery, weight increments, ESS, resampling
             schemes, adaptive lambda, and the umbrella grouping that keeps a sweep from collapsing.
  * ``tds``  Twisted Diffusion Sampling (Wu et al. 2306.17775, as adapted by ConforMix
             arXiv 2512.03312): the twisted proposal WITH the reward gradient and the exact
             importance-weight correction. Reuses every FK primitive above.
  * ``cast`` Our method runs jointly over K MSA-subsample conditionings, tilted by a
             site-measurable reward and resampled within (umbrella x site) cells.

``fk`` is the gradient ablation of ``tds``: same reward, same weights, same resampling, so a gap
between them is the gradient alone.

WHERE A GRADIENT LANDS IS A SEPARATE CHOICE FROM WHICH REWARD PRODUCED IT. A reward supplies dR/dx
and nothing else. :class:`GuidanceTerm` pairs it with a route (``coord`` or ``latent``) and
:func:`apply_terms` applies every block in one step, summing within a route so two latent terms cost
one vector-Jacobian product. ``site_route`` has NO ``latent`` option (see :data:`SITE_ROUTES`): a
latent is shared across samples while the pool assigns many candidate sites per particle, so a
shared latent gradient would sum contradictory site pulls.

The CDR and epitope rewards are pure geometry of the coordinates, so their gradients are CLOSED
FORM and cost no extra denoiser or confidence pass. The one approximation is the standard
straight-through step: the gradient is taken w.r.t. xhat_0 and applied to x_t, treating
d xhat_0 / d x_t as the identity.

Plain numpy and backend-agnostic; ``models/protenix_hooks`` turns these arrays into a coordinate
update on the backend's tensors.
"""
from __future__ import annotations


from typing import Optional, Tuple

from dataclasses import dataclass
from typing import Callable

import numpy as np


# --------------------------------------------------------------------- reward gradients







def ess(weights: np.ndarray) -> float:
    """Effective sample size of a (not necessarily normalised) weight vector."""
    w = np.asarray(weights, dtype=np.float64)
    s1 = w.sum()
    s2 = (w * w).sum()
    if s2 <= 0.0 or s1 <= 0.0:
        return float(len(w))
    return float((s1 * s1) / s2)



def ess_stratified(weights: np.ndarray, groups) -> np.ndarray:
    """Per-CELL effective sample size, which is the statistic that matches a stratified resample."""
    w = np.asarray(weights, dtype=np.float64)
    g = np.asarray(groups)
    return np.array([ess(w[g == gi]) for gi in np.unique(g)], dtype=np.float64)

def systematic_resample(weights: np.ndarray, u: float) -> np.ndarray:
    """Systematic resampling, lower variance than multinomial at small N (Singhal §, TDS)."""
    w = np.asarray(weights, dtype=np.float64)
    N = len(w)
    w = w / w.sum() if w.sum() > 0 else np.full(N, 1.0 / N)
    positions = (np.arange(N) + (u % 1.0)) / N
    cumsum = np.cumsum(w)
    cumsum[-1] = 1.0  # guard fp drift
    return np.searchsorted(cumsum, positions).astype(int)


def fk_log_increment(potential: str, r_prev: np.ndarray, r_cur: np.ndarray,
                     lam: float) -> np.ndarray:
    """Per-step log-potential increment log G_t for the Feynman-Kac weight update."""
    rp = np.asarray(r_prev, dtype=np.float64)
    rc = np.asarray(r_cur, dtype=np.float64)
    if potential == "difference":
        return lam * (rc - rp)
    if potential == "sum":
        return lam * rc
    raise ValueError(f"unknown FK potential {potential!r} (difference | sum)")


def normalize_log_weights(logw: np.ndarray) -> np.ndarray:
    """Stable softmax of log-weights → normalised weights (sum to 1)."""
    x = np.asarray(logw, dtype=np.float64)
    x = x - x.max()
    w = np.exp(x)
    return w / w.sum()


def fk_clamp_weights(w: np.ndarray) -> np.ndarray:
    """``w = clamp(w, 0, 1e10)`` with NaN -> 0, the guard the FK reference applies before resampling."""
    x = np.asarray(w, dtype=np.float64).copy()
    x[~np.isfinite(x)] = 0.0
    return np.clip(x, 0.0, 1e10)


def multinomial_resample(w: np.ndarray, u: float) -> np.ndarray:
    """Multinomial resampling, the scheme the FK-steering reference uses (``torch.multinomial``)."""
    p = fk_clamp_weights(w)
    tot = p.sum()
    if not np.isfinite(tot) or tot <= 0:
        return np.arange(len(p))
    rng = np.random.default_rng(int(abs(u) * 2 ** 32) & 0xFFFFFFFF)
    return rng.choice(len(p), size=len(p), replace=True, p=p / tot)


def _frac_list(v):
    """``0.25`` or ``"0.25,0.6"`` -> ``[0.25]`` / ``[0.25, 0.6]``. One value keeps the old behaviour."""
    if isinstance(v, str):
        return [float(x) for x in v.split(",") if x.strip() != ""]
    if isinstance(v, (list, tuple)):
        return [float(x) for x in v]
    return [float(v)]


def parse_resample_windows(spec_str):
    """``"0.25:0.35;0.60:0.80"`` -> ``("0.25,0.60", "0.35,0.80")``, or ``None`` if unset."""
    if not spec_str or not str(spec_str).strip():
        return None
    lo, hi = [], []
    for part in str(spec_str).split(";"):
        part = part.strip()
        if not part:
            continue
        if part.count(":") != 1:
            raise ValueError("resample_windows entry %r must be START:END" % part)
        a, b = part.split(":")
        try:
            a, b = float(a), float(b)
        except ValueError:
            raise ValueError("resample_windows entry %r has non-numeric bounds" % part)
        if not (0.0 <= a <= b <= 1.0):
            raise ValueError("resample_windows entry %r must satisfy 0 <= start <= end <= 1" % part)
        lo.append(a); hi.append(b)
    if not lo:
        raise ValueError("resample_windows %r contains no usable window; leave it empty to use the "
                         "single resample_start_frac/resample_end_frac pair" % spec_str)
    return ",".join("%g" % x for x in lo), ",".join("%g" % x for x in hi)


def in_resample_window(step_i: int, n_steps: int, start_frac, end_frac) -> bool:
    """Is this step inside ANY of the resampling windows?"""
    if n_steps <= 0:
        return True
    f = float(step_i) / float(n_steps)
    lo, hi = _frac_list(start_frac), _frac_list(end_frac)
    if len(lo) != len(hi):
        raise ValueError("resample_start_frac and resample_end_frac must have the same number of "
                         "windows, got %d and %d" % (len(lo), len(hi)))
    return any(a <= f <= b for a, b in zip(lo, hi))


def lambda_for_target_ess(r_prev: np.ndarray, r_cur: np.ndarray, potential: str,
                          n_particles: int, target_frac: float = 0.4,
                          lam_max: float = 100.0, logw_prev=None) -> float:
    """Largest lambda whose potential still leaves ESS >= ``target_frac`` * P (adaptive tempering)."""
    base = np.zeros(int(n_particles)) if logw_prev is None else np.asarray(logw_prev, dtype=np.float64)

    def _ess_at(lam):
        return ess(normalize_log_weights(base + fk_log_increment(potential, r_prev, r_cur, lam)))

    lo, hi = 0.0, float(lam_max)
    target = float(target_frac) * int(n_particles)
    if _ess_at(hi) >= target:
        return hi                                    # even the strongest tilt is safe
    if _ess_at(lo) < target:
        return lo    # already degenerate from accumulated weight alone: add no further tilt
    for _ in range(40):                              # bisection on a monotone-decreasing ESS(lambda)
        mid = 0.5 * (lo + hi)
        if _ess_at(mid) >= target:
            lo = mid
        else:
            hi = mid
    return lo


def group_ids(n_particles: int, n_groups: int) -> np.ndarray:
    """Round-robin umbrella index per particle, matching ``sweep_targets``'s assignment."""
    g = max(1, int(n_groups))
    return np.arange(int(n_particles)) % g




def stratified_resample(weights: np.ndarray, groups: np.ndarray, u: float) -> np.ndarray:
    """Systematic resampling WITHIN each umbrella group, preserving every group's size."""
    w = np.asarray(weights, dtype=np.float64)
    g = np.asarray(groups)
    out = np.arange(len(w))
    for gi in np.unique(g):
        members = np.where(g == gi)[0]
        if members.size:
            out[members] = members[systematic_resample(w[members], u)]   # normalises within the group
    return out


def kabsch_rmsd_grad(P: np.ndarray, Q: np.ndarray) -> Tuple[float, np.ndarray]:
    """Optimal-superposition RMSD between point sets ``P``, ``Q`` (both (M,3)) AND d(RMSD)/dP."""
    A = np.asarray(P, dtype=np.float64)
    B = np.asarray(Q, dtype=np.float64)
    Ac = A - A.mean(0)
    Bc = B - B.mean(0)
    U, S, Vt = np.linalg.svd(Ac.T @ Bc)                       # H = Ac^T Bc
    d = 1.0 if np.linalg.det(U @ Vt) >= 0 else -1.0           # proper rotation, not a reflection
    R = U @ np.diag([1.0, 1.0, d]) @ Vt                       # Bc R^T ~ Ac
    diff = Ac - Bc @ R.T
    M = len(A)
    rmsd = float(np.sqrt(max(float((diff ** 2).sum() / M), 0.0)))
    if rmsd <= 1e-12:
        return rmsd, np.zeros_like(A)
    g = diff / (M * rmsd)
    return rmsd, g - g.mean(0)


def scatter_token_grad_to_atoms(grad_tok: np.ndarray, atom_to_token: np.ndarray,
                                n_atom: int) -> np.ndarray:
    """Push a per-token gradient back onto atoms, the ADJOINT of ``residue_reps_from_atoms``."""
    tok = np.asarray(atom_to_token).astype(int).reshape(-1)[:n_atom]
    g = np.asarray(grad_tok, dtype=np.float64)
    cnt = np.zeros(g.shape[1], dtype=np.float64)
    np.add.at(cnt, tok, 1.0)
    cnt = np.maximum(cnt, 1.0)
    return g[:, tok, :] / cnt[tok][None, :, None]


def scatter_token_step_to_atoms(step_tok: np.ndarray, atom_to_token: np.ndarray,
                                n_atom: int) -> np.ndarray:
    """Apply a per-token displacement to atoms. This is not the adjoint of the scatter operation."""
    tok = np.asarray(atom_to_token).astype(int).reshape(-1)[:n_atom]
    return np.asarray(step_tok, dtype=np.float64)[:, tok, :]


def proposal_log_correction(delta: np.ndarray, eps: np.ndarray, s: float) -> np.ndarray:
    """log p/q̃ per particle, the importance-weight term that makes a GUIDED proposal exact."""
    d = np.asarray(delta, dtype=np.float64).reshape(len(delta), -1)
    e = np.asarray(eps, dtype=np.float64).reshape(len(delta), -1)
    s = float(s)
    if s <= 0:
        return np.zeros(len(d))
    return -((d * d).sum(1) + 2.0 * s * (e * d).sum(1)) / (2.0 * s * s)


def proposal_log_correction_torch(delta, eps, s: float):
    """On-device twin of :func:`proposal_log_correction`. Same formula, no host round-trip."""
    import torch
    P = int(delta.shape[0]) if delta.dim() == 3 else int(delta.shape[-3])
    if s <= 0:
        return torch.zeros(P, dtype=torch.float64, device=delta.device)
    d = delta.reshape(P, -1).double()
    e = eps.reshape(P, -1).double()
    s = float(s)
    return -((d * d).sum(1) + 2.0 * s * (e * d).sum(1)) / (2.0 * s * s)


def guidance_eps(sigma: float, scale: float, schedule: str = "sigma2") -> float:
    """The step coefficient ``eps_t`` multiplying the reward gradient. This is the ANNEALING schedule."""
    s = float(sigma)
    if schedule == "sigma2":
        return float(scale) * s * s / np.sqrt(1.0 + s * s)
    if schedule == "linear":
        return float(scale) * s
    raise ValueError(f"unknown guidance schedule {schedule!r} (sigma2 | linear)")


def guided_step(grad_atoms: np.ndarray, sigma: float, scale: float,
                max_norm: float = 0.0, schedule: str = "sigma2",
                max_rms_a: float = 0.0) -> np.ndarray:
    """The coordinate displacement to ADD to the proposal: ``eps_t * grad_x log G``."""
    g = np.asarray(grad_atoms, dtype=np.float64)
    step = guidance_eps(sigma, scale, schedule) * g
    cap = float(max_norm)
    if max_rms_a and float(max_rms_a) > 0:
        n_dof = int(np.asarray(step).shape[1]) * 3
        cap = float(max_rms_a) * (n_dof ** 0.5)
    if cap and cap > 0:
        n = np.linalg.norm(step.reshape(len(step), -1), axis=1)
        over = n > cap
        if over.any():
            step[over] *= (cap / n[over])[:, None, None]
    return step


def crowding_log_penalty(res_pool: np.ndarray, paratope_idx: np.ndarray, cells: np.ndarray,
                         beta: float, skip: Optional[np.ndarray] = None) -> np.ndarray:
    """Log-weight penalty for particles CROWDED by their own cell-mates, on the conformation axis."""
    from method.verifier.diversity import pairwise_superposed_rmsd

    n = int(res_pool.shape[0])
    pen = np.zeros(n, dtype=np.float64)
    if beta <= 0.0 or n < 3:
        return pen
    sub = np.asarray(res_pool)[:, np.asarray(paratope_idx), :]
    cells = np.asarray(cells)
    keep = np.ones(n, dtype=bool) if skip is None else ~np.asarray(skip, dtype=bool)
    for c in np.unique(cells[keep]):
        m = np.where((cells == c) & keep)[0]
        if m.size < 3:
            continue
        d = pairwise_superposed_rmsd(sub[m])
        off = d[~np.eye(m.size, dtype=bool)]
        h = float(np.median(off))
        if not np.isfinite(h) or h <= 1e-8:
            continue                                  # identical particles: no gradient of diversity
        k = np.exp(-(d ** 2) / (2.0 * h * h))
        np.fill_diagonal(k, 0.0)
        pen[m] = beta * k.sum(axis=1)
    return pen



ROUTES = ("coord", "latent", "off")


SITE_ROUTES = ("coord", "off")          # no latent: see DiffusionSearchSpec.site_route


def check_route(name: str, value: str) -> str:
    """Validate a route at PREFLIGHT, not mid-diffusion. An unknown route silently falling through
    to 'no gradient anywhere' is the failure this exists to stop: the arm label would still say the
    reward was applied."""
    v = (value or "coord").strip().lower()
    allowed = SITE_ROUTES if name == "site_route" else ROUTES
    if v not in allowed:
        raise ValueError("unknown guidance route %r for %s (expected one of %s)"
                         % (value, name, "|".join(allowed)))
    return v



def index_groups(ids):
    """``[(indices, value)]``, one entry per distinct id, as integer index arrays."""
    a = np.asarray(ids).ravel()
    return [(np.flatnonzero(a == v), int(v)) for v in np.unique(a)]


def permute_groups(group_ids, ancestors):
    """Carry each particle's GROUP through a resampling step: ``group_ids[ancestors]``."""
    return np.asarray(group_ids)[np.asarray(ancestors, dtype=int)]


def embedopt_rms_normalise(grad):
    """Per-token RMS normalisation, matching rs-station/embedopt exactly::"""
    import torch
    rms = torch.sqrt((grad ** 2).mean(dim=(-1, -2), keepdim=True) + 1e-20)
    return grad / rms


def embedopt_step(x_hat, deltas, dR_dx, scales, reg_scale: float = 0.0):
    """One EmbedOpt ascent step (rs-station/embedopt), on ADDITIVE deltas of the trunk embeddings."""
    import torch
    live = [d for d in deltas if d is not None]
    if not live:
        raise RuntimeError("embedopt: no delta tensors to optimise")
    if any(not d.requires_grad for d in live):
        raise RuntimeError("embedopt: a delta has requires_grad=False, so the denoiser ran outside "
                           "the autograd graph. Enable grad on that forward.")
    if not torch.is_tensor(dR_dx):
        dR_dx = torch.as_tensor(np.asarray(dR_dx), dtype=x_hat.dtype, device=x_hat.device)
    obj = (x_hat * dR_dx).sum()
    if reg_scale > 0.0:
        obj = obj - float(reg_scale) * sum((d ** 2).mean() for d in live)
    grads = iter(torch.autograd.grad(obj, tuple(live), retain_graph=True, allow_unused=False))
    sc = list(scales) or [0.0]
    with torch.no_grad():
        return [None if d is None else
                (d.detach() + embedopt_rms_normalise(next(grads))
                 * float(sc[min(k, len(sc) - 1)])).requires_grad_(True)
                for k, d in enumerate(deltas)]


@dataclass(frozen=True)
class GuidanceTerm:
    """One reward's gradient plus where it is applied."""

    name: str
    route: str
    grad_fn: Callable[[np.ndarray], np.ndarray]
    weight: float = 1.0


def sum_grads(terms, route: str, x):
    """Summed dR/dx over the terms routed to ``route``. None if that route has no active term."""
    tot = None
    for t in terms:
        if check_route(t.name, t.route) != route or float(t.weight) == 0.0:
            continue
        g = np.asarray(t.grad_fn(x), dtype=np.float64) * float(t.weight)
        tot = g if tot is None else tot + g
    return tot


def apply_terms(terms, x, sigma: float, scale: float, x_hat=None, z=None,
                max_norm: float = 0.0, schedule: str = "sigma2", max_rms_a: float = 0.0,
                latent_eta: float = 0.01, latent_reg_scale: float = 0.0):
    """Apply every block in one step. Returns ``(coord_displacement_or_None, new_z_or_z)``."""
    coord_g = sum_grads(terms, "coord", x)
    disp = None
    if coord_g is not None:
        disp = guided_step(coord_g, sigma, scale, max_norm=max_norm, schedule=schedule,
                           max_rms_a=max_rms_a)
    lat_g = sum_grads(terms, "latent", x)
    if lat_g is not None:
        if x_hat is None or z is None:
            raise RuntimeError("a term is routed to the latent but x_hat or the deltas were not "
                               "supplied, so the vector-Jacobian product cannot be formed")
        # `z` here is the list of DELTAS (embedopt optimises base+delta, not the embedding), and
        # `latent_eta` is reused as the per-tensor scale so a single knob still drives both.
        z = embedopt_step(x_hat, list(z) if isinstance(z, (list, tuple)) else [z],
                          lat_g, [latent_eta], reg_scale=latent_reg_scale)
    return disp, z

