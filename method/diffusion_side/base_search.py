"""SHARED primitives, the parts more than one family needs. No backend, no torch, all numpy.

A primitive used by exactly ONE family belongs in that family's module instead (``diffusion_search``
for search, ``guidance`` for fk/tds/cast, ``interacting_particles`` for af3_red). What is here:

  * geometry      ``residue_reps_from_atoms`` (atoms -> per-token reps), ``_pairwise_kabsch``
  * RMSD subsets  ``register_rmsd_subset`` / ``rmsd_subset``, which residues an RMSD is measured
                  over (paratope, all, chain). The registry raises on an unknown name.
  * rewards       ``soft_pae_interface`` (PAE-only interface fallback for the ipSAE reward),
                  ``cdr_rmsd_reward`` (paratope conformation, ConforMix-style adapting 2512.03312),
                  ``epitope_centroid_reward`` / ``_grad`` (binding-SITE CV)
  * umbrellas     ``sweep_targets`` (the per-particle d* assignment) and the multi-site
                  cross-product on it: ``epitope_sites`` reads the site count from the MAP,
                  ``umbrella_ids`` / ``epitope_site_ids`` give a particle's two coordinates,
                  ``steering_group_ids`` their joint CELL, which is the stratum a resample must stay
                  inside, and ``check_cell_budget`` refuses a grid too fine to resample within.
                  These stay here rather than with the umbrella resampling in ``guidance``, because
                  interaction methods use umbrellas without ever resampling.
  * physicality   ``count_ca_clashes``, ``plddt_window_drop``, ``physicality_mask``

Plain numpy, so it is unit-testable off-GPU. The hook supplies the per-step scores and tensors;
these functions make the reward / geometry / filtering decisions only.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

# ------------------------------------------------------------------ always-defined reward


def soft_pae_interface(pae: np.ndarray, chain_ids: Sequence, pae_cutoff: float = 10.0,
                       temp: float = 1.0) -> float:
    """Always-defined PAE-only interface reward, the fallback when hard ipSAE is UNDEFINED."""
    pae = np.asarray(pae, dtype=np.float64)
    chain = np.asarray(chain_ids)
    uniq = list(dict.fromkeys(chain.tolist()))
    if len(uniq) < 2:
        return 0.0
    best = 0.0
    for a in uniq:
        ia = np.where(chain == a)[0]
        ib = np.where(chain != a)[0]
        if ia.size == 0 or ib.size == 0:
            continue
        sub = pae[np.ix_(ia, ib)]
        soft = 1.0 / (1.0 + np.exp((sub - pae_cutoff) / temp))   # = sigmoid((cutoff − pae)/temp)
        best = max(best, float(soft.mean()))
    return best


# ------------------------------------------------------------------ SMC / FK helpers




def residue_reps_from_atoms(atom_coords: np.ndarray, atom_to_token: np.ndarray,
                            n_token: int) -> np.ndarray:
    """Per-token representative coordinate = the mean of that token's atoms."""
    a = np.asarray(atom_coords, dtype=np.float64)
    tok = np.asarray(atom_to_token).astype(int).reshape(-1)
    lead = a.shape[:-2]
    flat = a.reshape(-1, a.shape[-2], 3)
    cnt = np.bincount(tok, minlength=n_token).astype(np.float64)
    cnt = np.maximum(cnt, 1.0)
    out = np.empty((flat.shape[0], n_token, 3), dtype=np.float64)
    for b in range(flat.shape[0]):
        acc = np.zeros((n_token, 3), dtype=np.float64)
        np.add.at(acc, tok, flat[b])
        out[b] = acc / cnt[:, None]
    return out.reshape(*lead, n_token, 3)


RMSD_SUBSETS = {}


def register_rmsd_subset(name):
    """Decorator registering an RMSD subset selector: (n_token, ctx) -> index array."""
    def deco(fn):
        RMSD_SUBSETS[name] = fn
        return fn
    return deco


@register_rmsd_subset("paratope")
def _subset_paratope(n_token, ctx):
    """CDR/paratope tokens from $PROTENIX_CDR_MAP, antibody loop conformation (the default)."""
    idx = np.asarray(ctx.get("paratope_idx", []), dtype=int)
    return idx[(idx >= 0) & (idx < n_token)]


@register_rmsd_subset("all")
def _subset_all(n_token, ctx):
    """Every token, whole-complex motion. This is the subset AF3-ReD's reference uses."""
    return np.arange(int(n_token))


@register_rmsd_subset("chain")
def _subset_chain(n_token, ctx):
    """One chain, selected by ``rmsd_chain``, e.g. score only the antigen, or only the Fv."""
    cid = np.asarray(ctx.get("chain_id", []), dtype=int).reshape(-1)
    want = ctx.get("rmsd_chain")
    if cid.size != n_token or want is None or int(want) < 0:
        return np.arange(int(n_token))
    sel = np.where(cid == int(want))[0]
    if sel.size == 0:                     # a real id that matches nothing is still a dead subset
        return np.arange(int(n_token))
    return sel


def rmsd_subset(name, n_token, ctx):
    """Resolve a named subset, falling back to the paratope. Raises on an unknown name so a typo in a
    config cannot silently degrade to a different metric than the one being reported."""
    if name not in RMSD_SUBSETS:
        raise ValueError(f"unknown rmsd_subset {name!r} (have: {sorted(RMSD_SUBSETS)})")
    idx = np.asarray(RMSD_SUBSETS[name](int(n_token), ctx or {}), dtype=int)
    return idx


def epitope_sites(epitope_idx):
    """Normalise ``epitope_idx`` to a LIST of index arrays -- the candidate binding sites."""
    if epitope_idx is None:
        return []
    if isinstance(epitope_idx, (list, tuple)) and len(epitope_idx) and \
            isinstance(epitope_idx[0], (list, tuple, np.ndarray)):
        return [np.asarray(s, dtype=int) for s in epitope_idx if len(s)]
    arr = np.asarray(epitope_idx, dtype=int)
    if arr.ndim == 2:                       # already (n_site, n_res)
        return [np.asarray(s, dtype=int) for s in arr]
    return [arr] if arr.size else []


def umbrella_ids(n_particles, target_dist=None):
    """Particle -> umbrella id, read from the ACTUAL per-particle d* values rather than from a count."""
    P = int(n_particles)
    if target_dist is None:
        return np.zeros(P, dtype=int)
    t = np.asarray(target_dist, dtype=np.float64).reshape(-1)
    if t.size != P:
        return np.zeros(P, dtype=int)
    return np.unique(t, return_inverse=True)[1].astype(int)


def epitope_site_ids(n_particles, n_sites, target_dist=None):
    """Particle -> candidate site, as a FULL CROSS-PRODUCT with the d* umbrellas."""
    P, S = int(n_particles), max(int(n_sites), 1)
    if S == 1:
        return np.zeros(P, dtype=int)
    umb = umbrella_ids(P, target_dist)
    site = np.zeros(P, dtype=int)
    for u in np.unique(umb):
        m = np.where(umb == u)[0]
        site[m] = np.arange(m.size) % S
    return site


def reserve_mask(n_particles, reserve_frac=0.0, cells=None):
    """Boolean mask of the RESERVE particles: the ones that get NO reward tilting at all."""
    P = int(n_particles)
    f = float(reserve_frac)
    if P <= 0 or f <= 0.0:
        return np.zeros(max(P, 0), dtype=bool)
    if f >= 1.0:
        return np.ones(P, dtype=bool)
    stride = max(2, int(round(1.0 / f)))
    m = np.zeros(P, dtype=bool)
    if cells is None:
        m[::stride] = True                       # no cell structure to respect
        return m
    c = np.asarray(cells)
    for i, cid in enumerate(np.unique(c)):
        members = np.where(c == cid)[0]
        m[members[i % stride::stride]] = True
    return m


def repulsion_recipients(n_particles, reserve_frac=0.0, reserve_repulsion=True):
    """Boolean mask of the particles that RECEIVE the repulsion term."""
    P = int(n_particles)
    if reserve_repulsion or float(reserve_frac) <= 0.0:
        return np.ones(max(P, 0), dtype=bool)
    return ~reserve_mask(P, reserve_frac)


def base_cells(n_particles, target_dist=None, n_sites=1):
    """Joint (umbrella, site) cell id per particle, BEFORE any reserve is carved out."""
    P, S = int(n_particles), max(int(n_sites), 1)
    umb = umbrella_ids(P, target_dist)
    return umb if S == 1 else umb * S + epitope_site_ids(P, S, target_dist)


def reserve_mask_for(n_particles, reserve_frac, sweep=None, n_sites=1):
    """The reserve mask for a run, built from the sweep the sampler is actually using."""
    tg = sweep_targets(n_particles, sweep) if sweep else None
    return reserve_mask(n_particles, reserve_frac,
                        cells=base_cells(n_particles, tg, n_sites))


def subsample_ids(n_particles, n_subsamples, n_umbrellas=1, n_sites=1):
    """Particle -> MSA SUBSAMPLE, laid out so every (umbrella, site) cell holds all K subsamples."""
    K = max(1, int(n_subsamples))
    P = int(n_particles)
    if K == 1:
        return np.zeros(P, dtype=int)
    period = max(1, int(n_umbrellas)) * max(1, int(n_sites))
    return (np.arange(P) // period) % K


def check_subsample_spread(cells, subs, min_per_cell=1):
    """Raise unless every cell holds at least ``min_per_cell`` particles of EVERY subsample."""
    cells = np.asarray(cells)
    subs = np.asarray(subs)
    if cells.shape != subs.shape:
        raise ValueError("cells and subs must align: %s vs %s" % (cells.shape, subs.shape))
    want = np.unique(subs)
    if want.size < 2:
        return
    bad = []
    for c in np.unique(cells):
        m = subs[cells == c]
        counts = {int(k): int((m == k).sum()) for k in want}
        if min(counts.values()) < int(min_per_cell):
            bad.append((int(c), counts))
    if bad:
        raise ValueError(
            "subsample assignment ALIASES with the cell map: %d of %d cells lack a subsample "
            "(min %d required per cell). First offenders: %s. Use base_search.subsample_ids, "
            "not p %% K." % (len(bad), len(np.unique(cells)), min_per_cell, bad[:3]))


def steering_group_ids(n_particles, target_dist=None, n_sites=1, reserve_frac=0.0):
    """Joint (umbrella, site) CELL id per particle -- the correct stratum for resampling."""
    P = int(n_particles)
    cells = base_cells(P, target_dist, n_sites)
    res = reserve_mask(P, reserve_frac, cells=cells)   # same construction as reserve_mask_for
    if res.any():
        cells = np.asarray(cells).copy()
        cells[res] = int(cells.max()) + 1 if len(cells) else 0
    return cells


def contact_consistency(res_pool, paratope_idx, epitope_idx, cells, r_contact=10.0):
    """Per-cell mean pairwise Jaccard of the ANTIGEN CONTACT SET. Higher = a better-defined site."""
    pool = np.asarray(res_pool, dtype=np.float64)
    pi = np.asarray(paratope_idx, dtype=int)
    _sites = epitope_sites(epitope_idx)
    ei = (np.concatenate([np.asarray(x, dtype=int).ravel() for x in _sites])
          if _sites else np.asarray([], dtype=int))
    ei = np.unique(ei)
    cells = np.asarray(cells)
    if pi.size == 0 or ei.size == 0 or pool.ndim != 3:
        return {}
    n_res = pool.shape[1]
    pi = pi[pi < n_res]
    ei = ei[ei < n_res]
    if pi.size == 0 or ei.size == 0:
        return {}

    sets = []
    for k in range(len(pool)):
        d = np.linalg.norm(pool[k][ei][:, None, :] - pool[k][pi][None, :, :], axis=-1)
        hit = ei[(d.min(axis=1) <= float(r_contact))]
        sets.append(frozenset(int(x) for x in hit))

    out = {}
    for c in np.unique(cells):
        m = [i for i in np.where(cells == c)[0] if sets[i]]
        if len(m) < 2:
            continue                                  # no pair -> no evidence, not bad evidence
        js, sub = [], m[:60]                          # O(n^2); 60 members is plenty for a stable mean
        for a in range(len(sub)):
            for b in range(a + 1, len(sub)):
                x, y = sets[sub[a]], sets[sub[b]]
                js.append(len(x & y) / max(len(x | y), 1))
        if js:
            out[int(c)] = float(np.mean(js))
    return out


def reallocate_cells(cells, cell_score, min_per_cell=6, reserve=None, max_move_frac=0.25):
    """Shift resampling SLOTS from low-scoring to high-scoring cells, keeping every cell above a floor."""
    cells = np.asarray(cells).copy()
    if not cell_score:
        return cells
    res = np.zeros(len(cells), dtype=bool) if reserve is None else np.asarray(reserve, dtype=bool)
    free = ~res
    scored = [c for c in np.unique(cells[free]) if int(c) in cell_score]
    if len(scored) < 2:
        return cells
    order = sorted(scored, key=lambda c: cell_score[int(c)])
    budget = int(max(1, np.floor(float(max_move_frac) * int(free.sum()))))
    moved = 0
    lo, hi = 0, len(order) - 1
    while lo < hi and moved < budget:
        src, dst = order[lo], order[hi]
        if cell_score[int(dst)] <= cell_score[int(src)]:
            break                                     # nothing left to gain
        members = np.where((cells == src) & free)[0]
        if len(members) <= int(min_per_cell):
            lo += 1                                   # source is at the floor: take from the next-worst
            continue
        cells[members[-1]] = dst                      # one slot at a time, so the floor cannot be jumped
        moved += 1
    return cells

def cell_occupancy(n_particles, n_umbrellas, n_sites=1, reserve_frac=0.0):
    """``(tilted_per_cell, n_reserved)`` under the real assignment, not a formula."""
    U, S, P = max(int(n_umbrellas), 1), max(int(n_sites), 1), int(n_particles)
    if P <= 0:
        return np.zeros(U * S, dtype=int), 0
    tg = sweep_targets(P, np.arange(U, dtype=float)) if U > 1 else None
    cells = base_cells(P, tg, S)
    res = reserve_mask(P, reserve_frac, cells=cells)
    return np.bincount(np.asarray(cells)[~res], minlength=U * S), int(res.sum())


MIN_PER_CELL = 6          # below this a within-cell resample returns near-identity: measured at 4
                          # particles / 4 umbrellas the ancestors came back [0,1,2,3] at every event
MIN_PER_SUBSAMPLE = 3     # below this one resample can drop a latent from a cell by chance: measured
                          # at N=780 over 60 cells, 3 held up (K=4, best arm) and 2 did not (K=5)


def check_cell_budget(n_particles, n_umbrellas, n_sites=1, min_per_cell=MIN_PER_CELL,
                      reserve_frac=0.0, n_subsamples=1, min_per_subsample=MIN_PER_SUBSAMPLE):
    """``(level, message)`` for the S*U cell grid: level is ``"error"``, ``"warn"`` or ``None``."""
    U, S = max(int(n_umbrellas), 1), max(int(n_sites), 1)
    cells, P = U * S, int(n_particles)
    occ, n_res = cell_occupancy(P, U, S, reserve_frac)
    lo, hi = int(occ.min()), int(occ.max())
    tilted = P - n_res
    tag = "" if n_res == 0 else (" (%d of %d held in the reserve, leaving %d tilted)"
                                 % (n_res, P, tilted))
    if lo <= 0:
        return ("error",
                "%d sites x %d umbrellas = %d cells over %d particles%s: %d cell(s) end up with NO "
                "tilted particle, so those hypotheses are never steered. Use at most %d umbrellas, "
                "or raise n_samples to at least %d."
                % (S, U, cells, P, tag, int((occ <= 0).sum()), max(tilted // S, 1),
                   int(round(cells / max(1.0 - float(reserve_frac), 1e-9)))))
    if lo < int(min_per_cell):
        return ("warn",
                "%d sites x %d umbrellas = %d cells over %d particles%s = %d-%d tilted per cell "
                "(min %d < %d): within-cell resampling degenerates toward the identity, so the sweep "
                "may not select. Use at most %d umbrellas, or raise n_samples."
                % (S, U, cells, P, tag, lo, hi, lo, int(min_per_cell),
                   max(tilted // (S * int(min_per_cell)), 1)))
    if n_res and n_res < int(min_per_cell):
        return ("warn",
                "the reserve holds only %d of %d particles (< %d): too few to be a useful unsteered "
                "hedge, and its own within-cell resampling degenerates. Raise reserve_frac or "
                "n_samples." % (n_res, P, int(min_per_cell)))
    K = max(int(n_subsamples), 1)
    if K > 1:
        per = lo // K
        if per < 1:
            return ("error",
                    "%d particles per cell split %d ways leaves some latent with NO particle in a "
                    "cell, so the cross-subsample comparison the joint pool exists for cannot happen "
                    "there. Use at most %d umbrellas, or lower subsample_pool."
                    % (lo, K, max(tilted // (S * K), 1)))
        if per < int(min_per_subsample):
            return ("warn",
                    "%d particles per cell split %d ways = %d per latent per cell (< %d): a single "
                    "resample can eliminate a latent from a cell by drift. Measured at N=780 over "
                    "%d cells, 3 per latent held up (K=4, best arm) and 2 did not (K=5). "
                    "Use at most %d umbrellas, or lower subsample_pool."
                    % (lo, K, per, int(min_per_subsample), cells,
                       max(tilted // (S * K * int(min_per_subsample)), 1)))
        counts = np.bincount(subsample_ids(P, K, U, S), minlength=K)
        if int(counts.max()) != int(counts.min()):
            block = cells * K
            opts = [(P // block) * block, ((P // block) + 1) * block]
            fix = (" Use n_samples=%s for an exactly even split (%d per latent per cell)."
                   % (" or ".join(str(n) for n in opts), (opts[0] // cells) // K))
            return ("warn",
                    "%d particles over %d cells split %d ways is UNEVEN: %d repeats per cell do not "
                    "divide by %d, so the latents get %s particles (%.2fx imbalance). The run is "
                    "valid, but a comparison across K built on it confounds the latent count with "
                    "this imbalance.%s"
                    % (P, cells, K, P // cells, K,
                       "/".join(str(int(c)) for c in counts),
                       float(counts.max()) / max(float(counts.min()), 1.0), fix))
    return (None, None)


def _docked(pool, paratope_idx, epitope_idx, contact_gate, r_contact=10.0):
    """Per-particle "already has interface contact", or None when the gate is off."""
    if not contact_gate:
        return None
    A = np.asarray(pool, dtype=np.float64)[:, np.asarray(paratope_idx, int), :]
    B = np.asarray(pool, dtype=np.float64)[:, np.asarray(epitope_idx, int), :]
    hit = (np.linalg.norm(A[:, :, None, :] - B[:, None, :, :], axis=-1) < float(r_contact)).any(1)
    return hit.mean(1) >= float(contact_gate)      # share of epitope residues contacted

def _site_potential(dev, hinge=False, huber_delta=0.0, docked=None, floor=0.0, dist=None,
                    dev_cap=0.0):
    """Value and d/d(dev) of the site penalty, where ``dev = d - d*``. Returns ``(r, dr_ddev)``."""
    d = np.asarray(dev, dtype=np.float64)
    delta = float(huber_delta)
    if delta > 0.0:
        big = np.abs(d) > delta
        r = np.where(big, -(2.0 * delta * np.abs(d) - delta ** 2), -(d ** 2))
        g = np.where(big, -2.0 * delta * np.sign(d), -2.0 * d)
    else:
        r, g = -(d ** 2), -2.0 * d
    if dev_cap > 0.0:
        far = np.abs(d) > float(dev_cap)
        r = np.where(far, -(float(dev_cap) ** 2) if delta <= 0.0
                     else -(2.0 * delta * float(dev_cap) - delta ** 2), r)
        g = np.where(far, 0.0, g)
    if hinge:
        near = np.asarray(dev, dtype=np.float64) <= 0.0
        if docked is not None:
            near = near & np.asarray(docked, dtype=bool)   # only protect what is actually docked
        g = np.where(near, 0.0, g)
        r = np.where(near, 0.0, r)
    if floor and dist is not None:
        under = np.maximum(float(floor) - np.asarray(dist, dtype=np.float64), 0.0)
        r = r - under ** 2                     # quadratic wall below the floor
        g = g + 2.0 * under                    # dr/dd = +2*under, i.e. pushes the separation UP
    return r, g


def epitope_centroid_reward(res_pool, paratope_idx, epitope_idx, target_dist=0.0,
                            hinge=False, huber_delta=0.0, contact_gate=0.0, site_floor=0.0,
                            softmin_tau=0.0, dev_cap=0.0):
    """Binding-SITE reward on the paratope-to-epitope centroid distance."""
    pool = np.asarray(res_pool, dtype=np.float64)
    pi = np.asarray(paratope_idx, dtype=int)
    sites = epitope_sites(epitope_idx)
    if pi.size == 0 or not sites:
        return np.zeros(len(pool))
    if len(sites) > 1:
        # MULTI-SITE: each particle scores against ITS OWN candidate site. Recursing per site keeps
        # the single-site maths below as the one implementation, so the two cannot drift apart.
        tgt_all = np.broadcast_to(np.asarray(target_dist, dtype=np.float64), (len(pool),))
        gid = epitope_site_ids(len(pool), len(sites), tgt_all)
        out = np.zeros(len(pool))
        for g, ei_g in enumerate(sites):
            m = gid == g
            if m.any():
                out[m] = epitope_centroid_reward(pool[m], pi, ei_g, tgt_all[m], hinge, huber_delta, contact_gate, site_floor, 0.0, dev_cap)
        return out
    ei = np.asarray(sites[0], dtype=int)
    if ei.size == 0:
        return np.zeros(len(pool))
    d = np.linalg.norm(pool[:, pi, :].mean(1) - pool[:, ei, :].mean(1), axis=-1)
    tgt = np.broadcast_to(np.asarray(target_dist, dtype=np.float64), (len(pool),))
    return _site_potential(d - tgt, hinge, huber_delta, _docked(pool, pi, ei, contact_gate), site_floor, d, dev_cap)[0]


def epitope_centroid_grad(res_pool, paratope_idx, epitope_idx, target_dist=0.0,
                          hinge=False, huber_delta=0.0, contact_gate=0.0, site_floor=0.0,
                          softmin_tau=0.0, dev_cap=0.0, freeze_epitope=False):
    """d(epitope_centroid_reward)/d(coords), (P, n_res, 3)."""
    pool = np.asarray(res_pool, dtype=np.float64)
    pi = np.asarray(paratope_idx, dtype=int)
    sites = epitope_sites(epitope_idx)
    out = np.zeros_like(pool)
    if pi.size == 0 or not sites:
        return out
    if len(sites) > 1:
        tgt_all = np.broadcast_to(np.asarray(target_dist, dtype=np.float64), (len(pool),))
        gid = epitope_site_ids(len(pool), len(sites), tgt_all)
        for g, ei_g in enumerate(sites):
            m = gid == g
            if m.any():
                out[m] = epitope_centroid_grad(pool[m], pi, ei_g, tgt_all[m], hinge, huber_delta,
                                               contact_gate, site_floor, 0.0, dev_cap,
                                               freeze_epitope=freeze_epitope)
        return out
    ei = np.asarray(sites[0], dtype=int)
    if ei.size == 0:
        return out
    v = pool[:, pi, :].mean(1) - pool[:, ei, :].mean(1)
    dist = np.linalg.norm(v, axis=-1)
    tgt = np.broadcast_to(np.asarray(target_dist, dtype=np.float64), (len(pool),))
    safe = np.maximum(dist, 1e-8)
    coef = _site_potential(dist - tgt, hinge, huber_delta, _docked(pool, pi, ei, contact_gate), site_floor, dist, dev_cap)[1][:, None] * (v / safe[:, None])
    out[:, pi, :] = coef[:, None, :] / pi.size
    if not freeze_epitope:
        out[:, ei, :] = -coef[:, None, :] / ei.size
    return out


def contact_shell(res_pool, paratope_idx, epitope_idx, r_contact=8.0, sigma=2.0,
                  saturate_tau=0.0):
    """Smooth surrogate for the NUMBER OF INTERFACE CONTACTS, and its gradient."""
    pool = np.asarray(res_pool, dtype=np.float64)
    pi = np.asarray(paratope_idx, dtype=int)
    P = len(pool)
    out_g = np.zeros_like(pool)
    sites = epitope_sites(epitope_idx)
    if len(sites) > 1:
        raise ValueError(f"contact_shell is single-site (got {len(sites)}): dispatch in the caller")
    ei = sites[0] if sites else np.zeros(0, dtype=int)
    if pi.size == 0 or ei.size == 0:
        return np.zeros(P), out_g
    s2 = max(float(sigma), 1e-6) ** 2
    A = pool[:, pi, :]                                   # (P, Np, 3)
    B = pool[:, ei, :]                                   # (P, Ne, 3)
    d = A[:, :, None, :] - B[:, None, :, :]              # (P, Np, Ne, 3)
    r = np.linalg.norm(d, axis=-1)                       # (P, Np, Ne)
    safe = np.maximum(r, 1e-8)
    dev = r - float(r_contact)
    w = np.exp(-(dev ** 2) / (2.0 * s2))                 # the shell
    dw = -(dev / s2) * w                                 # dw/dr, shared by both aggregations
    if float(saturate_tau) > 0.0:
        tau = float(saturate_tau)

        def _lse(x, axis):
            mx = x.max(axis=axis, keepdims=True)
            ex = np.exp((x - mx) / tau)
            ssum = np.maximum(ex.sum(axis=axis, keepdims=True), 1e-30)
            return mx + tau * np.log(ssum), ex / ssum          # (value, softmax) both keepdims

        m_p, sm_p = _lse(w, 2)                                  # best epitope partner per paratope
        m_e, sm_e = _lse(w, 1)                                  # best paratope partner per epitope
        C = 0.5 * (m_p[:, :, 0].mean(axis=1) + m_e[:, 0, :].mean(axis=1))
        # dC/dw_pe: each half is a mean of log-sum-exps, so the weights are the softmaxes. Both are
        # NON-NEGATIVE, so neither half can repel -- the failure the softmax-weighted mean had.
        dm = 0.5 * (sm_p / float(pi.size) + sm_e / float(ei.size))
        coef = (dm * dw)[..., None] * (d / safe[..., None])
        out_g[:, pi, :] = coef.sum(axis=2)
        out_g[:, ei, :] = -coef.sum(axis=1)
        return C, out_g
    n_pair = float(pi.size * ei.size)
    C = w.sum(axis=(1, 2)) / n_pair
    # dC/dr = -(dev/s2) * w , dr/dA = +u , dr/dB = -u  with u = d/r
    coef = dw[..., None] * (d / safe[..., None]) / n_pair   # (P, Np, Ne, 3)
    out_g[:, pi, :] = coef.sum(axis=2)
    out_g[:, ei, :] = -coef.sum(axis=1)
    return C, out_g



def orientation_reward(res_pool, paratope_idx, epitope_idx, antibody_idx, tol_deg=0.0):
    """ORIENTATION term that is ORTHOGONAL to the site term. Returns ``(r, grad)``."""
    pool = np.asarray(res_pool, dtype=np.float64)
    P = pool.shape[0]
    pi = np.asarray(paratope_idx, dtype=int)
    ei = np.asarray(epitope_idx, dtype=int)
    ai = np.asarray(antibody_idx, dtype=int)
    g = np.zeros_like(pool)
    if pi.size == 0 or ei.size == 0 or ai.size == 0:
        return np.zeros(P), g
    c = pool[:, pi, :].mean(axis=1)                       # paratope centroid
    f = pool[:, ai, :].mean(axis=1)                       # antibody (Fv) centroid
    e = pool[:, ei, :].mean(axis=1)                       # epitope centroid
    u, v = c - f, e - f
    nu = np.linalg.norm(u, axis=1); nv = np.linalg.norm(v, axis=1)
    ok = (nu > 1e-8) & (nv > 1e-8)
    r = np.zeros(P)
    if not ok.any():
        return r, g
    cos = np.einsum("pi,pi->p", u, v) / (nu * nv)
    np.clip(cos, -1.0, 1.0, out=cos)
    dcos = np.zeros_like(cos)          # dr/dcos, chained into the centroid gradients below
    if tol_deg and float(tol_deg) > 0.0:
        tol = np.deg2rad(float(tol_deg))
        th = np.arccos(cos)
        dev = np.maximum(0.0, th - tol)
        r[ok] = -(dev[ok] ** 2)
        # dr/dtheta = -2 dev , dtheta/dcos = -1/sqrt(1-cos^2)  ->  dr/dcos = 2 dev / sqrt(1-cos^2)
        # the root vanishes only at theta = 0 (where dev = 0 anyway) and theta = pi, so clamp it
        sin = np.sqrt(np.maximum(1.0 - cos ** 2, 1e-12))
        dcos[ok] = 2.0 * dev[ok] / sin[ok]
    else:
        r[ok] = cos[ok]
        dcos[ok] = 1.0
    # d cos / du and d cos / dv, then chain through the three centroids
    du = np.zeros_like(u); dv = np.zeros_like(v)
    du[ok] = dcos[ok][:, None] * (v[ok] / (nu[ok] * nv[ok])[:, None]
                                  - cos[ok][:, None] * u[ok] / (nu[ok] ** 2)[:, None])
    dv[ok] = dcos[ok][:, None] * (u[ok] / (nu[ok] * nv[ok])[:, None]
                                  - cos[ok][:, None] * v[ok] / (nv[ok] ** 2)[:, None])
    g[:, pi, :] += (du / pi.size)[:, None, :]
    g[:, ei, :] += (dv / ei.size)[:, None, :]
    g[:, ai, :] -= ((du + dv) / ai.size)[:, None, :]
    return r, g



def relative_scale(base, extra):
    """Scale factor making ``extra`` contribute a FRACTION of ``base``'s own magnitude."""
    b = float(np.abs(np.asarray(base, dtype=np.float64)).mean())
    e = float(np.abs(np.asarray(extra, dtype=np.float64)).mean())
    if not np.isfinite(b) or not np.isfinite(e) or e <= 1e-12:
        return 0.0
    return b / e

def antibody_tokens(paratope_idx, chain_of):
    """All tokens on the CHAINS that carry the paratope, i.e. the whole Fv (VH+VL, or VHH)."""
    if chain_of is None:
        raise RuntimeError(
            "antibody_tokens needs chain_of (asym_id) and got None. Without it the antibody core "
            "cannot be identified and the orientation term would silently score 0 on every "
            "particle. Pass input_feature_dict['asym_id'], or disable orient_enabled.")
    pi = np.asarray(paratope_idx, dtype=int)
    ch = np.asarray(chain_of).reshape(-1)
    if pi.size == 0:
        return np.array([], dtype=int)
    return np.nonzero(np.isin(ch, list(antibody_chains(paratope_idx, chain_of))))[0].astype(int)


def antibody_chains(paratope_idx, chain_of):
    """The SET of chain ids that carry at least one paratope token."""
    pi = np.asarray(paratope_idx, dtype=int)
    ch = np.asarray(chain_of).reshape(-1)
    if pi.size == 0:
        return set()
    return set(ch[pi].tolist())

def facing_shell(res_pool, paratope_idx, epitope_idx, chain_of=None,
                 r_contact: float = 8.0, sigma: float = 2.0, break_r: float = 4.5):
    """ORIENTATION term: do the two surfaces FACE each other? ``(F, dF)``, F (P,), dF (P, n, 3)."""
    pool = np.asarray(res_pool, dtype=np.float64)
    P, n_res = pool.shape[0], pool.shape[1]
    out_g = np.zeros_like(pool)
    pi = np.asarray(paratope_idx, dtype=int)
    ei = np.asarray(epitope_idx, dtype=int)
    if pi.size == 0 or ei.size == 0:
        return np.zeros(P), out_g

    if chain_of is None:
        step = np.linalg.norm(pool[:, 1:, :] - pool[:, :-1, :], axis=-1).mean(axis=0)
        ch = np.concatenate([[0], np.cumsum(step > float(break_r))])     # new label after each break
    else:
        ch = np.asarray(chain_of)
    prev = np.arange(n_res) - 1
    nxt = np.arange(n_res) + 1
    ok = np.ones(n_res, dtype=bool)
    ok[0] = ok[-1] = False
    inner = np.ones(n_res, dtype=bool)
    inner[1:-1] = (ch[:-2] == ch[1:-1]) & (ch[2:] == ch[1:-1])
    ok &= inner
    prev = np.clip(prev, 0, n_res - 1)
    nxt = np.clip(nxt, 0, n_res - 1)

    cen = pool.mean(axis=1, keepdims=True)                              # (P,1,3)
    m = pool - 0.5 * (pool[:, prev, :] + pool[:, nxt, :])               # backbone normal
    m = np.where(ok[None, :, None], m, pool - cen)                      # break/terminus fallback
    m_len = np.maximum(np.linalg.norm(m, axis=-1, keepdims=True), 1e-9)
    nrm = m / m_len

    d = pool[:, ei, :][:, None, :, :] - pool[:, pi, :][:, :, None, :]   # (P,Np,Ne,3) e - p
    r = np.linalg.norm(d, axis=-1)
    safe = np.maximum(r, 1e-9)
    u = d / safe[..., None]
    sh = np.exp(-((r - float(r_contact)) ** 2) / (2.0 * float(sigma) ** 2))
    dsh_dr = sh * (-(r - float(r_contact)) / float(sigma) ** 2)

    np_, ne_ = pi.size, ei.size
    fp = (nrm[:, pi, :][:, :, None, :] * u).sum(-1)                     # <n_p, u>
    fe = (nrm[:, ei, :][:, None, :, :] * -u).sum(-1)                    # <n_e, -u>
    f = fp * fe
    act = f > 0.0                                                       # relu gate
    F = (sh * np.where(act, f, 0.0)).sum(axis=(1, 2)) / float(np_ * ne_)

    # ---- gradient. Three routes: through r (the shell), through u, and through the normals n.
    w = 1.0 / float(np_ * ne_)
    g_r = np.where(act, dsh_dr * f, 0.0)[..., None] * u * w             # d/d(e) via r; -1 for p
    # via u: d f / d u = fe * n_p - fp * n_e , and d u / d(x_e) = (I - u u^T)/r
    df_du = (np.where(act, fe, 0.0)[..., None] * nrm[:, pi, :][:, :, None, :]
             - np.where(act, fp, 0.0)[..., None] * nrm[:, ei, :][:, None, :, :])
    proj = df_du - (df_du * u).sum(-1)[..., None] * u
    g_u = (sh[..., None] * proj / safe[..., None]) * w
    np.add.at(out_g, (slice(None), ei), (g_r + g_u).sum(axis=1))
    np.add.at(out_g, (slice(None), pi), -(g_r + g_u).sum(axis=2))

    # via the normals: dF/dn_p = sum_e sh*fe*u  (and the mirror for n_e), then push through
    # n = m/|m| (Jacobian (I - n n^T)/|m|) and m = x_i - (x_{i-1}+x_{i+1})/2.
    gp = (sh * np.where(act, fe, 0.0))[..., None] * u * w               # (P,Np,Ne,3)
    ge = (sh * np.where(act, fp, 0.0))[..., None] * -u * w
    dn = np.zeros_like(pool)
    np.add.at(dn, (slice(None), pi), gp.sum(axis=2))
    np.add.at(dn, (slice(None), ei), ge.sum(axis=1))
    dm = (dn - (dn * nrm).sum(-1)[..., None] * nrm) / m_len             # through the normalisation
    np.add.at(out_g, (slice(None), np.arange(n_res)[ok]), dm[:, ok, :])
    np.add.at(out_g, (slice(None), prev[ok]), -0.5 * dm[:, ok, :])
    np.add.at(out_g, (slice(None), nxt[ok]), -0.5 * dm[:, ok, :])
    bad = ~ok
    if bad.any():                                                       # fallback branch: n from x-cen
        np.add.at(out_g, (slice(None), np.arange(n_res)[bad]), dm[:, bad, :])
        out_g -= dm[:, bad, :].sum(axis=1, keepdims=True) / float(n_res)
    return F, out_g


def approach_normal(res_pool, epitope_idx, chain_of=None, r_local=15.0):
    """Ground-truth-free "outward at the epitope": unit(mu_e - centroid(antigen within r_local))."""
    pool = np.asarray(res_pool, dtype=np.float64)
    ei = np.asarray(epitope_idx, dtype=int)
    if ei.size == 0:
        return np.zeros((len(pool), 3))
    me = pool[:, ei, :].mean(1)
    if chain_of is not None:
        ch = np.asarray(chain_of)
        ag = np.isin(ch, np.unique(ch[ei]))
    else:
        ag = np.ones(pool.shape[1], dtype=bool)
        ag[ei] = True
    idx = np.where(ag)[0]
    out = np.zeros((len(pool), 3))
    for k in range(len(pool)):
        d = np.linalg.norm(pool[k, idx, :] - me[k], axis=1)
        near = idx[d < float(r_local)]
        loc = pool[k, near, :].mean(0) if near.size > 3 else pool[k, idx, :].mean(0)
        v = me[k] - loc
        n = np.linalg.norm(v)
        out[k] = v / n if n > 1e-8 else np.array([0.0, 0.0, 1.0])
    return out


def direction_reward(res_pool, paratope_idx, epitope_idx, chain_of=None, r_local=15.0):
    """APPROACH-DIRECTION reward and gradient: ``R = <u, n>`` with ``u = unit(mu_p - mu_e)``."""
    pool = np.asarray(res_pool, dtype=np.float64)
    pi = np.asarray(paratope_idx, dtype=int)
    sites = epitope_sites(epitope_idx)
    R = np.zeros(len(pool))
    dR = np.zeros_like(pool)
    if pi.size == 0 or not sites:
        return R, dR
    ei = np.asarray(sites[0], dtype=int)
    if ei.size == 0:
        return R, dR
    n = approach_normal(pool, ei, chain_of, r_local)          # (P,3), stop-gradient
    v = pool[:, pi, :].mean(1) - pool[:, ei, :].mean(1)        # (P,3)
    d = np.linalg.norm(v, axis=-1)
    safe = np.maximum(d, 1e-8)[:, None]
    u = v / safe
    R = np.einsum("pk,pk->p", u, n)
    # dR/dv = (n - (u.n) u)/||v||, the component of n tangential to u
    g = (n - R[:, None] * u) / safe
    dR[:, pi, :] = g[:, None, :] / pi.size
    dR[:, ei, :] = -g[:, None, :] / ei.size
    return R, dR


def clash_penalty(res_pool, idx_a, idx_b, d_min: float = 4.0, cross_only: bool = True):
    """FLAT-BOTTOM interpenetration penalty and its gradient. ``(C, dC)``, C shape (P,), dC (P,n,3)."""
    pool = np.asarray(res_pool, dtype=np.float64)
    P = len(pool)
    out_g = np.zeros_like(pool)
    ia = np.asarray(idx_a, dtype=int)
    ib = np.asarray(idx_b, dtype=int)
    if ia.size == 0 or ib.size == 0:
        return np.zeros(P), out_g
    if not cross_only:
        ia = ib = np.unique(np.concatenate([ia, ib]))
    A = pool[:, ia, :]                                   # (P, Na, 3)
    B = pool[:, ib, :]                                   # (P, Nb, 3)
    d = A[:, :, None, :] - B[:, None, :, :]              # (P, Na, Nb, 3)
    r = np.linalg.norm(d, axis=-1)
    safe = np.maximum(r, 1e-8)
    viol = np.maximum(0.0, float(d_min) - r)             # 0 where already far enough
    n_pair = float(ia.size * ib.size)
    C = -(viol ** 2).sum(axis=(1, 2)) / n_pair
    # dC/dr = +2*viol/n_pair (pushing r UP), dr/dA = +u, dr/dB = -u
    coef = (2.0 * viol / n_pair)[..., None] * (d / safe[..., None])
    np.add.at(out_g, (slice(None), ia), coef.sum(axis=2))
    np.add.at(out_g, (slice(None), ib), -coef.sum(axis=1))
    return C, out_g


def site_contact_reward(res_pool, paratope_idx, epitope_idx, target_dist=0.0,
                        contact_weight=1.0, r_contact=8.0, contact_sigma=2.0,
                        hinge=False, huber_delta=0.0, contact_gate=0.0, site_floor=0.0,
                        softmin_tau=0.0, dev_cap=0.0,
                      contact_saturate_tau=0.0):
    """Combined binding-SITE + CONTACT reward."""
    sites = epitope_sites(epitope_idx)
    if len(sites) > 1 and softmin_tau > 0.0:
        tgt_all = np.broadcast_to(np.asarray(target_dist, dtype=np.float64), (len(res_pool),))
        rj = np.stack([site_contact_reward(res_pool, paratope_idx, ei_j, tgt_all,
                                           contact_weight=contact_weight, r_contact=r_contact,
                                           contact_sigma=contact_sigma, hinge=hinge,
                                           huber_delta=huber_delta, contact_gate=contact_gate,
                                           site_floor=site_floor, softmin_tau=0.0, dev_cap=dev_cap,
                                           contact_saturate_tau=contact_saturate_tau)
                       for ei_j in sites])
        m = rj.max(0)
        return m + float(softmin_tau) * np.log(np.exp((rj - m) / float(softmin_tau)).sum(0))
    if len(sites) > 1:
        pool = np.asarray(res_pool, dtype=np.float64)
        tgt_all = np.broadcast_to(np.asarray(target_dist, dtype=np.float64), (len(pool),))
        gid = epitope_site_ids(len(pool), len(sites), tgt_all)
        out = np.zeros(len(pool))
        for g, ei_g in enumerate(sites):
            m = gid == g
            if m.any():
                out[m] = site_contact_reward(pool[m], paratope_idx, ei_g, tgt_all[m],
                                             contact_weight=contact_weight, r_contact=r_contact,
                                             contact_sigma=contact_sigma, hinge=hinge,
                                             huber_delta=huber_delta, contact_gate=contact_gate,
                                             site_floor=site_floor, softmin_tau=softmin_tau,
                                             dev_cap=dev_cap,
                                             contact_saturate_tau=contact_saturate_tau)
        return out
    site = epitope_centroid_reward(res_pool, paratope_idx, epitope_idx, target_dist, hinge, huber_delta, contact_gate, site_floor, 0.0, dev_cap)
    if not contact_weight:
        return site
    C, _ = contact_shell(res_pool, paratope_idx, epitope_idx, r_contact, contact_sigma,
                         saturate_tau=contact_saturate_tau)
    return site + float(contact_weight) * C


def site_contact_grad(res_pool, paratope_idx, epitope_idx, target_dist=0.0,
                      contact_weight=1.0, r_contact=8.0, contact_sigma=2.0,
                      hinge=False, huber_delta=0.0, contact_gate=0.0, site_floor=0.0,
                      softmin_tau=0.0, dev_cap=0.0,
                      contact_saturate_tau=0.0):
    """d(site_contact_reward)/d(coords), (P, n_res, 3). Both terms in closed form: no extra NFE."""
    sites = epitope_sites(epitope_idx)
    if len(sites) > 1 and softmin_tau > 0.0:
        tgt_all = np.broadcast_to(np.asarray(target_dist, dtype=np.float64), (len(res_pool),))
        gj = np.stack([site_contact_grad(res_pool, paratope_idx, ei_j, tgt_all,
                                           contact_weight, r_contact, contact_sigma, hinge, huber_delta, contact_gate, site_floor, 0.0, dev_cap)
                       for ei_j in sites])
        rj = np.stack([site_contact_reward(res_pool, paratope_idx, ei_j, tgt_all,
                                           contact_weight, r_contact, contact_sigma, hinge, huber_delta, contact_gate, site_floor, 0.0, dev_cap)
                       for ei_j in sites])
        w = np.exp((rj - rj.max(0)) / float(softmin_tau))
        w = w / w.sum(0)                       # d/dx of tau*logsumexp is the softmax average
        return (w[:, :, None, None] * gj).sum(0)
    if len(sites) > 1:
        pool = np.asarray(res_pool, dtype=np.float64)
        tgt_all = np.broadcast_to(np.asarray(target_dist, dtype=np.float64), (len(pool),))
        gid = epitope_site_ids(len(pool), len(sites), tgt_all)
        out = np.zeros_like(pool)
        for g, ei_g in enumerate(sites):
            m = gid == g
            if m.any():
                out[m] = site_contact_grad(pool[m], paratope_idx, ei_g, tgt_all[m],
                                           contact_weight, r_contact, contact_sigma, hinge, huber_delta, contact_gate, site_floor, softmin_tau, dev_cap)
        return out
    g = epitope_centroid_grad(res_pool, paratope_idx, epitope_idx, target_dist, hinge, huber_delta, contact_gate, site_floor, 0.0, dev_cap)
    if not contact_weight:
        return g
    _, gc = contact_shell(res_pool, paratope_idx, epitope_idx, r_contact, contact_sigma,
                          saturate_tau=contact_saturate_tau)
    return g + float(contact_weight) * gc


def cdr_rmsd_reward(res_pool: np.ndarray, ref_res: np.ndarray, paratope_idx: np.ndarray,
                    target_rmsd=0.0) -> np.ndarray:
    """Per-particle CDR/paratope conformational-diversity steering reward, backend-agnostic."""
    from method.verifier.diversity import kabsch_rmsd   # numpy primitive, imported at call time
    idx = np.asarray(paratope_idx).astype(int)
    ref = np.asarray(ref_res, dtype=np.float64)[idx]
    r = np.array([kabsch_rmsd(np.asarray(c, dtype=np.float64)[idx], ref) for c in res_pool],
                 dtype=np.float64)
    d = np.asarray(target_rmsd, dtype=np.float64)
    if d.ndim == 0:
        return r if d <= 0 else -(r - d) ** 2
    d = np.broadcast_to(d, r.shape)                      # per-particle targets (the sweep)
    return np.where(d <= 0, r, -(r - d) ** 2)




def _pairwise_kabsch(res_pool: np.ndarray, paratope_idx: np.ndarray):
    """(rmsd matrix, per-pair difference tensor) over the paratope: ``rho`` (P,P) and ``diff`` (P,P,M,3)
        with ``diff[i,j] = A_i - A_j R*_ij^T`` (both centred), i.e. rho_ij * d rho_ij/dX_i * M.
    """
    idx = np.asarray(paratope_idx).astype(int)
    A = np.asarray(res_pool, dtype=np.float64)[:, idx, :]
    P, M = A.shape[0], A.shape[1]
    Ac = A - A.mean(1, keepdims=True)                      # centre each particle's paratope
    # H[i,j] = Ac_i^T Ac_j  (P,P,3,3) via one einsum, then one batched SVD
    H = np.einsum("ima,jmb->ijab", Ac, Ac)
    U, S, Vt = np.linalg.svd(H.reshape(P * P, 3, 3))
    d = np.where(np.linalg.det(U @ Vt) >= 0, 1.0, -1.0)
    D = np.zeros((P * P, 3, 3)); D[:, 0, 0] = D[:, 1, 1] = 1.0; D[:, 2, 2] = d
    R = (U @ D @ Vt).reshape(P, P, 3, 3)                   # maps Ac_j into Ac_i's frame
    diff = Ac[:, None, :, :] - np.einsum("jmb,ijab->ijma", Ac, R)
    rho = np.sqrt(np.maximum((diff ** 2).sum((-1, -2)) / M, 0.0))
    return rho, diff



def _pairwise_kabsch_site(res_pool: np.ndarray, paratope_idx: np.ndarray, antigen_idx: np.ndarray):
    """SITE distance: paratope displacement after superposing on the ANTIGEN. ``(rho, diff)``."""
    X = np.asarray(res_pool, dtype=np.float64)
    pi = np.asarray(paratope_idx).astype(int)
    gi = np.asarray(antigen_idx).astype(int)
    G = X[:, gi, :]
    A = X[:, pi, :]
    P, M = A.shape[0], A.shape[1]
    # centre BOTH sets on the ANTIGEN centroid, so the paratope keeps its offset from the antigen.
    gc = G.mean(1, keepdims=True)
    Gc, Ac = G - gc, A - gc
    H = np.einsum("ima,jmb->ijab", Gc, Gc)
    U, S, Vt = np.linalg.svd(H.reshape(P * P, 3, 3))
    d = np.where(np.linalg.det(U @ Vt) >= 0, 1.0, -1.0)
    D = np.zeros((P * P, 3, 3)); D[:, 0, 0] = D[:, 1, 1] = 1.0; D[:, 2, 2] = d
    R = (U @ D @ Vt).reshape(P, P, 3, 3)              # maps j's antigen frame into i's
    diff = Ac[:, None, :, :] - np.einsum("jmb,ijab->ijma", Ac, R)
    rho = np.sqrt(np.maximum((diff ** 2).sum((-1, -2)) / M, 0.0))
    return rho, diff


# ------------------------------------------------------------------ d* sweep (ConforMix umbrellas)


def sweep_targets(n_particles: int, targets: Sequence[float]) -> np.ndarray:
    """Per-particle target RMSD d*, assigned round-robin over ``targets``."""
    t = np.asarray(list(targets), dtype=np.float64)
    if t.size == 0:
        return np.zeros(int(n_particles), dtype=np.float64)
    return t[np.arange(int(n_particles)) % t.size]


# ------------------------------------------------------------------ physicality filter (ConforMix)


def count_ca_clashes(ca: np.ndarray, clash_r: float = 3.5, seq_exclude: int = 1,
                     chain_ids=None) -> int:
    """Cα pairs closer than ``clash_r`` Å, excluding pairs within ``seq_exclude`` in sequence
        (consecutive Cα sit ~3.8 Å apart by construction, so they are not clashes).
    """
    x = np.asarray(ca, dtype=np.float64)
    d = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)
    i, j = np.triu_indices(len(x), k=1)
    close = d[i, j] < float(clash_r)
    if chain_ids is None:
        adj = (j - i) <= int(seq_exclude)
    else:
        ch = np.asarray(chain_ids, dtype=object)
        adj = (ch[i] == ch[j]) & ((j - i) <= int(seq_exclude))
    return int((close & ~adj).sum())


def plddt_window_drop(plddt: np.ndarray, ref_plddt: np.ndarray, window: int = 10) -> float:
    """Worst FRACTIONAL pLDDT drop vs the reference over any sliding ``window`` of residues:
        max_w (1 − mean(plddt_w)/mean(ref_plddt_w)). 0 means never worse than the reference.
    """
    p = np.asarray(plddt, dtype=np.float64).reshape(-1)
    q = np.asarray(ref_plddt, dtype=np.float64).reshape(-1)
    n = min(p.size, q.size)
    if n == 0:
        return 0.0
    w = max(1, min(int(window), n))
    ker = np.ones(w) / w
    pm, qm = np.convolve(p[:n], ker, mode="valid"), np.convolve(q[:n], ker, mode="valid")
    ok = qm > 0
    return float(np.max(1.0 - pm[ok] / qm[ok])) if ok.any() else 0.0


def physicality_mask(plddt_pool=None, ref_plddt=None, ca_pool=None, window: int = 10,
                     max_drop: float = 0.20, clash_r: float = 3.5, max_clashes: int = 0,
                     use_clash: bool = True, use_plddt: bool = True,
                     return_detail: bool = False):
    """Boolean KEEP mask over particles: True = physically acceptable (the ConforMix filtering step)."""
    have_p = use_plddt and plddt_pool is not None and ref_plddt is not None
    have_c = use_clash and ca_pool is not None
    P = len(plddt_pool) if plddt_pool is not None else (len(ca_pool) if ca_pool is not None else 0)
    detail = {}
    if have_p:
        detail["plddt_window_drop"] = np.array(
            [plddt_window_drop(plddt_pool[i], ref_plddt, window=window) <= float(max_drop)
             for i in range(P)], dtype=bool)
    if have_c:
        detail["ca_clash"] = np.array(
            [count_ca_clashes(ca_pool[i], clash_r=clash_r) <= int(max_clashes)
             for i in range(P)], dtype=bool)
    keep = np.ones(P, dtype=bool)
    for m in detail.values():
        keep &= m
    return (keep, detail) if return_detail else keep




# ------------------------------------------------------------------ Noise-search ES


def anneal_scale(t: float, tau1: float = 0.2, tau2: float = 0.9) -> float:
    """CADS's piecewise-linear schedule (Sadat et al., ICLR 2024), reused as a STRENGTH ramp."""
    t = float(t)
    if t >= float(tau2):
        return 1.0
    if t <= float(tau1):
        return 0.0
    return (t - float(tau1)) / max(float(tau2) - float(tau1), 1e-9)


def normalise_step(phi: np.ndarray, target_rms_a: float, subset_idx=None) -> np.ndarray:
    """Rescale an update so the POOL-WIDE RMS displacement is exactly ``target_rms_a`` Angstroms."""
    idx = None if subset_idx is None else np.asarray(subset_idx).astype(int)
    sub = phi if idx is None else phi[:, idx, :]
    rms = float(np.sqrt(np.mean((sub ** 2).sum(-1)))) if sub.size else 0.0
    if rms <= 1e-12 or float(target_rms_a) <= 0.0:
        return np.zeros_like(phi)
    return phi * (float(target_rms_a) / rms)




def cdr_distogram_reward(res_pool, ref_res, paratope_idx, target_dev=0.0, delta=2.0):
    """CONFORMATIONAL steering on the intra-paratope DISTANCE MATRIX, superposition-free."""
    pool = np.asarray(res_pool, dtype=np.float64)
    idx = np.asarray(paratope_idx, dtype=int)
    g = np.zeros_like(pool)
    if idx.size < 2:
        return np.zeros(len(pool)), g
    ref = np.asarray(ref_res, dtype=np.float64)[idx]
    iu = np.triu_indices(idx.size, k=1)
    d_ref = np.linalg.norm(ref[iu[0]] - ref[iu[1]], axis=-1)          # (K,)
    P = pool[:, idx, :]                                               # (P, k, 3)
    v = P[:, iu[0], :] - P[:, iu[1], :]                               # (P, K, 3)
    d = np.linalg.norm(v, axis=-1)                                    # (P, K)
    dev = d - d_ref[None, :]
    clipped = np.clip(dev, -float(delta), float(delta))
    # RMS deviation of the distogram, the scalar CV that d* is expressed in
    rms_dev = np.sqrt(np.mean(clipped ** 2, axis=1))
    t = np.broadcast_to(np.asarray(target_dev, dtype=np.float64), (len(pool),))
    # d*==0 maximise the deviation, d*>0 a harmonic well on it, mirroring cdr_rmsd_reward exactly
    R = np.where(t <= 0, rms_dev, -((rms_dev - t) ** 2))
    # chain rule: dR/d(rms_dev) * d(rms_dev)/d(clipped) * d(d)/dx, zero outside the clip
    dR_drms = np.where(t <= 0, 1.0, -2.0 * (rms_dev - t))             # (P,)
    active = np.abs(dev) < float(delta)
    safe = np.maximum(rms_dev, 1e-12)[:, None]
    dr_dclip = clipped / (safe * clipped.shape[1])                    # d(rms)/d(clipped_i)
    u = np.where(d[..., None] > 1e-8, v / np.maximum(d[..., None], 1e-8), 0.0)
    coef = (dR_drms[:, None] * dr_dclip * active)[..., None] * u      # (P, K, 3)
    gp = np.zeros_like(P)
    np.add.at(gp, (slice(None), iu[0]), coef)
    np.add.at(gp, (slice(None), iu[1]), -coef)
    g[:, idx, :] = gp
    return R, g


def cdr_rmsd_grad(res_pool: np.ndarray, ref_res: np.ndarray, paratope_idx: np.ndarray,
                  target_rmsd=0.0) -> np.ndarray:
    """d(reward)/d(residue coords) for the CDR reward, (P, n_res, 3), zero off the paratope."""
    from method.diffusion_side.guidance import kabsch_rmsd_grad
    idx = np.asarray(paratope_idx).astype(int)
    ref = np.asarray(ref_res, dtype=np.float64)[idx]
    pool = np.asarray(res_pool, dtype=np.float64)
    d = np.broadcast_to(np.asarray(target_rmsd, dtype=np.float64), (len(pool),))
    out = np.zeros_like(pool)
    for i, c in enumerate(pool):
        rmsd, g = kabsch_rmsd_grad(c[idx], ref)
        out[i, idx] = g if d[i] <= 0 else (-2.0 * (rmsd - d[i])) * g
    return out
