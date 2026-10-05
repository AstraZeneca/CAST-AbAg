"""Family ``search`` (explore over the initial noise), plus the shared spec.

Two things live here: the executable :class:`DiffusionSearchSpec` that configures EVERY
diffusion-side method, and the ``search`` family itself.

  * ``noise_search`` evolutionary search over the INITIAL noise, where each candidate is a full
                     sampling run projected back onto the Gaussian shell. (Jajal et al. 2506.00299)

The taxonomy is also here (:data:`DIFFUSION_FAMILIES`, :data:`ALL_ALGORITHMS`, :func:`family_of`),
because it is the one table both ``run_ablation`` and the job preflight import, which is what stops
the runner and the validator from drifting apart.

``models/protenix_hooks.install_diffusion`` reads the spec from ``$PROTENIX_DIFFUSION_SPEC`` and
dispatches on ``algorithm``. The plug-in reward is ``ipsae`` (interface confidence on x_hat0),
``cdr_rmsd`` (paratope conformation) or ``epitope`` (binding-site centroid distance). Rewards are
scored on a coarse schedule and are scalars per particle, so algorithm and reward compose without
touching the base model.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import List, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ipSAE reward cutoffs (match method/verifier/ipsae.py defaults / the B0.b run).
_PAE_CUT, _PLDDT_CUT = 10.0, 0.0

# STALE: `svgd` and every `svgd_*` field below are UNUSED and UNMAINTAINED. No shipped config, job
# template or reported result selects them, and they are excluded from the paper. They remain only
# because the transport path shares its displacement counters and its paratope resolution with
# `af3_red`, which is reported, so removing them is a sampler change rather than a deletion. Treat
# `algorithm=svgd` as unsupported and do not extend it.
DIFFUSION_FAMILIES = {
    "best_of_n": (),
    "search": ("noise_search",),
    "guidance": ("fk", "tds", "cast"),
    "interaction": ("svgd", "af3_red"),
}
ALL_ALGORITHMS = tuple(a for algos in DIFFUSION_FAMILIES.values() for a in algos)


SITE_REWARDS = ("epitope", "site_contact")
CONF_REWARDS = ("cdr_rmsd", "cdr_distogram")

ALL_REWARDS = ("ipsae", "none") + CONF_REWARDS + SITE_REWARDS


def contact_weight_of(spec) -> float:
    """contact_weight, gated on reward=site_contact. 0.0 means the contact shell is OFF."""
    return (float(getattr(spec, "contact_weight", 0.0))
            if str(getattr(spec, "reward", "")) == "site_contact" else 0.0)


def clash_weight_of(spec) -> float:
    """clash_weight, gated on its OWN flag clash_enabled, independent of the reward mode."""
    return (float(getattr(spec, "clash_weight", 0.0))
            if bool(getattr(spec, "clash_enabled", False)) else 0.0)


def reserve_frac_of(spec) -> float:
    """reserve_frac, gated on its OWN flag reserve_enabled. 0.0 when the flag is off."""
    return (float(getattr(spec, "reserve_frac", 0.0))
            if bool(getattr(spec, "reserve_enabled", False)) else 0.0)



def realloc_frac_of(spec) -> float:
    """``realloc_max_move_frac``, gated on ``realloc_enabled``. 0.0 when the flag is off."""
    return (float(getattr(spec, "realloc_max_move_frac", 0.0))
            if bool(getattr(spec, "realloc_enabled", False)) else 0.0)

def direction_weight_of(spec) -> float:
    """direction_weight, gated on its OWN flag direction_enabled. 0.0 when the flag is off."""
    return (float(getattr(spec, "direction_weight", 0.0))
            if bool(getattr(spec, "direction_enabled", False)) else 0.0)



def orient_weight_of(spec) -> float:
    """orient_weight, gated on its OWN flag orient_enabled, independent of the reward mode."""
    return (float(getattr(spec, "orient_weight", 0.0))
            if bool(getattr(spec, "orient_enabled", False)) else 0.0)

def facing_weight_of(spec) -> float:
    """facing_weight, gated on its OWN flag facing_enabled, independent of the reward mode."""
    return (float(getattr(spec, "facing_weight", 0.0))
            if bool(getattr(spec, "facing_enabled", False)) else 0.0)


def family_of(algorithm: str) -> str:
    """The family an algorithm belongs to. Raises on an unknown algorithm rather than guessing."""
    for fam, algos in DIFFUSION_FAMILIES.items():
        if algorithm in algos:
            return fam
    raise ValueError("unknown diffusion algorithm %r (%s)" % (algorithm, ", ".join(ALL_ALGORITHMS)))


@dataclass(frozen=True)
class DiffusionSearchSpec:
    # The population (#particles) is N_sample = model.run.n_samples_per_seed, the hook
    # reads it from the sampler call, so it is NOT duplicated here. Reward = plug-in ipSAE on x̂₀.

    # --- reward: the scalar the search maximises (backend-agnostic, see method/diffusion_side/README.md) ---
    reward: str = "ipsae"              # ipsae | cdr_rmsd | cdr_distogram | epitope (paratope-epitope centroid
    epitope_source: str = "predicted"
    epitope_map_builder: str = ""
    svgd_bandwidth: float = 0.0        # 0 = SVGD median heuristic
    rmsd_subset: str = "paratope"      # paratope | all | chain
    rmsd_chain: int = -1               # only for rmsd_subset=chain; -1 = unset
    cdr_target_rmsd: float = 0.0       # single umbrella: 0 = unbounded maximise (COLLAPSES the pool), >0 = harmonic at d*
    umbrella_sweep: str = ""
    sweep_units: str = "auto"          # auto | rmsd_A | centroid_A
    contact_weight: float = 0.0        # 0 => identical to reward=epitope (exactly, bit-for-bit)
    contact_r: float = 8.0             # contact distance the shell peaks at (A), CB-CB-ish
    contact_sigma: float = 2.0         # shell width (A)
    contact_saturate_tau: float = 0.0
    guidance_scale: float = 0.0
    guidance_max_norm: float = 700.0   # clip the per-particle guidance displacement (A/step), 0 = off
    guidance_max_rms_a: float = 0.0
    guidance_schedule: str = "sigma2"  # sigma2 = eps_t = scale*sigma^2/sqrt(1+sigma^2), the TDS/ConforMix
    phys_use_clash: bool = False       # non-adjacent Ca clash test (coords only, no extra NFE)
    phys_use_plddt: bool = False       # pLDDT window-drop vs the reference (ConforMix's primary
    physicality_filter: bool = False   # master switch; OFF for every algorithm: a post-hoc/weighted filter
                                       # replaces rejects with duplicates of survivors, which
    clash_r: float = 3.5               # Ca pair closer than this (A), non-adjacent, counts as a clash
    max_clashes: int = 0               # drop a particle with more than this many clashes
    plddt_window: int = 10             # pLDDT window length for the (reusable) window-drop criterion
    plddt_max_drop: float = 0.20       # drop if any window falls this fraction below the reference

    # --- shared steering schedule (keeps the expensive confidence-head reward coarse) ---
    schedule_every: int = 20           # score / branch / resample every this many denoise steps
    schedule_end_frac: float = 1.0
    schedule_start_frac: float = 0.5   # skip steering for the first frac of steps (early-σ x̂₀ noisy)
    mini_rollout_steps: int = 4        # (b) score a short DETERMINISTIC rollout (this many denoise


    # --- family `guidance`: fk + tds (reward weighting / resampling) ---
    potential: str = "difference"      # difference | sum
    weight_level_correction: bool = False
    lam: float = 4.0                   # tilt temperature λ. NOT ignored when lam_adaptive: it is
    resample: str = "ess"              # ess | every_k | none
    resample_every_k: int = 1          # in units of schedule_every events (every_k mode)
    ess_per_cell_trigger: bool = False
    ess_fraction: float = 0.5          # resample when ESS < ess_fraction · N (ess mode), the FK
    algorithm: str = "tds"
    red_weight: float = 100.0
    red_sigma: float = 2.0
    red_n_smooth: int = 10             # +-residues the Ca force is averaged over (chain-aware). The
    red_bias_tmin: float = 2.0         # no bias at noise levels <= this
    red_bias_restrict: bool = True     # reference default
    red_cut_rmsd: float = 0.5          # reference default (Angstrom)
    red_rmsd_subset: str = "all"       # all (paper) | paratope | chain
    red_plddt_min: float = 60.0        # reference excludes residues below this from the bias RMSD.
    lam_adaptive: bool = True
    target_ess_frac: float = 0.4
    resample_scheme: str = "systematic"  # systematic (TDS default) | multinomial (FK reference)
    # The FK reference resamples only between 20% and 80% of the trajectory. Cloning near the end is
    # unrecoverable: there is no remaining noise to re-diversify the duplicates.
    resample_start_frac: float = 0.2
    resample_end_frac: float = 0.8
    resample_windows: str = ""
    # STALE, see the note at DIFFUSION_FAMILIES. These four and `svgd_bandwidth` above, together
    # with `reserve_repulsion`, `svgd_site_kernel` and `site_archive_weight`, exist only for
    # algorithm=svgd and are inert under every supported algorithm.
    svgd_stepsize: float = 0.25
    svgd_adagrad_alpha: float = 0.9
    svgd_repulsion_only: bool = False
    # ABLATION: drop the kernel repulsion, keep the drive. Mutually exclusive with
    # svgd_repulsion_only -- setting both leaves phi identically zero, so preflight rejects it.
    svgd_drive_only: bool = False
    svgd_drive_self_norm: bool = False
    svgd_site_kernel: bool = False
    site_step_a: float = 0.0
    site_archive_weight: float = 0.0
    site_anneal_tau1: float = 0.2
    site_anneal_tau2: float = 0.9
    boltz_perturb_enabled: bool = False
    boltz_perturb_tcp: bool = False
    boltz_perturb_tbp: bool = False
    boltz_perturb_noise_scale: float = 0.3    # their make_token_bias_noise_fn default
    boltz_perturb_psi: float = 0.7            # their default. Mixes toward the ORIGINAL, so psi=0
                                              # is the identity, NOT full noise.
    boltz_perturb_rescale: bool = True        # restore the original mean/std on the targeted block
    boltz_perturb_norm_proj: bool = True      # each perturbed vector keeps its original L2 norm
    boltz_perturb_mode: str = "cross_and_ll"
    # TADS: anneal the noise scale over sampling, full early and `tads_floor` late. False applies
    # noise_scale flat, which is what the source does with its schedule disabled.
    boltz_perturb_tads: bool = True
    boltz_perturb_tads_floor: float = 0.0
    site_hinge: bool = False
    site_huber_delta: float = 0.0
    site_contact_gate: float = 0.0
    site_floor: float = 0.0
    site_softmin_tau: float = 0.0
    site_dev_cap: float = 0.0

    clash_enabled: bool = False
    clash_weight: float = 0.0
    clash_d_min: float = 4.0
    orient_enabled: bool = False
    orient_weight: float = 0.0
    orient_tol_deg: float = 0.0
    facing_enabled: bool = False
    facing_weight: float = 0.0
    facing_sigma_max: float = 0.0

    site_freeze_epitope: bool = False

    repulsion_beta: float = 0.0
    distogram_delta: float = 2.0
    site_route: str = "coord"          # coord | off
    conf_route: str = "coord"          # coord | latent | off
    latent_eta: float = 0.01
    latent_reg_scale: float = 0.0      # L2 weight on the deltas, their embedding_reg_scale
    subsample_pool: int = 1
    # SITE REALLOCATION. Gated on its own flag for the same reason reserve_frac is (see
    # reserve_frac_of): a stray fraction in a config must not silently start moving budget.
    realloc_enabled: bool = False
    realloc_max_move_frac: float = 0.25    # cap per event, so no single step collapses the hedge
    realloc_min_per_cell: int = 6          # the floor below which within-cell resampling degenerates
    realloc_contact_r: float = 10.0        # contact radius for the antigen contact set
    direction_enabled: bool = False
    direction_weight: float = 0.0
    direction_r_local: float = 15.0       # neighbourhood for the local normal, as validated
    reserve_enabled: bool = False
    reserve_frac: float = 0.0
    reserve_repulsion: bool = False

    # --- noise_search (②) ---
    popsize: int = 16                  # ES population λ per generation
    generations: int = 10              # ES generations G
    sigma0: float = 0.5                # initial ES step size (in noise-shell units)
    elite_frac: float = 0.5            # μ/λ
    sigma_decay: float = 0.9           # geometric step-size decay per generation
    on_shell: bool = True              # renormalise candidates to ‖z‖≈√d (DNO guard)

    noise_search_algo: str = "cosyne"  # cosyne | es
    tournament_size: int = 4           # k-way tournament for parent selection
    mutation_stdev: float = 0.1        # GaussianMutation sigma. None-equivalent is 0.0 (off)
    mutation_probability: float = 0.0  # element-wise gate, 0 = every element mutated
    permute_all: bool = False          # EvoTorch default: rank-weighted, not blanket, permutation
    elitism_ratio: float = 0.0         # num_elites = int(popsize * elitism_ratio), 0 = no elitism
    cosyne_eta: float = 0.0            # >0 selects SimulatedBinaryCrossOver over OnePointCrossOver
    num_children: int = 0              # 0 => EvoTorch's default cross-over rate of 2.0 (= 2*popsize)

    # --- ipSAE reward cutoffs (canonical ipSAE is PAE-only, no distance gate) ---
    pae_cutoff: float = _PAE_CUT
    plddt_cutoff: float = _PLDDT_CUT

    def __post_init__(self) -> None:
        if self.weight_level_correction and int(self.subsample_pool) <= 1:
            object.__setattr__(self, "weight_level_correction", False)
            logger.warning(
                "weight_level_correction=True is a JOINT-model setting and was requested with "
                "subsample_pool=%d; forcing it to False. With one latent, r_0 varies only by "
                "initialisation noise, so adding lam*r_0 back reverts the FK weights from "
                "improvement to absolute reward and reinstates the degeneracy `difference` "
                "prevents. Set subsample_pool>1 if the level correction was actually intended.",
                int(self.subsample_pool))

    def label(self) -> str:
        if self.algorithm == "af3_red":
            # AF3-ReD uses NO reward weights and NO resampling: reporting lambda/resample here would
            # describe machinery that never ran.
            return (f"af3_red(w={self.red_weight:g},sigma={self.red_sigma:g},"
                    f"n_smooth={self.red_n_smooth},tmin={self.red_bias_tmin:g},"
                    f"width=sigma+noise_level)")
        if self.algorithm == "svgd":
            # SVGD is transport, not SMC: an FK-style label would misreport what ran.
            if self.svgd_site_kernel:
                _arch = (f",archive={self.site_archive_weight:g}"
                         if self.site_archive_weight > 0 else "")
                return (f"site-repulsion(kernel=antigen-aligned,step={self.site_step_a:g}A/step"
                        f",anneal={self.site_anneal_tau1:g}-{self.site_anneal_tau2:g}{_arch},"
                        f"h={'median' if not self.svgd_bandwidth else self.svgd_bandwidth})")
            _mode = ('repulsion-only' if self.svgd_repulsion_only
                     else 'drive-only' if self.svgd_drive_only else 'drive+repulsion')
            return (f"svgd({_mode},"
                    f"stepsize={self.svgd_stepsize},adagrad_alpha={self.svgd_adagrad_alpha},"
                    f"h={'median' if not self.svgd_bandwidth else self.svgd_bandwidth},"
                    f"mr={self.mini_rollout_steps})")
        if self.algorithm in ("fk", "tds", "cast"):
            sw = f",sweep={self.umbrella_sweep}" if self.sweep_list() else ""
            gd = f",guided={self.guidance_scale}" if self.guidance_scale else ""
            # an oracle run and a deployable one must not share a label
            src = f"[{self.epitope_source}]" if self.reward in SITE_REWARDS else ""
            # λ must read as adaptive when it is: a fixed "λ=4.0" on an adaptively-tempered run would
            # misreport the tilt that was actually applied (it is re-solved at every event).
            lam = (f"λ=ess{self.target_ess_frac:g}" if self.lam_adaptive else f"λ={self.lam}")
            return (f"{self.algorithm}({self.potential},{lam},resample={self.resample}"
                    f"/{self.resample_scheme}[{self.resample_windows or ('%g-%g' % (self.resample_start_frac, self.resample_end_frac))}],"
                    f"reward={self.reward}{src}{sw}{gd},mr={self.mini_rollout_steps})")
        if self.algorithm == "noise_search":
            return f"noise_search(λ={self.popsize},G={self.generations},σ0={self.sigma0},reward={self.reward})"
        return f"diffusion_search({self.algorithm})"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["label"] = self.label()
        return d

    def ipsae_kwargs(self) -> dict:
        return {"pae_cutoff": self.pae_cutoff, "plddt_cutoff": self.plddt_cutoff}

    def reward_affects_sampling(self) -> bool:
        """Does the reward change ANY sampling decision, or is scoring it pure waste?"""
        if self.reward == "none":
            return False
        if self.algorithm == "af3_red":
            return False                      # history-based bias, the reference has no reward term
        if self.algorithm == "svgd":
            # the SITE path is repulsion-only by construction and never reads score_res, so it
            # ignores the reward whatever svgd_repulsion_only says
            return not (self.svgd_repulsion_only or self.svgd_site_kernel)
        if self.algorithm == "noise_search":
            return True                       # selection IS the reward
        # fk / tds
        if self.physicality_filter:
            return True                       # the physicality mask down-weights particles
        if self.guidance_scale:
            return True                       # the gradient of the reward moves particles
        if self.lam_adaptive:
            return True                       # lambda is SOLVED from the reward spread
        # lam == 0 => the FK increment is exp(0 * dr) = 1, so weights never leave uniform, ESS never
        # drops, and resampling never fires. The reward is then read by nothing but the log line.
        return bool(self.lam)

    def uses_twisted_proposal(self) -> bool:
        """Does this algorithm shift the churn mean by the reward gradient (TDS's twist)?"""
        return self.algorithm in ("tds", "cast") and float(self.guidance_scale) > 0.0

    def sweep_list(self) -> list:
        """Parsed ``umbrella_sweep`` as a list of d* floats (empty list = no sweep)."""
        raw = self.umbrella_sweep
        if raw is None:
            return []
        if isinstance(raw, (list, tuple)):        # a config may hand us a real list instead of "1,2,3"
            return [float(x) for x in raw]
        if str(raw).strip().lower() == "auto":
            raise ValueError(
                "umbrella_sweep='auto' is no longer supported -- give explicit d* values "
                "(e.g. '7,9,11,13,15,17') or leave the field at its default.")
        toks = [x for x in str(raw).replace(" ", "").split(",") if x and x.lower() != "none"]
        return [float(x) for x in toks]


def spec_from_dict(d: dict) -> "DiffusionSearchSpec":
    """Rebuild from a JSON dict, ignoring unknown keys (e.g. the 'label' echo)."""
    fields = DiffusionSearchSpec.__dataclass_fields__
    return DiffusionSearchSpec(**{k: v for k, v in d.items() if k in fields})



def project_to_shell(z: np.ndarray) -> np.ndarray:
    """Renormalise a noise vector to the Gaussian shell ‖z‖≈√d that the denoiser expects."""
    z = np.asarray(z, dtype=np.float64)
    d = z.size
    n = np.linalg.norm(z)
    if n <= 0:
        return z
    return z * (np.sqrt(d) / n)


def es_ask(mean: np.ndarray, sigma: float, popsize: int, rng: np.random.Generator,
           on_shell: bool = True) -> np.ndarray:
    """Draw one ES generation: ``popsize`` candidates ~ N(mean, sigma²I), shell-projected."""
    d = mean.size
    samples = mean[None, :] + sigma * rng.standard_normal((popsize, d))
    if on_shell:
        samples = np.stack([project_to_shell(s) for s in samples])
    return samples



def centered_ranks(fitnesses: Sequence[float]) -> np.ndarray:
    """EvoTorch ``ranking_method="centered"``: linear ranks in [-0.5, +0.5], worst -> best."""
    f = np.asarray(fitnesses, dtype=np.float64)
    n = len(f)
    if n == 1:
        return np.zeros(1)
    order = np.argsort(f, kind="stable")          # ascending: worst first
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64) / (n - 1)   # [0, 1]
    return ranks - 0.5                                         # [-0.5, +0.5]


def cosyne_permute(X: np.ndarray, fitnesses: Sequence[float], rng: np.random.Generator,
                   permute_all: bool = False) -> np.ndarray:
    """CosynePermutation: per-column exchange among the selected solutions."""
    X = np.asarray(X, dtype=np.float64)
    P, d = X.shape
    if permute_all:
        prob = np.ones((P, d), dtype=np.float64)
    else:
        r = centered_ranks(fitnesses)
        prob = np.repeat((1.0 - np.power(r + 0.5, 1.0 / float(d)))[:, None], d, axis=1)
    mask = rng.random((P, d)) <= prob

    # rows to RECEIVE a permuted value, gathered to the top of each column
    perm_sort = np.repeat(np.arange(P)[:, None], d, axis=1)
    perm_sort[~mask] += P + 1
    perm_sort = np.sort(perm_sort, axis=0)
    # rows to DONATE, in random order among the masked entries of that column
    perm_rand = rng.random((P, d))
    perm_rand[~mask] = 1.0
    permutations = np.argsort(perm_rand, axis=0)
    mask_sorted = np.sort(mask.astype(np.int8), axis=0)[::-1].astype(bool)

    out = X.copy()
    rows, cols = np.nonzero(mask_sorted)
    if rows.size:
        out[perm_sort[rows, cols], cols] = X[permutations[rows, cols], cols]
    return out


def cosyne_tournament(X: np.ndarray, fitnesses: Sequence[float], tournament_size: int,
                      n_parents: int, rng: np.random.Generator) -> np.ndarray:
    """Tournament selection: n_parents winners of ``tournament_size``-way tournaments (higher=better)."""
    f = np.asarray(fitnesses, dtype=np.float64)
    P = len(f)
    k = max(2, min(int(tournament_size), P))
    picks = rng.integers(0, P, size=(n_parents, k))
    return np.asarray(X)[picks[np.arange(n_parents), np.argmax(f[picks], axis=1)]]


def one_point_crossover(parents: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """EvoTorch OnePointCrossOver -- the Cosyne default when ``eta`` is not given."""
    pa = np.asarray(parents, dtype=np.float64)
    n, d = pa.shape
    half = n // 2
    a, b = pa[:half], pa[half:2 * half]
    cut = rng.integers(1, d, size=half)
    idx = np.arange(d)[None, :] < cut[:, None]
    return np.concatenate([np.where(idx, a, b), np.where(idx, b, a)], axis=0)


def sbx_crossover(parents: np.ndarray, eta: float, rng: np.random.Generator) -> np.ndarray:
    """EvoTorch SimulatedBinaryCrossOver -- used instead of one-point when ``eta`` is given."""
    pa = np.asarray(parents, dtype=np.float64)
    n, d = pa.shape
    half = n // 2
    a, b = pa[:half], pa[half:2 * half]
    u = rng.random((half, d))
    beta = np.where(u <= 0.5, (2 * u) ** (1.0 / (eta + 1.0)),
                    (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0)))
    return np.concatenate([0.5 * ((1 + beta) * a + (1 - beta) * b),
                           0.5 * ((1 - beta) * a + (1 + beta) * b)], axis=0)


def gaussian_mutation(X: np.ndarray, stdev: float, rng: np.random.Generator,
                      probability: Optional[float] = None) -> np.ndarray:
    """EvoTorch GaussianMutation: additive N(0, stdev²), element-wise gated by ``probability``."""
    X = np.asarray(X, dtype=np.float64)
    noise = stdev * rng.standard_normal(X.shape)
    if probability is not None:
        noise = np.where(rng.random(X.shape) <= float(probability), noise, 0.0)
    return X + noise


def es_tell(samples: np.ndarray, fitnesses: Sequence[float], sigma: float,
            elite_frac: float = 0.5, sigma_decay: float = 0.9) -> Tuple[np.ndarray, float]:
    """(μ,λ) recombination update: new mean = fitness-weighted mean of the top-μ elites."""
    f = np.asarray(fitnesses, dtype=np.float64)
    X = np.asarray(samples, dtype=np.float64)
    N = len(f)
    mu = max(1, int(round(elite_frac * N)))
    elite_idx = np.argsort(-f, kind="stable")[:mu]
    # log-decreasing weights, normalised over the elites.
    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    w = w / w.sum()
    new_mean = (w[:, None] * X[elite_idx]).sum(axis=0)
    return new_mean, float(sigma * sigma_decay)
