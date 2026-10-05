# `method/diffusion_side`, diffusion-side test-time scaling

Test-time-scaling methods that act **inside the reverse diffusion**, coupling the `N` samples of a
seed instead of drawing them independently. A population of `N` particles is stepped through the
denoiser and, on a coarse schedule, a scalar reward drives a selection / weighting / steering decision.

| family | module | algorithm | mechanism | reference |
|--------|--------|-----------|-----------|-----------|
| `best_of_n` | `best_of_n.py` | | independent draws, `argmax` reward post-hoc (the null lever) | |
| `search` | `diffusion_search.py` | `noise_search` | evolutionary search over the initial noise | Jajal et al. [2506.00299](https://arxiv.org/abs/2506.00299) |
| `guidance` | `guidance.py` | `fk` | reward + importance weights + resampling, **no gradient** | Singhal et al. [2501.06848](https://arxiv.org/abs/2501.06848) |
| `guidance` | `guidance.py` | `tds` | twisted proposal **with** the reward gradient + exact weight correction | Wu et al. [2306.17775](https://arxiv.org/abs/2306.17775) |
| `guidance` | `guidance.py` | `cast` | `tds` over `K` subsample latents in one filter, tilted by a site reward | this work |
| `interaction` | `interacting_particles.py` | `af3_red` | Gaussian repulsive bias vs previously completed structures | OkazakiLab, doi 10.64898/2025.12.17.693105 |
| `interaction` | `interacting_particles.py` | `svgd` **(stale)** | deterministic transport through a kernel, no weights, no resampling | Liu & Wang [1608.04471](https://arxiv.org/abs/1608.04471) |

`svgd` is **stale and unsupported**. Nothing ships that selects it and no reported result uses it.
It is still present because its transport path shares state with `af3_red`, so removing it is a
sampler change rather than a deletion. Do not build on it.

**Families group by MECHANISM.** `search` explores over candidates or over the initial noise.
`guidance` steers each particle by a per-particle reward.`interaction` couples particles to each
other (`svgd`, through the kernel) or to earlier structures (`af3_red`). A config names the family in
`method.diffusion_side` and the algorithm in `method.diffusion_search.algorithm`, the internal `kind`
is derived from the algorithm and is never set by hand. The authoritative table is
`DIFFUSION_FAMILIES` in `diffusion_search.py`, imported by both `run_ablation` and the job preflight
so the runner and the validator cannot drift apart.

Every algorithm takes only a **scalar reward per particle**, so all three rewards (`ipsae`,
`cdr_rmsd`, `epitope`) compose with any of them. This document is centred on `tds` with the
**CDR-conformation** and **binding-site** rewards.

**Backend independence.** Nothing here imports a base model. `base_search.py` (shared geometry, rewards and
physicality) and the three family modules are pure numpy, the only model-specific code is the
adapter `models/protenix_hooks.py`, which supplies the denoiser, the noise schedule and the atom→token
map. Porting to another base model means writing a new adapter, not touching this package.

---

## 1. Notation

| symbol | meaning |
|---|---|
| `i = 1…N`, `t = 0…T−1` | particle index, denoise-step index |
| `x_t^i` | atom coordinates (Å) of particle `i` at step `t` |
| `σ_0 > … > σ_T = 0` | noise schedule |
| `D_θ(x, σ)` | base denoiser, returns the clean estimate `x̂₀` |
| `γ_t, η, λ_n` | EDM churn factor, step scale, noise scale |
| `σ̂_t = σ_t(1+γ_t)` | churned noise level |
| `s_t = λ_n\sqrt{σ̂_t^2 − σ_t^2}` | std of the noise actually injected this step |
| `ξ_t^i ∼ N(0, I)` | the step's standard normal draw |
| `R_t` | the step's random SE(3) augmentation (rotation, translation cancels below) |
| `a_i` | ancestor index of particle `i` after resampling (identity if no resample) |
| `r(·)`, `λ` | scalar reward, FK tilt temperature, the potential is `G = exp(λ r)` |
| `ε_t` | guidance coefficient (§2.1) |
| `Δ_t^i = ε_t R_t ∇_x r` | applied guidance displacement, **this is `dx` in the code** |
| `w_t^i`, `β` | particle importance weight, physicality penalty |

---

## 2. Hyperparameter reference

Every coefficient below appears in the master equation of §3. This section maps each `DiffusionSearchSpec`
field to its symbol, states what it controls, and records the production value, so the defaults can be
argued about rather than inherited.

**Reference only.** The experiments behind these defaults, the ESS pathology, and the `λ`/`ε_t` results
live in `doc/reward_research_2026-08-10.md`. Findings are kept there so there is one place to read them
and one place to revise them.

### 2.1 The twist `Δ_t^i`: how far a particle is moved

$$
\Delta_t^{i} \;=\, \underbrace{\varepsilon_t}_{\text{schedule}}\,R_t\,\nabla_x r,
\qquad
\varepsilon_t \;=\, \underbrace{c}_{\texttt{guidance\_scale}}\cdot\frac{\hat\sigma_t^{2}}{\sqrt{1+\hat\sigma_t^{2}}},
\qquad
\lVert\Delta_t^{i}\rVert \;\le\, \underbrace{M}_{\texttt{guidance\_max\_norm}}
$$

| field | production value | controls |
|---|---|---|
| `guidance_scale` `c` | 75 | step size of the twist. Multiplies `∇r` through `ε_t`. |
| `guidance_max_norm` `M` | 700 | per-particle clip on `‖Δ‖`. MEASURED: the mean per-particle norm is 219 to 291 against the 700 rail, and `saturated=True` (94% of events) means at least ONE particle clipped, not the typical one. The clip trims a tail, it does not set the magnitude. |
| `guidance_schedule` | `sigma2` | `ε_t ∝ σ²/√(1+σ²)`, i.e. the posterior uncertainty of `x̂₀`. `linear` is kept only for reproducing older runs and is ~10× too strong at `σ=0.1`. |

Because the clip binds, a `guidance_scale` sweep is **not** a magnitude sweep. Sweep `M` for magnitude
and `c` only to change which direction survives clipping.

### 2.2 The weight update: how a particle is selected

$$
\log w_{t+1}^{i}-\log w_{t}^{i} \;=\.
\underbrace{\lambda\big(r_t^{i}-r_{t-1}^{a_i}\big)}_{\text{(d) reward tilt}}
\;\underbrace{-\;\frac{\lVert\Delta_t^i\rVert^{2}+2s_t\langle\xi_t^i,\Delta_t^i\rangle}{2s_t^{2}}}_{\text{(e) importance correction}}
$$

| field | production value | controls |
|---|---|---|
| `lam` `λ` | 4.0 (cap) | tilt temperature: how much a reward **difference** becomes log-weight. |
| `lam_adaptive` | `true` | solve `λ` instead of fixing it. |
| `target_ess_frac` | 0.4 | the ESS the solver targets: largest `λ` with `ESS ≥ 0.4·N`. |
| `potential` | `difference` | `Δr` per step rather than the raw `r`. |

**`λ` is solved ONCE, from the reward alone.** `proposal_log_correction` is deliberately excluded,
because including it drove the solver to `λ = 0` and silently disabled steering altogether. That is a worse
failure, and it is documented at `models/protenix_hooks.py:~2581`. The cost is that term (e) contributes to ESS
unmanaged.

In production `λ` lands at **0.024 to 0.078**. Term (e) is applied at EVERY step, term (d) only at
scoring events. See the research doc for what each term contributes to ESS and which one separates a
working arm from a collapsing one.

### 2.3 Scheduling: how often the system is checked

| field | production value | controls |
|---|---|---|
| `schedule_start_frac` | 0.5 | first steered step. **Term (e) is applied EVERY step, so at 0.5 exactly 100 of 200 steps of correction accumulate into `logw` before anything measures ESS.** |
| `schedule_every` | 20 | steps between score / branch / resample events. |
| `ess_fraction` | 0.5 | resample when `ESS < 0.5·N`. |
| `resample_start_frac` / `_end_frac` | 0.2 / 0.8 | window in which resampling may fire at all. |
| `resample_scheme` | `systematic` | lower variance than the FK reference's `multinomial`. |

### 2.4 Reward shape

| field | production value | controls |
|---|---|---|
| `contact_weight` | 30 | weight of the contact-count term against the site distance. `r = −(‖μ_p−μ_e‖−d^*)^2 + w·C`, a **linear** combination, so this rotates the gradient direction rather than only scaling it. |
| `contact_r`, `contact_sigma` | 8.0, 2.0 | where the contact shell peaks and how wide it is. |
| `site_hinge`, `site_huber_delta` | `true`, 4.0 | one-sided hinge, no outward push on an already-close particle. |
| `site_dev_cap` | 12 | release cap: zero gradient beyond this, so a wrong patch stops dragging. |
| `umbrella_sweep` | 12 values from 7 to 17 | the `d^*` grid. Count matters more than the reward shape for pool spread. |

### 2.5 Knobs that interact

Three couplings are worth knowing before sweeping anything, because each makes a knob mean something
other than it appears to. The measurements behind them are in the research doc.

- **`guidance_scale` is not a magnitude knob** while the clip binds. Sweep `guidance_max_norm` for
  magnitude, `guidance_scale` only to change which direction survives clipping.
- **`guidance_scale` is not linear in its effect either**, because `∇r` grows as a particle sits further
  from its target, which partly cancels a reduction in `ε_t`.
- **Resampling more often trades away** the accumulated importance information that makes TDS
  asymptotically exact. It is a trade, and should be stated whenever it is made.

---

## 3. The method

### 3.1 What the twist does to the base model's score

The base sampler draws each reverse step from a Gaussian whose mean is set by the denoiser
`D_θ`, which is equivalent to following the model's own score

$$
s_\theta(x,\sigma) \;=\, \nabla_x \log p_\sigma(x) \;\approx\, \frac{D_\theta(x,\sigma)-x}{\sigma^{2}} .
$$

We do not want samples from `p_σ`. We want them from the reward-tilted target

$$
\pi_\sigma(x) \;\propto\, p_\sigma(x)\,G(x), \qquad G(x)=\exp\!\big(\lambda\, r(x)\big),
$$

whose score is simply the base score plus the reward gradient,

$$
\nabla_x \log \pi_\sigma \;=\, \underbrace{s_\theta(x,\sigma)}_{\text{base model}} \;+\, \underbrace{\lambda \nabla_x r(x)}_{\text{tilt}} .
$$

**The twist is how that extra term is applied without retraining anything.** Rather than editing the
denoiser, we shift the MEAN of the step's Gaussian by

$$
\Delta_t^{i} \;=\, \varepsilon_t\, R_t \nabla_x r\big(\hat x_0^{\,i}\big),
\qquad
\varepsilon_t \;=\, c\,\frac{\hat\sigma_t^{2}}{\sqrt{1+\hat\sigma_t^{2}}} ,
$$

evaluated on the denoiser's clean estimate `x̂₀` rather than on the noisy `x_t`, because the reward is
only defined on structures. `ε_t` is the posterior uncertainty of `x̂₀` given `x_t`, so the step is
large while the structure is undecided and goes to zero as it commits. `R_t` is the step's random
rotation, applied to the gradient because the reward is rotation invariant but its gradient is a
vector field.

**`c` AND `λ` ARE NOT INDEPENDENT IN THEORY, AND WE TREAT THEM AS IF THEY WERE.** The twist that
targets `π ∝ p·exp(λ r)` is, to first order in the same expansion,

$$
\Delta^{*} \;\approx\, \underbrace{\sigma^{2}}_{\text{transition covariance}} \cdot\, \lambda \,\nabla_x r ,
$$

i.e. the coefficient on the gradient IS `λ`. There is one free parameter, not two. Our `ε_t` carries
its own `c = guidance_scale` instead, so the proposal and the target it is supposed to twist toward
are set independently. At production values, `c = 75` against a solved `λ ≈ 0.05`, and MEASURED over the σ range the
sampler actually steers at (705 down to 19, guidance is inactive below that):

| `σ` | `ε_t` applied | matched `λσ²` | ratio |
|---|---|---|---|
| 704.6 | 52,800 | 24,800 | **2x** |
| 144.7 | 10,850 | 1,050 | 10x |
| 18.6 | 1,395 | 17 | 80x |

**They nearly agree at high noise and diverge as `σ` falls**, because `ε_t ∝ σ` at large `σ`
while the matched twist goes as `σ²`. So the defect is not a constant offset, it is a SHAPE
mismatch that grows through the trajectory.

**Read that factor as an order of magnitude, not a measurement.** `λ` carries units of `1/r`, and the
site reward is in `Å²`, while `√(1+σ²)` adds a dimensionless 1 to a quantity in `Å²`, so `c` and `λ`
are not strictly commensurable and the ratio depends on the reward's scale. The STRUCTURAL point does
not depend on the number: theory gives one coefficient and the code exposes two, so nothing keeps
them consistent.

This stays UNBIASED, because the correction below prices whatever shift is applied. What it costs is
VARIANCE: a proposal mismatched to its target is the textbook cause of weight degeneracy, which is
the ESS collapse reported in the research doc. The reference implementation pins the step to the
schedule and so never has this freedom. Matching `c` to `λ` removes the cause, whereas the
scheduling and clip knobs of §2.5 only reduce the symptom.

**Shifting a Gaussian's mean is exactly what makes the correction closed form.** The proposal is no
longer the model's own transition, so each particle carries an importance weight. For a mean shift
`Δ` on a Gaussian of std `s`, the log ratio is exact,

$$
\log\frac{p}{\tilde q}\;=\;-\,\frac{\lVert\Delta_t^i\rVert^{2}+2 s_t\langle\xi_t^i,\Delta_t^i\rangle}{2 s_t^{2}} ,
$$

with `ξ` the standard normal actually drawn. This term is what separates TDS from plain guidance.
Remove it and the twist is a biased heuristic. Keep it and the particle system targets `π` in the
large-`N` limit. At production settings it also dominates the log-weight variance at any single event,
and because it is applied at EVERY step it collapses ESS in every arm before the first check. That
collapse is not what separates a working arm from a failing one, though: recovery after the first
resample is, and recovery is governed by the reward. See the research doc for the measurements.

Putting the pieces together, one step for particle `i` is

$$
\boxed{\.
x_{t+1}^{i} \;=\, \tilde x_t^{i} \;+\, \eta\,(\sigma_{t+1}-\hat\sigma_t)\,
\frac{\tilde x_t^{i}-D_\theta(\tilde x_t^{i},\hat\sigma_t)}{\hat\sigma_t},
\qquad
\tilde x_t^{i} \;=\, \underbrace{R_t x_t^{a_i}}_{\text{resample}} + \underbrace{\Delta_t^{i}}_{\text{twist}} + \underbrace{s_t \xi_t^{i}}_{\text{churn}}
\;}
$$

**Degenerate limit, the correctness gate.** With `ε_t = 0` and `λ = 0` there is no twist and no
reweighting, and this collapses exactly to the stock EDM step, i.e. plain Best-of-N. Every added term
is a strict extension of the base sampler.

### 3.2 The rewards actually in use

Two, and they are one expression. Let `μ_p` be the paratope centroid, `μ_e` the centroid of the
candidate epitope patch, and `d*` the particle's umbrella target.

$$
r_i \;=\, \underbrace{-\big(\lVert \mu_p-\mu_e\rVert - d^{*}\big)^{2}}_{\text{binding-site term}} \;+\, \underbrace{w\, C_i}_{\text{contact term}}
$$

**The binding-site term** is a one-sided hinge on a single scalar, the paratope-to-epitope centroid
separation. One-sided means a particle already closer than `d*` receives no outward push, which measures
better than the two-sided harmonic. It constrains ONE degree of freedom of a six-DOF pose, so
a particle can satisfy it perfectly with zero native contacts. That is the gap the second term was
meant to close.

**The contact term** is a smooth surrogate for the number of interface contacts, a shell rather than
a well,

$$
C_i \;=\, \operatorname*{mean}_{p \in \text{paratope},\, e \in \text{epitope}} \exp\!\Big(-\frac{(r_{pe}-r_c)^{2}}{2\varsigma^{2}}\Big),
\qquad r_c = 8\,\text{Å},\;\varsigma = 2\,\text{Å} .
$$

A shell peaks at contact distance. A decaying well `exp(-r²/2ς²)` would be maximised at `r=0` and
would reward interpenetration, which is exactly the failure seen at `w = 100`.

**What the contact term does NOT do, measured.** It does not add register or orientation. `C` is a
mean over pairwise DISTANCES, so it is symmetric under permuting which paratope residue faces which
epitope residue, so a ROTATED antibody can outscore the native pose. What it genuinely raises is
fnat, while iRMS and LRMS degrade: it recovers native contact PAIRS without the geometry that holds
them. The measured rotation rates and per-term deltas are in the research doc. Set
`contact_weight = 0` to recover the pure site reward bit for bit.

**Umbrellas.** Particles are split round-robin across one umbrella per `d*`, and the multi-site case
takes the cross product with the candidate sites, so a cell is a `(umbrella, site)` pair. Windows must
OVERLAP: `d* = 2,4,6,8` measured worse than `1,2,3,4`, because an unreachable far umbrella just fills
with clones. Umbrella COUNT is the largest single lever on the pool spread that Best-of-N selects
from, more than the reward shape.

### 3.3 What the resampling does

Resampling is what turns a set of independently guided trajectories into a particle FILTER. Without
it the weights only record which particles happened to score well, and nothing reallocates effort
toward them.

At a resample event the normalised weights `w^i ∝ exp(logw^i)` are used to draw ancestors `a_i`, each
surviving particle is replaced by a copy of its ancestor, and `logw` is reset to zero. Three details
matter more than the mechanism.

**It is stratified by cell, not global.** Drawing globally would let one umbrella or one candidate
site clone over all the others within a few events, which destroys the hedge the sweep exists to
provide. `stratified_resample` preserves each `(umbrella, site)` cell's size exactly, so budget can
never migrate between binding hypotheses. Reserved, unsteered particles are given their own cell for
the same reason.

**It is systematic, not multinomial.** Lower variance for the same expected counts. The offset is a
deterministic function of the step index, so a rerun of the same arm resamples identically.

**Its usefulness depends entirely on ESS.** Resampling at `ESS = N` is a no-op, and resampling at
`ESS = 1` clones a single particle `N` times, which is worse than not resampling at all because it
destroys genealogical diversity while looking healthy in the output spread. The measured production
regime is much closer to the second case than the first, which is why §2.5 flags the scheduling
knobs as coupled.

---
