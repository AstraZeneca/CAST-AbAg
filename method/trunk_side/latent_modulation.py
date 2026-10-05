"""Latent modulation of the trunk pair representation z_pre. Two families:

- ``rescale``     Pair-Representation Scaling (Suzuki et al., bioRxiv 2026.01.23.701250):
                  isotropic ``z <- (1+beta)*z``, one scalar beta swept over a symmetric grid.
                  A forward hook, applied at every recycle.
- ``conformets``  ConforNets (Lee, Kalicki, Jeon, Qabel, Fadini and AlQuraishi, arXiv:2604.18559):
                  a channel-wise affine transform of z_pre, ``phi(z) = z W^T + b`` with
                  ``W in R^{c x c}``, ``b in R^c`` and c = c_z, initialised to identity plus small
                  Gaussian noise.

ConforNets, as the paper specifies it and as :func:`optimize_conformets` implements it:

  * ``k_nets`` (default 2) nets are instantiated and their (W, b) jointly optimised by Adam to
    MAXIMISE a pairwise diversity loss across the k decoded outputs, selected by ``spec.objective``:
      - distogram-CDF:  ``L_dist  = - sum_{i!=j} MSE(CDF^(i), CDF^(j))``  (from distogram logits)
      - coordinate:     ``L_coord = - sum_{i!=j} MSE(Align(X^(i)), X^(j))``  (one-step-denoised)
  * Applied ONCE before the FINAL Pairformer pass, with gradients enabled only in that pass.
    Earlier recycles run unmodulated, so the forward hook must modulate the last pass only.
  * Optimiser recipe: Adam, ``lr`` 0.001 halved every ``lr_halve_every`` (5) steps,
    ``grad_clip_norm`` 10 per net, ``n_steps`` 20, snapshotting at ``snapshot_steps``
    {5, 10, 15, 20}. That is ``k_nets * len(snapshot_steps)`` distinct latents per seed.

``conformets`` is a GRADIENT method, not a cheap input swap: it needs a differentiable path from
z_pre through the final Pairformer and a decoder to the diversity loss.
:func:`optimize_conformets` takes the ``decode_fn`` the backend hook supplies. Only ``rescale`` is
a pure forward hook.

This module holds the serialisable :class:`LatentModulationSpec`, the diversity loss
:func:`conformets_diversity_loss`, the optimisation loop :func:`optimize_conformets`, and
:func:`init_conformet` for the identity-plus-noise init.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple

import numpy as np


ModulationKind = Literal["none", "rescale", "conformets"]

# Protenix / OpenFold3 pair-rep channel dimension (c_z). ConforNets acts on this.
DEFAULT_CZ = 128


@dataclass(frozen=True)
class LatentModulationSpec:
    """One recipe for modulating z_pre (before Pairformer). z_pre only, no z_post."""

    kind: ModulationKind = "none"
    seed: int = 0

    beta: float = 0.0                           # z <- (1+beta) * z_pre , 0 == stock inference
    beta_grid: List[float] = field(default_factory=lambda: [-0.6, -0.3, 0.0, 0.3, 0.6])  # Boltz-2 grid

    # --- ConforNets: full channel affine phi(z) = z Wᵀ + b, init identity ---
    channel_dim: int = DEFAULT_CZ               # c_z
    init_noise_std: float = 1e-3                # Gaussian symmetry-break on W, b at init
    # Unsupervised diversity-optimisation hyperparameters (consumed by the hook):
    k_nets: int = 2                             # jointly-optimised ConforNets
    objective: Literal["distogram_cdf", "coordinate"] = "distogram_cdf"  # diversity loss (paper)
    lr: float = 1e-3                            # Adam lr
    lr_halve_every: int = 5                     # halve lr every N steps
    grad_clip_norm: float = 10.0                # per-net gradient clip
    n_steps: int = 20                           # optimisation steps
    snapshot_steps: List[int] = field(default_factory=lambda: [5, 10, 15, 20])
    variant_index: int = 0

    def label(self) -> str:
        if self.kind == "none":
            return "none"
        if self.kind == "rescale":
            return f"rescale@beta{self.beta:+.2f}"
        if self.kind == "conformets":
            return (f"conformets@k{self.k_nets}_lr{self.lr:g}_steps{self.n_steps}"
                    f"_snap{self.variant_index}_s{self.seed}")
        raise ValueError(f"Unknown modulation kind: {self.kind!r}")

    def resolve_for_seed(self, seed_i: int) -> "LatentModulationSpec":
        """Return the concrete modulation for trunk-latent variant ``seed_i``."""
        import dataclasses as _dc
        if self.kind == "rescale":
            grid = self.beta_grid or [0.0]
            return _dc.replace(self, beta=float(grid[seed_i % len(grid)]))
        if self.kind == "conformets":
            return _dc.replace(self, variant_index=seed_i)
        return _dc.replace(self, seed=self.seed + seed_i)


def init_conformet(spec: LatentModulationSpec, net_index: int = 0,
                   c: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Reference identity+noise initialisation of one ConforNet's (W, b)."""
    c = int(c if c is not None else spec.channel_dim)
    rng = np.random.default_rng((spec.seed, net_index))
    W = np.eye(c) + spec.init_noise_std * rng.standard_normal((c, c))
    b = spec.init_noise_std * rng.standard_normal(c)
    return W, b


def conformets_diversity_loss(outputs, objective: str = "distogram_cdf"):
    """The ConforNets **negative** diversity loss to MINIMISE (paper §, maximise diversity)."""
    import torch
    k = len(outputs)
    if objective == "distogram_cdf":
        cdfs = [torch.cumsum(torch.softmax(o.float(), dim=-1), dim=-1) for o in outputs]  # fp32 for stable loss
        terms = [torch.mean((cdfs[i] - cdfs[j]) ** 2) for i in range(k) for j in range(i + 1, k)]
    elif objective == "coordinate":
        terms = [torch.mean((outputs[i].float() - outputs[j].float()) ** 2) for i in range(k) for j in range(i + 1, k)]
    else:
        raise ValueError(f"unknown ConforNets objective: {objective!r}")
    # Paper writes Σ_{i≠j} (ordered pairs). MSE is symmetric so that is 2·Σ_{i<j}. Keep the ×2
    # so the loss magnitude matches the paper (their lr=0.001 was tuned against it).
    return -2.0 * torch.stack(terms).sum()    # maximise pairwise MSE ⇒ minimise its negative


def optimize_conformets(z_pre, decode_fn, spec: LatentModulationSpec):
    """Reference ConforNets unsupervised optimisation (paper-faithful)."""
    import torch
    c = z_pre.shape[-1]                                         # RUNTIME channel dim (Protenix-v2: 256)
    work_dtype = torch.float32                                  # fp32 (W,b) → stable Adam even if z_pre is bf16
    nets = []
    for i in range(spec.k_nets):
        W_np, b_np = init_conformet(spec, net_index=i, c=c)     # I + noise, per-net seed
        W = torch.tensor(W_np, dtype=work_dtype, device=z_pre.device, requires_grad=True)
        b = torch.tensor(b_np, dtype=work_dtype, device=z_pre.device, requires_grad=True)
        nets.append((W, b))
    opt = torch.optim.Adam([p for W, b in nets for p in (W, b)], lr=spec.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=spec.lr_halve_every, gamma=0.5)

    snapshots = []
    flat = z_pre.reshape(-1, c)
    for step in range(1, spec.n_steps + 1):
        opt.zero_grad()
        # affine in the model dtype (grad flows back to the fp32 W,b through the cast)
        outs = [decode_fn((flat @ W.to(flat.dtype).T + b.to(flat.dtype)).reshape(z_pre.shape))
                for W, b in nets]
        loss = conformets_diversity_loss(outs, spec.objective)
        loss.backward()
        for W, b in nets:                                       # clip per-net (paper)
            torch.nn.utils.clip_grad_norm_([W, b], spec.grad_clip_norm)
        opt.step()
        sched.step()
        if step in spec.snapshot_steps:
            snapshots.extend([(W.detach().cpu().numpy(), b.detach().cpu().numpy()) for W, b in nets])
    return snapshots                                            # k_nets × |snapshot_steps| (W,b) pairs
