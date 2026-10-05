"""Boltz-Perturb (Token Conditioning / Token Bias Perturbation) on the denoiser's conditioning.

Ported from MSDLLCpapers/boltz-perturb, ``src/boltz/model/modules/diffusionv2.py``
(``make_s_noise_fn`` and ``make_token_bias_noise_fn``), mirroring that source in order and
arithmetic. Spec fields are ``boltz_perturb_*`` on ``DiffusionSearchSpec``; the hook is
``models/protenix_hooks.py``, which refuses ``enabled`` without a track rather than running an
unperturbed pool under this method's label.

Port decisions, the only departures from the source, each forced by the backbone:

  * **ligand -> antibody.** Their masks key off ``ligand_token_mask`` on a protein-ligand complex.
    The analogue here is the ANTIBODY, with the antigen as "protein", so ``mode`` reads
    ``cross_only`` = the antibody x antigen block, ``ll_only`` = antibody x antibody,
    ``cross_and_ll`` = every pair touching the antibody, ``all`` = everything.
  * **Tensors.** TCP perturbs the single/per-token representation and TBP the pair representation.
    Here those are ``s_trunk`` and ``pair_z``; on theirs, token embeddings and the token transition
    bias. Same roles, different names.
  * **TADS.** Their schedule lives in the sampling loop and keys off the sampler's own noise
    variance, which is not exposed at this hook point, so :func:`tads_noise_scale` implements the
    time-annealed decay on the schedule fraction instead. This is the one piece that is
    behaviourally equivalent rather than line-identical, isolated in its own function.

Spec defaults are the source's own: ``noise_scale`` 0.3, ``psi`` 0.7, ``rescale`` and
``norm_proj`` True, ``mode`` "cross_and_ll".

UPSTREAM LICENCE. The two functions this file ports are MIT, and the notice travels with the port::

    MIT License
    Copyright (c) 2026 Merck & Co., Inc., Rahway, NJ, USA and its affiliates.
    All rights reserved.
    Contributors: Hyeyun Jung (MSD), BoRam Lee (MSD), Alan C. Cheng (MSD)

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software
    and associated documentation files (the "Software"), to deal in the Software without
    restriction, including without limitation the rights to use, copy, modify, merge, publish,
    distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or
    substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
    BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
    NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
    DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Boltz-Perturb is itself built on Boltz-2 (https://github.com/jwohlwend/boltz), also MIT, whose
terms apply as well. Nothing from Boltz-2 is vendored here, only the perturbation arithmetic above.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

_EPS = 1e-6            # the source's epsilon, used verbatim in every division


@dataclass(frozen=True)
class BoltzPerturbSpec:
    """Boltz-Perturb settings. Field defaults are the reference implementation's own."""

    enabled: bool = False
    # TCP: perturb the single / per-token conditioning (their `s`).
    tcp: bool = False
    # TBP: perturb the pair conditioning / attention bias (their `token_trans_bias`).
    tbp: bool = False
    noise_scale: float = 0.3          # their make_token_bias_noise_fn default
    psi: float = 0.7                  # their default. Higher = MORE of the rescaled noise.
    rescale: bool = True              # their default
    norm_proj: bool = True            # their default
    mode: str = "cross_and_ll"        # cross_only | cross_and_ll | all | ll_only
    # TADS: anneal the noise scale over sampling. tads=False applies noise_scale flat, which is what
    # the source does when the schedule is disabled.
    tads: bool = True
    tads_floor: float = 0.0           # noise scale at the end of sampling


def tads_noise_scale(noise_scale: float, t: float, spec: "BoltzPerturbSpec") -> float:
    """Time-annealed noise scale. ``t`` is the schedule fraction, 1 at the start -> 0 at the end."""
    if not spec.tads:
        return float(noise_scale)
    f = min(max(float(t), 0.0), 1.0)
    return float(spec.tads_floor) + (float(noise_scale) - float(spec.tads_floor)) * f


def pair_mask_for_mode(ab_token_mask, mode: str):
    """(N, N) boolean pair mask, mirroring the source's `mode` branch with ligand -> antibody."""
    lig = np.asarray(ab_token_mask, dtype=bool)
    prot = ~lig
    if mode == "cross_only":
        return (prot[:, None] & lig[None, :]) | (lig[:, None] & prot[None, :])
    if mode == "cross_and_ll":
        return lig[:, None] | lig[None, :]
    if mode == "all":
        return np.ones((lig.size, lig.size), dtype=bool)
    if mode == "ll_only":
        return lig[:, None] & lig[None, :]
    raise ValueError("Unknown mode: %r" % (mode,))


def tcp_perturb(s, ab_token_mask, spec: BoltzPerturbSpec, t: float = 1.0, rng=None):
    """Token Conditioning Perturbation. Verbatim port of `s_noise_fn`; `s` is (..., N, D)."""
    s = np.asarray(s, dtype=np.float64)
    m = np.asarray(ab_token_mask, dtype=bool)
    ns = tads_noise_scale(spec.noise_scale, t, spec)
    if ns <= 0:
        return s
    rng = rng or np.random.default_rng()

    # Step 1. Generate Noise -- scaled by the std of the WHOLE tensor, then zeroed outside the mask.
    # (The source takes s.std() over everything, not over the masked region. TBP differs, see below.)
    noise = ns * s.std() * rng.standard_normal(s.shape)
    noise[..., ~m, :] = 0.0

    # Step 2. Add noise to s. NO SHRINK -- this is the step CADS does differently.
    s_tilde = s + noise
    s_prime = s_tilde

    # Step 3. Rescale s_tilde to match original mean and std on the masked tokens.
    if spec.rescale:
        mu_o = s[..., m, :].mean(); sg_o = s[..., m, :].std()
        mu_t = s_tilde[..., m, :].mean(); sg_t = s_tilde[..., m, :].std()
        s_prime = s.copy()
        s_prime[..., m, :] = ((s_tilde[..., m, :] - mu_t) / (sg_t + _EPS)) * sg_o + mu_o
        s_prime[..., ~m, :] = s[..., ~m, :]

    # Step 4. Interpolate between ORIGINAL s and the perturbed s_prime. Toward the ORIGINAL --
    # the single most consequential difference from CADS, which mixes toward the noised tensor.
    s_hat = s.copy()
    s_hat[..., m, :] = spec.psi * s_prime[..., m, :] + (1.0 - spec.psi) * s[..., m, :]
    s_hat[..., ~m, :] = s[..., ~m, :]

    # Step 5. Norm-preserving projection: each masked token's embedding keeps its original L2 norm.
    if spec.norm_proj:
        n_o = np.linalg.norm(s[..., m, :], axis=-1, keepdims=True)
        n_p = np.linalg.norm(s_hat[..., m, :], axis=-1, keepdims=True)
        s_hat[..., m, :] = (s_hat[..., m, :] / (n_p + _EPS)) * n_o
    return s_hat


def tbp_perturb(bias, ab_token_mask, spec: BoltzPerturbSpec, t: float = 1.0, rng=None):
    """Token Bias Perturbation. Verbatim port of `token_bias_noise_fn`; `bias` is (..., N, N, D)."""
    bias = np.asarray(bias, dtype=np.float64)
    ns = tads_noise_scale(spec.noise_scale, t, spec)
    if ns <= 0:
        return bias
    rng = rng or np.random.default_rng()
    pm = pair_mask_for_mode(ab_token_mask, spec.mode)

    # local_std is taken over the TARGETED PAIRS ONLY -- unlike TCP, which uses the whole tensor.
    # Keeping this asymmetry is deliberate: it is what the source does.
    local_std = bias[..., pm, :].std()

    # Step 1. Add noise, masked to the targeted pairs.
    noise = rng.standard_normal(bias.shape)
    bias_tilde = bias.copy()
    bias_tilde[..., pm, :] = bias[..., pm, :] + ns * local_std * noise[..., pm, :]

    # Step 2. Rescale to the original mean/std on the targeted pairs.
    if spec.rescale:
        o = bias[..., pm, :]; p = bias_tilde[..., pm, :]
        bias_prime = bias_tilde.copy()
        bias_prime[..., pm, :] = ((p - p.mean()) / (p.std() + _EPS)) * o.std() + o.mean()
    else:
        bias_prime = bias_tilde

    # Step 3. Interpolate with the ORIGINAL bias.
    bias_hat = bias.copy()
    bias_hat[..., pm, :] = spec.psi * bias_prime[..., pm, :] + (1.0 - spec.psi) * bias[..., pm, :]

    # Step 4. Norm projection: each targeted pair vector keeps its original L2 norm.
    if spec.norm_proj:
        o_n = np.linalg.norm(bias[..., pm, :], axis=-1, keepdims=True)
        p_n = np.linalg.norm(bias_hat[..., pm, :], axis=-1, keepdims=True)
        bias_hat[..., pm, :] = bias_hat[..., pm, :] / (p_n + _EPS) * o_n
    return bias_hat


def tcp_perturb_torch(s, ab_token_mask, spec: BoltzPerturbSpec, t: float = 1.0, generator=None):
    """TCP on a torch tensor. Mirrors :func:`tcp_perturb` step for step."""
    import torch
    ns = tads_noise_scale(spec.noise_scale, t, spec)
    if ns <= 0:
        return s
    dt, dev = s.dtype, s.device
    sf = s.float()
    m = torch.as_tensor(ab_token_mask, dtype=torch.bool, device=dev)
    noise = torch.randn(sf.shape, device=dev, dtype=torch.float32, generator=generator)
    noise = ns * sf.std() * noise
    noise[..., ~m, :] = 0.0
    s_tilde = sf + noise                                    # add, never shrink
    s_prime = s_tilde
    if spec.rescale:
        o = sf[..., m, :]; p = s_tilde[..., m, :]
        s_prime = sf.clone()
        s_prime[..., m, :] = ((p - p.mean()) / (p.std() + _EPS)) * o.std() + o.mean()
        s_prime[..., ~m, :] = sf[..., ~m, :]
    s_hat = sf.clone()
    s_hat[..., m, :] = spec.psi * s_prime[..., m, :] + (1.0 - spec.psi) * sf[..., m, :]  # toward ORIGINAL
    s_hat[..., ~m, :] = sf[..., ~m, :]
    if spec.norm_proj:
        n_o = sf[..., m, :].norm(dim=-1, keepdim=True)
        n_p = s_hat[..., m, :].norm(dim=-1, keepdim=True)
        s_hat[..., m, :] = (s_hat[..., m, :] / (n_p + _EPS)) * n_o
    return s_hat.to(dt)


def tbp_perturb_torch(bias, ab_token_mask, spec: BoltzPerturbSpec, t: float = 1.0, generator=None):
    """TBP on a torch tensor. Mirrors :func:`tbp_perturb` step for step."""
    import torch
    ns = tads_noise_scale(spec.noise_scale, t, spec)
    if ns <= 0:
        return bias
    dt, dev = bias.dtype, bias.device                       # float32 throughout, see tcp_perturb_torch
    bf = bias.float()
    pm = torch.as_tensor(pair_mask_for_mode(ab_token_mask, spec.mode), dtype=torch.bool, device=dev)
    local_std = bf[..., pm, :].std()                        # targeted pairs only, unlike TCP
    noise = torch.randn(bf.shape, device=dev, dtype=torch.float32, generator=generator)
    bias_tilde = bf.clone()
    bias_tilde[..., pm, :] = bf[..., pm, :] + ns * local_std * noise[..., pm, :]
    if spec.rescale:
        o = bf[..., pm, :]; p = bias_tilde[..., pm, :]
        bias_prime = bias_tilde.clone()
        bias_prime[..., pm, :] = ((p - p.mean()) / (p.std() + _EPS)) * o.std() + o.mean()
    else:
        bias_prime = bias_tilde
    bias_hat = bf.clone()
    bias_hat[..., pm, :] = spec.psi * bias_prime[..., pm, :] + (1.0 - spec.psi) * bf[..., pm, :]
    if spec.norm_proj:
        o_n = bf[..., pm, :].norm(dim=-1, keepdim=True)
        p_n = bias_hat[..., pm, :].norm(dim=-1, keepdim=True)
        bias_hat[..., pm, :] = bias_hat[..., pm, :] / (p_n + _EPS) * o_n
    return bias_hat.to(dt)



_OK = [True]           # self-check accumulator, see the SELF-CHECK section at the end


def _chk(name, cond, extra=""):
    """One self-check assertion. Prints rather than raises so a run reports EVERY failure, not the
    first -- a partial picture is how a second defect hides behind the one you just fixed."""
    _OK[0] &= bool(cond)
    print(("  PASS " if cond else "  FAIL ") + name + ("  " + extra if extra else ""))


def self_check() -> int:
    import torch
    rng = np.random.default_rng(0)
    n_ab, n_ag, d = 8, 12, 8               # 8 antibody tokens, 12 antigen, 8 channels
    n = n_ab + n_ag
    ab = np.zeros(n, bool)
    ab[:n_ab] = True
    s = torch.as_tensor(rng.standard_normal((n, d)))
    z = torch.as_tensor(rng.standard_normal((n, n, d)))
    sp = BoltzPerturbSpec(enabled=True, tcp=True, tbp=True)

    print("== mask arithmetic (ligand -> antibody) ==")
    for mode, want in (("cross_only", 2 * n_ab * n_ag),          # both off-diagonal blocks
                       ("cross_and_ll", n * n - n_ag * n_ag),    # everything touching the antibody
                       ("all", n * n),
                       ("ll_only", n_ab * n_ab)):
        got = int(pair_mask_for_mode(ab, mode).sum())
        _chk("mode=%s" % mode, got == want, "%d == %d" % (got, want))

    print("== TADS anneal ==")
    _chk("t=1 -> full scale", abs(tads_noise_scale(0.3, 1.0, sp) - 0.3) < 1e-12)
    _chk("t=0.5 -> half", abs(tads_noise_scale(0.3, 0.5, sp) - 0.15) < 1e-12)
    _chk("t=0 -> floor", abs(tads_noise_scale(0.3, 0.0, sp) - 0.0) < 1e-12)
    _chk("tads=False is flat",
        abs(tads_noise_scale(0.3, 0.0, BoltzPerturbSpec(tads=False)) - 0.3) < 1e-12)

    print("== TCP (single track), torch path ==")
    out = tcp_perturb_torch(s, ab, sp, t=1.0, generator=torch.Generator().manual_seed(7))
    _chk("antigen tokens untouched", torch.allclose(out[n_ab:], s[n_ab:]))
    _chk("antibody tokens moved", not torch.allclose(out[:n_ab], s[:n_ab]))
    _chk("norm_proj preserves L2",
        torch.allclose(out[:n_ab].norm(dim=-1), s[:n_ab].norm(dim=-1), atol=1e-5))
    _chk("psi=0 is the IDENTITY, not the noised tensor (this is what CADS does differently)",
        torch.allclose(tcp_perturb_torch(s, ab, BoltzPerturbSpec(psi=0.0), t=1.0,
                                         generator=torch.Generator().manual_seed(7)), s, atol=1e-5))
    _chk("noise_scale=0 returns the input untouched",
        tcp_perturb_torch(s, ab, BoltzPerturbSpec(noise_scale=0.0), t=1.0) is s)
    _chk("t=0 under TADS returns the input untouched", tcp_perturb_torch(s, ab, sp, t=0.0) is s)

    print("== TBP (pair track), torch path ==")
    o = tbp_perturb_torch(z, ab, sp, t=1.0, generator=torch.Generator().manual_seed(7))
    pm = torch.as_tensor(pair_mask_for_mode(ab, sp.mode))
    _chk("out-of-mask pairs untouched", torch.allclose(o[~pm], z[~pm]))
    _chk("in-mask pairs moved", not torch.allclose(o[pm], z[pm]))
    _chk("norm_proj preserves L2", torch.allclose(o[pm].norm(dim=-1), z[pm].norm(dim=-1), atol=1e-5))

    print("== BFLOAT16 INPUT (what the denoiser actually hands us) ==")
    sb, zb = s.bfloat16(), z.bfloat16()
    ob = tcp_perturb_torch(sb, ab, sp, t=1.0, generator=torch.Generator().manual_seed(7))
    _chk("TCP returns bfloat16 unchanged in dtype", ob.dtype == torch.bfloat16, str(ob.dtype))
    _chk("TCP bf16 leaves antigen tokens untouched", torch.equal(ob[n_ab:], sb[n_ab:]))
    zbo = tbp_perturb_torch(zb, ab, sp, t=1.0, generator=torch.Generator().manual_seed(7))
    _chk("TBP returns bfloat16 unchanged in dtype", zbo.dtype == torch.bfloat16, str(zbo.dtype))
    _chk("TBP bf16 leaves out-of-mask pairs untouched", torch.equal(zbo[~pm], zb[~pm]))
    _chk("bf16 result tracks the float32 result",
        (ob.float() - out).abs().max().item() < 0.05,
        "max |bf16 - fp32| = %.4f" % (ob.float() - out).abs().max().item())

    print("== psi is monotone: more psi, further from the original ==")
    ds = []
    for psi in (0.0, 0.25, 0.5, 0.75, 1.0):
        p = tcp_perturb_torch(s, ab, BoltzPerturbSpec(psi=psi), t=1.0,
                              generator=torch.Generator().manual_seed(3))
        ds.append(float((p[:n_ab] - s[:n_ab]).abs().mean()))
    _chk("monotone in psi", all(a <= b + 1e-9 for a, b in zip(ds, ds[1:])),
        " -> ".join("%.4f" % x for x in ds))

    print("== numpy and torch paths agree on displacement (200 draws each) ==")
    sn = np.asarray(s)
    m_np = float(np.mean([np.abs(tcp_perturb(sn, ab, sp, t=1.0, rng=np.random.default_rng(i))[:n_ab]
                                 - sn[:n_ab]).mean() for i in range(200)]))
    m_pt = float(np.mean([float((tcp_perturb_torch(s, ab, sp, t=1.0,
                                                   generator=torch.Generator().manual_seed(i))[:n_ab]
                                 - s[:n_ab]).abs().mean()) for i in range(200)]))
    _chk("mean displacement matches", abs(m_np - m_pt) < 0.05 * max(m_np, m_pt),
        "numpy %.4f vs torch %.4f" % (m_np, m_pt))

    print("\nRESULT: " + ("ALL PASS" if _OK[0] else "FAILURES ABOVE"))
    return 0 if _OK[0] else 1


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(self_check())
