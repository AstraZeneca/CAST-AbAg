"""InterfaceScoreVerifier, computes the interface and optionally the coord / JSON / pool signals.

For each sample it loads Ca coords + pLDDT (and the PAE when available) and, in one pass over the
target, attaches the signal groups that are switched on:
  * PAE interface scores ipSAE / pDockQ2 / LIS / iLIS   (``compute_interface``, default True),
  * coordinate-only signals i_plddt / n_interface_* / clash_ca (``compute_coord``, default False),
  * summary-JSON signals min/avg_pair_iptm (``compute_pair_iptm``, default False),
  * pool consensus_density (``compute_consensus``, default False).

Only the PAE scores are on by default because they are the sole time-sensitive ones: the PAE is
reclaimed after inference, so they must be computed while it is fresh. The other three need only
the cif and summary JSON, which survive, so they are computed post-hoc on demand
(``evaluate.py --verifiers``). Native confidence-head fields (ranking_score, iptm, ptm, plddt,
gpde, disorder, has_clash) ride along from ``iter_predictions`` and are always saved.

``rank_by`` selects only which score is the primary Best-of-N key. The PAE is optional: where it
was reclaimed, the coord / JSON / consensus groups still compute from the cif. Backend-agnostic:
any :class:`models.base.ModelBackend` with ``iter_predictions`` + ``load_pae_and_coords`` works.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from models.base import ModelBackend, PredictionSample

from .biophysical import compute_biophys
from .cdr import cdr_residues_from_cif, compute_cdr_paratope_4a_heavy
from .cif_io import load_ca_coords_chains
from .consensus import DEFAULT_DENSITY_SIGMA_A, DEFAULT_RMSD_THRESHOLD_A, consensus_scores
from .ipsae import (
    DEFAULT_DIST_CUTOFF_A,
    DEFAULT_PAE_CUTOFF_A,
    DEFAULT_PLDDT_CUTOFF,
    compute_ipsae,
    dump_result as dump_ipsae_result,
)
from .interface_scores import (
    compute_coord_signals,
    compute_interface_scores,
    compute_summary_pair_signals,
)

# rank_by values this verifier can serve as the primary key.
INTERFACE_SCORES = frozenset({"ipsae", "pdockq2", "lis", "ilis"})


class InterfaceScoreVerifier:
    """Compute ipSAE + pDockQ2 + LIS + iLIS + coord/JSON signals (+ optional consensus)."""

    def __init__(
        self,
        backend: ModelBackend,
        predictions_dir: str | Path,
        rank_by: str = "pdockq2",
        pae_cutoff: float = DEFAULT_PAE_CUTOFF_A,
        dist_cutoff: float = DEFAULT_DIST_CUTOFF_A,
        plddt_cutoff: float = DEFAULT_PLDDT_CUTOFF,
        fallback_rank_by: str = "ranking_score",
        results_dir: Optional[str | Path] = None,
        antibody_chains: Optional[List[str]] = None,
        compute_interface: bool = True,
        compute_coord: bool = False,
        compute_pair_iptm: bool = False,
        compute_consensus: bool = False,
        compute_biophys: bool = False,
        compute_cdr: bool = False,
        hmmer_path: Optional[str] = None,
        rmsd_threshold: float = DEFAULT_RMSD_THRESHOLD_A,
        density_sigma: float = DEFAULT_DENSITY_SIGMA_A,
    ):
        self.backend = backend
        self.predictions_dir = Path(predictions_dir)
        self.rank_by = rank_by
        self.pae_cutoff = pae_cutoff
        self.dist_cutoff = dist_cutoff
        self.plddt_cutoff = plddt_cutoff
        self.fallback_rank_by = fallback_rank_by
        self.results_dir = Path(results_dir) if results_dir else None
        self.antibody_chains = antibody_chains
        # per-group gates: compute only the selected signal groups (see evaluate.py --verifiers).
        self.compute_interface = compute_interface   # PAE scores ipsae/pdockq2/lis/ilis
        self.compute_coord = compute_coord           # i_plddt / n_interface_* / clash_ca
        self.compute_pair_iptm = compute_pair_iptm    # JSON min/avg_pair_iptm
        self.compute_consensus = compute_consensus   # pool consensus (fraction) + consensus_density
        self.compute_biophys = compute_biophys       # heavy-atom BSA / salt bridges / hbonds
        self.compute_cdr = compute_cdr               # ANARCI paratope_cdr_fraction (once per target)
        self.hmmer_path = hmmer_path
        self.rmsd_threshold = rmsd_threshold
        self.density_sigma = density_sigma

    def iter_samples(self, target_id: str) -> List[PredictionSample]:
        samples = self.backend.iter_predictions(self.predictions_dir, target_id)
        coords_pool: List = []          # per-sample coords for the pool consensus
        chains_pool: List = []
        for s in samples:
            pae, coords, chains, plddt, err = self.backend.load_pae_and_coords(
                self.predictions_dir, target_id, s.seed, s.sample_idx,
            )
            if coords is None or chains is None:
                if s.cif_path:
                    coords, chains, plddt = load_ca_coords_chains(s.cif_path)
                pae = None
            if self.compute_consensus:          # only build the pool when it is needed
                coords_pool.append(coords)
                chains_pool.append(chains)
            if coords is None or chains is None:
                s.extras["interface_score_error"] = err or "no coords (cif unreadable)"
                s.notes = (s.notes + f"; interface_fallback={self.fallback_rank_by}").lstrip("; ")
                continue

            if self.compute_interface and pae is not None:   # PAE scores (need the residue PAE)
                ir = compute_ipsae(
                    pae, coords, chains, plddt=plddt,
                    pae_cutoff=self.pae_cutoff, dist_cutoff=self.dist_cutoff,
                    plddt_cutoff=self.plddt_cutoff, antibody_chains=self.antibody_chains,
                )
                isc = compute_interface_scores(
                    pae, coords, chains, plddt=plddt, antibody_chains=self.antibody_chains,
                )
                s.ipsae = ir.ipsae
                s.pdockq2 = isc.pdockq2
                s.lis = isc.lis
                s.ilis = isc.ilis
                if self.results_dir is not None:
                    tag = f"{target_id}__seed{s.seed}__s{s.sample_idx}"
                    dump_ipsae_result(ir, self.results_dir / "metrics" / "ipsae" / f"{tag}.json")
                if isc.error and ir.error:
                    s.notes = (s.notes + f"; interface_err={isc.error}").lstrip("; ")
            elif self.compute_interface and err:
                s.extras["interface_score_error"] = f"{err} (coord+json signals only, no PAE)"

            if self.compute_coord:              # coordinate-only signals (cif alone)
                cs = compute_coord_signals(
                    coords, chains, plddt=plddt, antibody_chains=self.antibody_chains,
                )
                s.i_plddt = cs.i_plddt
                s.n_interface_residues = cs.n_interface_residues
                s.n_interface_contacts = cs.n_interface_contacts
                s.clash_ca = cs.clash_ca
            if self.compute_pair_iptm:          # summary-JSON free reads (chain_pair_iptm in extras),
                sp = compute_summary_pair_signals(   # ab-ag masked using the loaded chains in matrix order
                    s.extras, list(dict.fromkeys(chains)), antibody_chains=self.antibody_chains,
                )
                s.min_pair_iptm = sp.min_pair_iptm
                s.avg_pair_iptm = sp.avg_pair_iptm
            if self.compute_biophys and s.cif_path:   # heavy-atom bundle (reads the cif directly)
                bp = compute_biophys(s.cif_path, antibody_chains=self.antibody_chains)
                if bp.error is None:
                    s.interface_bsa = bp.interface_bsa
                    s.bsa_hydrophobic_frac = bp.bsa_hydrophobic_frac
                    s.n_salt_bridges = bp.n_salt_bridges
                    s.n_hbond_contacts = bp.n_hbond_contacts

        if self.compute_consensus:
            frac, dens = consensus_scores(
                coords_pool, chains_pool, antibody_chains=self.antibody_chains,
                rmsd_threshold=self.rmsd_threshold, density_sigma=self.density_sigma,
            )
            for s, f, d in zip(samples, frac, dens):
                s.consensus = f
                s.consensus_density = d

        if self.compute_cdr:
            # IMGT-number the antibody ONCE per target (sequence is shared by all samples).
            cdr_map = {}
            for s in samples:
                if s.cif_path:
                    cdr_map = cdr_residues_from_cif(s.cif_path, self.antibody_chains, hmmer_path=self.hmmer_path)
                    if cdr_map:
                        break
            if cdr_map:
                for s in samples:
                    if s.cif_path:
                        cd = compute_cdr_paratope_4a_heavy(s.cif_path, self.antibody_chains, cdr_map)
                        if cd.error is None:
                            s.paratope_cdr_fraction = cd.paratope_cdr_fraction
                            s.n_cdr_contacts = cd.n_cdr_contacts
                            s.n_paratope_residues = cd.n_paratope_residues
        return samples
