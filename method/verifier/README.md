# `method/verifier`, Best-of-N verifiers

A verifier scores a single sampled structure with **no native reference**, so it can run at
deployment time to pick the Best-of-N structure. The goal is to proxy DockQ (see
`../metrics`). Scores fall into families: PAE-based interface scores, native confidence fields,
a pool-based consensus score, physics-inspired biophysical descriptors, and a CDR/paratope
signal. All ab-ag aggregation takes the max of a chain pair's two directions and averages over
the antibody-antigen interfaces only, excluding the intra-antibody VH-VL interface
(`ipsae._abag_scalar`).

Throughout, $\mathrm{PAE}_{ij}$ is the predicted aligned error between residues $i$ and $j$. A separate,
non-verifier module (`diversity.py`) provides RMSD-based pool diversity and inference-time steering geometry,
documented at the end.

## Verifiers

### ipSAE, `ipsae.py`

Interface predicted Structure Alignment Error. A PAE-based interface verifier (the strongest
interface-score proxy for DockQ in our retro-validation). The interface is
defined by the PAE cutoff $c$ alone (no 3D distance gate, which is what distinguishes it from ipTM).
For an ordered chain pair $A \to B$ and each residue $i \in A$:

$$
V_i = \{\, j \in B : \mathrm{PAE}_{ij} < c \,\}, \qquad n_i = |V_i|
$$

$$
d_0(n) = \max\!\Big( 1.0,\ 1.24\,(\max(n, 26) - 15)^{1/3} - 1.8 \Big)
$$

$$
\mathrm{ptm}_i = \frac{1}{n_i} \sum_{j \in V_i} \frac{1}{1 + \big(\mathrm{PAE}_{ij} / d_0(n_i)\big)^2}, \qquad \mathrm{ipSAE}_{A \to B} = \max_{i \in A} \mathrm{ptm}_i
$$

The symmetric report is $\max(\mathrm{ipSAE}_{A \to B},\, \mathrm{ipSAE}_{B \to A})$.

- **Implementation**: `compute_ipsae()` (canonical numpy, non-differentiable), plus `soft_ipsae()`
  (differentiable PyTorch surrogate for guidance: sigmoid mask, softplus $d_0$, LogSumExp over
  residues). $d_0$ in `_d0()`.
- **Reference**: Dunbrack Lab, biorxiv 2025.02.10.637595. Code: https://github.com/DunbrackLab/ipSAE

### pDockQ2, `interface_scores.py`

Predicted DockQ v2, a calibrated logistic on the PAE-weighted interface pLDDT. Over the contact set
$C$ (Cα pairs $< 8$ Å across the interface):

$$
\mathrm{ipPAE} = \frac{1}{|C|} \sum_{(i,j) \in C} \frac{1}{1 + (\mathrm{PAE}_{ij} / 10)^2}, \qquad x = \mathrm{ipLDDT} \cdot \mathrm{ipPAE}
$$

$$
\mathrm{pDockQ2} = \frac{L}{1 + e^{-k (x - x_0)}} + b
$$

with $L = 1.31034849$, $x_0 = 84.7326239$, $k = 0.0747157696$, $b = 0.00501886443$, and $\mathrm{ipLDDT}$
the contact-weighted mean pLDDT. About $0.23$ marks a likely acceptable interface.

- **Implementation**: `compute_interface_scores()` / `_directional_scores()`, constants `PD2_*`.
  Cross-checked against `af_analysis.compute_pdockQ2`.
- **Reference**: Zhu et al., *Evaluation of AlphaFold-Multimer on multi-chain complexes*,
  Bioinformatics 2023. Package: https://github.com/samuelmurail/af_analysis

### LIS / iLIS, `interface_scores.py`

Local Interaction Score. Rewards low-PAE interface blocks. Over the inter-chain block:

$$
\mathrm{LIS} = \operatorname*{mean}_{\mathrm{PAE}_{ij} < 12} \left( 1 - \frac{\mathrm{PAE}_{ij}}{12} \right), \qquad \mathrm{iLIS} = \sqrt{\mathrm{LIS} \cdot \mathrm{cLIS}}
$$

where $\mathrm{cLIS}$ is the same average restricted to pairs that also form a Cα contact ($< 8$ Å),
so $\mathrm{iLIS} = 0$ with no confident contact. $\mathrm{iLIS} \ge 0.203$ flags a confident interaction.

- **Implementation**: same `compute_interface_scores()`, constants `LIS_*`.
- **Reference**: Kim et al., bioRxiv 2024. Code: https://github.com/flyark/AFM-LIS

### Coordinate signals, `interface_scores.py`

Cheap geometry read off the predicted structure: `i_plddt` (mean pLDDT of interface residues,
both sides), `n_interface_residues`, `n_interface_contacts`, `clash_ca`. `compute_coord_signals()`.

### Pair-ipTM summary, `interface_scores.py`

`min_pair_iptm` / `avg_pair_iptm`, read free from the backend's chain-pair ipTM JSON, ab-ag masked.
`compute_summary_pair_signals()`.

### Consensus (fraction + density), `consensus.py`

The pool-based selector. Ranks a sample by how many of the other $N$ samples share its antibody pose,
the standard docking crowd/cluster rule (ClusPro, HADDOCK). Unsupervised, uses no model internals.
Split into antibody (H/L) vs antigen, Kabsch-superpose every sample onto a reference by its **antigen**
Cα (common frame), carry the antibody along, and let $\mathrm{RMSD}_{kl}$ be the antibody-Cα RMSD
between samples $k$ and $l$ in that frame. Then, with threshold $\tau = 8$ Å and width $\sigma = 4$ Å:

$$
\mathrm{consensus}_k = \frac{1}{N} \sum_{l=1}^{N} \mathbb{1}\!\left[\mathrm{RMSD}_{kl} < \tau\right] \quad\text{(hard fraction)}, \qquad \mathrm{consensus\_density}_k = \frac{1}{N} \sum_{l=1}^{N} e^{-\mathrm{RMSD}_{kl}^2 / \sigma^2} \quad\text{(soft)}
$$

- **Implementation**: `consensus_scores()` returns `(fraction, density)`.`_superpose()` Kabsch.
- **Reference**: consensus / cluster-density ranking, the idea that a correct pose recurs across a
  sampled ensemble while wrong poses scatter. The exact form here (rank a structure by the count of
  pool members within an RMSD cutoff, or by a Gaussian-weighted neighbour count) is the **SPICKER
  cluster-density** rule for picking near-native folds from a decoy pool, where cluster density
  correlates with RMSD-to-native (Zhang & Skolnick, *SPICKER: A clustering approach to identify
  near-native protein folds*, J. Comput. Chem. 25(6):865-871, 2004, DOI 10.1002/jcc.20011), the same
  population/cluster-size idea appears in protein docking (Kozakov et al., Biophys. J. 2005, DOI
  10.1529/biophysj.104.058768. ClusPro, Nat. Protoc. 2017). Our density is the Gaussian-kernel
  (kernel density estimate) form of that count. The same consensus principle underpins modern
  large-ensemble AlphaFold complex prediction, which generates thousands of models and ranks by
  ensemble consensus rather than single-model confidence (Liu et al., MULTICOM4 in CASP16, 2025.
  Genz et al., Protein Science 2025, DOI 10.1002/pro.70327).

### Biophysical interface, `biophysical.py`

Physics-inspired descriptors that need the heavy atoms (read via biotite, not the Cα loader). The
headline is the buried surface area, the standard two-sided interface area:

$$
\mathrm{BSA} = \mathrm{SASA}(\mathrm{ab}) + \mathrm{SASA}(\mathrm{ag}) - \mathrm{SASA}(\mathrm{complex})
$$

Alongside it: `bsa_hydrophobic_frac` (carbon/sulfur share of the BSA), `n_salt_bridges`
(Asp/Glu-O $\leftrightarrow$ Lys/Arg/His-N cross-interface pair $< 4$ Å), and `n_hbond_contacts`
(N/O donor-acceptor heavy-atom pairs $< 3.5$ Å).

- **Implementation**: `compute_biophys()` / `compute_biophys_array()`. SASA via biotite
  Shrake-Rupley in `_sasa()`.
- **Reference**: Shrake & Rupley 1973 (SASA algorithm), mirrors the reference implementation
  `interface_metrics.get_interface_bsa`. Package: https://github.com/biotite-dev/biotite

### CDR / paratope focus, `cdr.py`

Is the antibody binding through its CDR loops or its framework? With the paratope $P$ = antibody
residues contacting the antigen:

$$
\mathrm{paratope\_cdr\_fraction} = \frac{\lvert P \cap \mathrm{CDR} \rvert}{\lvert P \rvert}
$$

High is good (CDR-driven interface). CDR residues come from IMGT numbering (ANARCI), computed once
per target and reused for every sample, the per-sample step is pure heavy-atom contact geometry.

- **Implementation**: `cdr_residues_from_cif()` (ANARCI) + `compute_cdr_paratope()` (geometry).
- **Reference**: ANARCI, Dunbar & Deane, Bioinformatics 2016
  (https://github.com/oxpig/ANARCI). IMGT numbering (Lefranc et al.). Mirrors the reference implementation
  `structure_analyzer._compute_paratope_focus_metrics`.

### Native confidence fields, `confidence.py`

Ranks purely by a field the backend already emits. The default selection verifier is `ranking_score`,
the model's own composite:

$$
\mathrm{ranking\_score} = 0.8 \cdot \mathrm{ipTM} + 0.2 \cdot \mathrm{pTM}
$$

Any of `ranking_score`, `iptm`, `ptm`, `plddt`, `gpde` can be the selector. No computation, just
enumerate and sort (`ConfidenceVerifier`).

## Diversity and steering geometry, `diversity.py`

Not a verifier: reusable pairwise-prediction RMSD / geometry for measuring how a Best-of-N pool is spread and
for **inference-time steering** (a reward that pushes a sample away from a reference, in the style of
ConforMix, arXiv:2512.03312). Ground-truth-free (measured between predictions). Coordinates are Cα only, and
the caller frames the pool by Kabsch-superposing every antigen onto a reference and carrying the antibody, so
antibody coordinates arrive here in a common antigen frame. Each of the two Figure 10 axes has a POOLED form
(a per-target diversity, the median over the $\binom{n}{2}$ structure pairs) and a SINGLE-PAIR form (a steering
collective variable of one candidate versus a reference). Both use the interface contact masks at cutoff
$r_c = 10$ Å across the interface,

$$
E = \{\, j \in \mathrm{ag} : \textstyle\min_i \lVert \mathrm{ag}_j - \mathrm{ab}_i \rVert < r_c \,\}, \qquad
P = \{\, i \in \mathrm{ab} : \textstyle\min_j \lVert \mathrm{ab}_i - \mathrm{ag}_j \rVert < r_c \,\}
$$

the epitope $E$ and paratope $P$ (`epitope_mask` / `paratope_mask`), $P$ can instead be the IMGT CDR residues
from `cdr.py`.

### Binding position, `diversity.py`

WHERE the antibody binds, orientation-free (Figure 10a). The binding position is the epitope centroid, the
mean of the contacted antigen Cα. The pooled diversity over $n$ antibodies in the shared antigen frame is the
median pairwise centroid distance, and the single-pair steering shift is the centroid move versus a reference:

$$
\bar{c} = \frac{1}{|E|} \sum_{j \in E} \mathrm{ag}_j, \qquad
\mathrm{binding\_diversity} = \operatorname*{median}_{a < b} \lVert \bar{c}_a - \bar{c}_b \rVert, \qquad
\mathrm{binding\_shift} = \lVert \bar{c} - \bar{c}_\mathrm{ref} \rVert
$$

- **Implementation**: `epitope_centroid()` (falls back to the single nearest antigen residue if nothing is
  within $r_c$), `binding_diversity()` (pooled, Figure 10a), `binding_shift()` (single-pair steering CV). Uses
  centroids only, so it is invariant to antibody orientation and internal conformation.
- **Reference**: epitope footprint radius $\approx 16$ Å from a 600 to 900 Å² / 15 to 25-residue epitope
  (Ramaraj et al. 2012 *Biochim Biophys Acta* 1824:520. MacCallum, Martin & Thornton 1996 *J Mol Biol*
  262:732), the diagnostic split for panel (a).

### CDR conformation, `diversity.py`

The internal CDR-loop conformation with global pose removed (Figure 10b), the diversity the pool most lacks
and the recommended steering axis. It is the optimal-superposition (Kabsch-Umeyama) Cα RMSD on the paratope.
For two point sets $A, B \in \mathbb{R}^{M \times 3}$, centre both, take the SVD of the $3 \times 3$
cross-covariance $A^\top B = U \Sigma V^\top$ with singular values $\sigma_1 \ge \sigma_2 \ge \sigma_3$, and
correct for reflection with $s = \operatorname{sign} \det(U V^\top)$:

$$
\mathrm{RMSD}(A, B) = \sqrt{ \frac{1}{M} \Big( \lVert A \rVert_F^2 + \lVert B \rVert_F^2 - 2\,(\sigma_1 + \sigma_2 + s\,\sigma_3) \Big) }
$$

Writing $X^{(a)}_P$ for the paratope Cα of structure $a$, the pooled diversity is the median pairwise RMSD over
the paratope and the single-pair steering reward is the RMSD versus a reference:

$$
\mathrm{cdr\_conformation\_diversity} = \operatorname*{median}_{a < b} \mathrm{RMSD}\big( X^{(a)}_P, X^{(b)}_P \big), \qquad
\mathrm{cdr\_conformation\_rmsd} = \mathrm{RMSD}\big( X_P, X^\mathrm{ref}_P \big)
$$

- **Implementation**: `kabsch_rmsd()` (single pair, one SVD), `median_pairwise_superposed_rmsd()` (pooled, a
  fresh per-pair Kabsch via batched $3 \times 3$ SVD, so frame-invariant and measuring CONFORMATION),
  `cdr_conformation_diversity()` (pooled, Figure 10b), `cdr_conformation_rmsd()` (single-pair steering CV). The
  sibling `median_pairwise_rmsd_in_frame()` skips superposition and uses the Gram identity
  $\lVert X_i - X_j \rVert^2 = \lVert X_i \rVert^2 + \lVert X_j \rVert^2 - 2 \langle X_i, X_j \rangle$ to keep
  position, so it measures POSE rather than conformation. For gradient-based guidance, port `kabsch_rmsd` to
  PyTorch (the superposition is differentiable through the SVD).
- **Reference**: Kabsch 1976 *Acta Cryst* A32:922. Umeyama 1991 *IEEE PAMI* 13:376, closed form from the
  cross-covariance singular values, Coutsias et al. 2004 *J Comput Chem* 25:1849. The $2$ Å
  distinct-conformation split for panel (b) follows the CoDNaS rigid-class mean of $0.83$ Å (Monzon et al.).
  Steering away from a default by an RMSD reward: ConforMix, arXiv:2512.03312.

**Callers**: `experiments/scripts/failmode.py` (the two Figure 10 axes) and
`experiments/scripts/diversity_axes.py` (the other five diversity axes) import these primitives.

## Infrastructure (not scores)

- `score.py`, `make_verifier`, `augment_signals` (post-hoc groups: coord / pair_iptm / consensus /
  biophys / cdr), parallel workers.
- `interface_verifier.py`, `ipsae_verifier.py`, `base.py`, verifier classes / the `Verifier` protocol.
- `cif_io.py`, shared mmCIF Cα reader (`load_ca_coords_chains`) used by ipSAE / interface scores /
  consensus.
