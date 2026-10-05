# `method/metrics`, evaluation metrics

Ground-truth structure-quality metrics. Unlike the verifiers (which score a sample
with no native reference), a metric compares a predicted structure against the known
native, so it is used for **evaluation only**, never for deployment-time selection.
DockQ is the headline target the verifiers are trained to proxy. All metrics are run
through the parallel driver in `score.py`.

## Metrics

### DockQ, `dockq.py`
The docking-quality target metric. A single continuous score in $[0, 1]$ combining three
per-interface terms.

Per interface, with the scaling function $s(x, d) = \dfrac{1}{1 + (x/d)^2}$:

$$\mathrm{DockQ} = \frac{1}{3}\Big( F_\mathrm{nat} + s(\mathrm{iRMS},\, 1.5) + s(\mathrm{LRMS},\, 8.5) \Big)$$

where $F_\mathrm{nat}$ is the fraction of native residue-residue contacts recovered,
$\mathrm{iRMS}$ the interface RMSD, and $\mathrm{LRMS}$ the ligand RMSD.

- **Project aggregation** $\mathrm{dockq\_abag} = \dfrac{1}{|I_\mathrm{abag}|}\sum_{p \in I_\mathrm{abag}} \mathrm{DockQ}_p$,
  the mean over the **antibody-antigen interfaces only** (the intra-antibody VH-VL interface is
  excluded). Implemented in `antigen_mean_dockq()`, the three terms are decomposed in
  `antigen_mean_dockq_terms()`.
- **Implementation**: `compute_dockq()` shells out to the reference DockQ **v2** binary and parses
  its stdout (`_parse_stdout`). No re-implementation of the math.
- **Reference**: Basu & Wallner, *DockQ: A Quality Measure for Protein-Protein Docking Models*,
  PLoS ONE 2016. Code: https://github.com/bjornwallner/DockQ

### lDDT, `lddt.py`
Local Distance Difference Test. Superposition-free per-residue distance-agreement score in $[0, 1]$.
Over the set of preserved inter-atomic distances $D$ (native distances below an inclusion radius),
lDDT is the fraction reproduced within a set of tolerance thresholds $\mathcal{T} = \{0.5, 1, 2, 4\}$ Å:

$$\mathrm{lDDT} = \frac{1}{|\mathcal{T}|} \sum_{\Delta \in \mathcal{T}} \frac{1}{|D|} \sum_{(i,j) \in D} \mathbb{1}\!\left[\, \big| d^{\mathrm{model}}_{ij} - d^{\mathrm{native}}_{ij} \big| < \Delta \,\right]$$

- **Implementation**: `compute_lddt()` calls `ost compare-structures` (OpenStructure) in JSON mode
  and parses the result. FoldBench pins OpenStructure 2.8.
- **Reference**: Mariani, Biasini, Barbato & Schwede, *lDDT: a local superposition-free score for
  comparing protein structures and models*, Bioinformatics 2013. Tool: https://openstructure.org

### TM-score, `tmscore.py`
Global fold-similarity score in $(0, 1]$, length-normalised so it is comparable across sizes
($>0.5$ = same fold). Over $L_\mathrm{ali}$ aligned residue pairs at distance $d_i$, normalised by
target length $L$:

$$\mathrm{TM} = \frac{1}{L} \sum_{i=1}^{L_\mathrm{ali}} \frac{1}{1 + (d_i / d_0)^2}, \qquad d_0 = 1.24\,\sqrt[3]{L - 15} - 1.8$$

- **Implementation**: `compute_tmscore()` runs `TMalign` (or `USalign`) as a subprocess and reads
  the reported TM-score.
- **Reference**: Zhang & Skolnick, *Scoring function for automated assessment of protein structure
  template quality*, Proteins 2004. TM-align: Zhang & Skolnick, NAR 2005. US-align:
  https://github.com/pylelab/USalign

### Cα-RMSD, `rmsd.py`
Root-mean-square deviation of Cα atoms after optimal rigid-body superposition. Pure numpy, no
external binary. After the Kabsch rotation aligns model coordinates $x_i$ onto native $y_i$:

$$\mathrm{RMSD} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} \lVert x_i - y_i \rVert^2}$$

- **Implementation**: `_kabsch()` (SVD rotation) + `ca_rmsd()`. `pairwise_ca_rmsd()` builds the
  all-vs-all matrix used by the diversity metric.
- **Reference**: Kabsch, *A solution for the best rotation to relate two sets of vectors*,
  Acta Cryst. A 1976 (1978 correction).

### Diversity, `diversity.py`
Ensemble-spread summaries over a target's samples, for the success-vs-diversity Pareto view. From
the pairwise Cα-RMSD matrix $R$ over the $N$ samples it reports the mean spread
$\bar{R} = \dfrac{2}{N(N-1)} \sum_{i<j} R_{ij}$ and the number of clusters at a `cluster_threshold_A`
(default 2.0 Å) cutoff.

- **Implementation**: `diversity_summary()` on top of `rmsd.pairwise_ca_rmsd()`.

## Driver, `score.py`
`score_metrics()` / `score_one()` run the metrics in parallel: a process pool over targets and a
thread pool over samples (the external binaries release the GIL). `MetricSpec` selects which
metrics to compute (default `dockq, lddt, tm, ca_rmsd, diversity`) and caches per-sample results.
