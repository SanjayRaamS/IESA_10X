# References

Every augmentation, noise model and algorithm choice in Drift-Sense, mapped to
the literature that justifies it. Grouped **by design decision**, so each entry
answers "why is the code like this?" rather than just listing papers.

Citation keys (`[Reimer1998]` etc.) are used in code comments — mainly
`core/sem.py`, `core/preprocess.py`, `core/lattice.py`, `core/refine.py`.

Every citation below was checked against the publisher/indexer record for
title, authors, venue, volume, pages and year before shipping. Where a number
we quote comes from a figure or table we could not read as text, it is marked
**[reported]** and attributed to the secondary source we actually read. Where
our model deliberately departs from the cited reality, the entry says so.

---

## 1. Layout geometry — DRAM word-line / bit-line arrays
**Code:** `core/layout.py: render_dram()` — two orthogonal line families,
`pitch_x` (bit-lines, vertical) and `pitch_y` (word-lines, horizontal), each
drawn independently from `U(30, 60)` px, plus sub-array breaks whose gaps are
**integer multiples of the pitch**.

- **[Schloesser2008]** T. Schloesser, F. Jakubowski, J. v. Kluge, et al.,
  "6F² buried wordline DRAM cell for 40 nm and beyond," *IEEE International
  Electron Devices Meeting (IEDM) Technical Digest*, 2008, pp. 809–812.
  DOI: 10.1109/IEDM.2008.4796820.
- **[Ha2023]** D. Ha et al., "Highly Manufacturable, Cost-Effective, and
  Monolithically Stackable 4F² Single-Gated IGZO Vertical Channel Transistor
  (VCT) for sub-10 nm DRAM," *IEEE IEDM*, 2023, paper 6.3.

**What they justify.** A 6F² cell is built from word-lines and bit-lines at
1F line / 1F space, so **both families sit at pitch 2F — the two pitches are
equal by construction**, and the array is a genuinely square-ish 2-D lattice
rather than a 1-D grating. That is why `render_dram` draws `pitch_x` and
`pitch_y` from the *same* distribution instead of forcing one much coarser
than the other. [Ha2023] is the recent process point: the move to 4F² tightens
the cell but keeps the same orthogonal two-family topology, so the geometric
model does not go stale.

**What they justify about the aperiodic content.** Sub-array breaks in a real
DRAM are *skipped line positions* on the same litho grid, not arbitrary gaps.
`layout.py` enforces this (gap = integer × pitch) because a fractional gap
every N lines would move the FFT fundamental to `pitch + gap/N` and quietly
corrupt the scale estimate in `core/lattice.py`. This is the single most
consequential geometry decision in the dataset.

---

## 2. Layout geometry — FinFET fin / gate arrays
**Code:** `core/layout.py: render_finfet()` — dense vertical fins at
`pitch_x ~ U(20, 36)` px crossed by horizontal gate bars at
`pitch_y = pitch_x × U(3, 5)`.

- **[Auth2012]** C. Auth, C. Allen, A. Blattner, et al., "A 22 nm high
  performance and low-power CMOS technology featuring fully-depleted tri-gate
  transistors, self-aligned contacts and high density MIM capacitors,"
  *Symposium on VLSI Technology (VLSIT)*, 2012, pp. 131–132.
  DOI: 10.1109/VLSIT.2012.6242496.
- **[Auth2017]** C. Auth, A. Aliyarukunju, M. Asoro, et al., "A 10 nm high
  performance and low-power CMOS technology featuring 3rd generation FinFET
  transistors, Self-Aligned Quad Patterning, contact over active gate and
  cobalt local interconnects," *IEEE IEDM*, 2017, pp. 29.1.1–29.1.4.
  DOI: 10.1109/IEDM.2017.8268472.

**What they justify.** A FinFET array really is two crossed line families of
*different* pitch and *different* linewidth, with the fin family much finer
than the gate family, and the crossings optically distinct from either line
alone — which is exactly the three-grey-level structure `render_finfet`
produces (`_FIN`, `_GATE`, `_FIN + _GATE`). Confirmed numbers:

| Node | Fin pitch | Contacted gate pitch | gate:fin ratio |
|---|---|---|---|
| Intel 22 nm [Auth2012] | 60 nm **[reported]** | **90 nm** (quoted verbatim from the paper: "Contacted gate pitch is scaled to 90 nm") | ≈ 1.5 |
| Intel 10 nm [Auth2017] | 34 nm **[reported]** | 54 nm **[reported]** | ≈ 1.6 |

**[reported]** values above are from the IEEE/press summaries of those papers;
Table I of [Auth2012] is a raster figure and its fin-pitch cell could not be
extracted as text, so we do not claim to have read 60 nm in the primary
source. The 90 nm gate pitch *was* read directly from the paper body.

**Where we deliberately depart from reality — state this on slide 9.** Real
logic has a gate:fin pitch ratio of roughly **1.5–1.6**. Our generator uses
**3–5×**. This is not an attempt at fidelity and should not be defended as
one. The reason is the imaging chain, not the layout: at a ~10× magnification
ratio a 20–36 px reference fin pitch lands at **2–3.6 px per period in the
search image**, right at the Nyquist limit, and the fin family is effectively
erased by the beam PSF (see §5). If the gate family were only 1.5× coarser it
would be erased too, and *both* lattice families would be gone — the pair
would be unsolvable for reasons that have nothing to do with our algorithm.
Widening the ratio to 3–5× keeps one coarse family (6–18 px/period in the
search image) alive through demagnification, so the FFT stage in
`core/lattice.py` has a fundamental to lock onto. It is a **solvability**
choice, and it makes our FinFET pairs *easier* than a true 1.6× layout would
be. A reviewer should read our FinFET numbers with that caveat attached.

---

## 3. Edge brightening and topographic contrast (the SE yield model)
**Code:** `core/sem.py` — secondary-electron yield rises where the surface
tilts, so feature edges are rendered brighter than either adjacent flat
region.

- **[Reimer1998]** L. Reimer, *Scanning Electron Microscopy: Physics of Image
  Formation and Microanalysis*, 2nd ed., Springer Series in Optical Sciences
  vol. 45, Springer, 1998. (SE yield vs. surface tilt; the ~1/cos θ
  dependence that produces edge brightening.)
- **[Goldstein2018]** J. I. Goldstein, D. E. Newbury, J. R. Michael,
  N. W. M. Ritchie, J. H. J. Scott, D. C. Joy, *Scanning Electron Microscopy
  and X-Ray Microanalysis*, 4th ed., Springer, 2018. (SE1/SE2 emission
  physics and topographic contrast formation.)

**What they justify.** Edge brightening is the reason a top-down SEM of a line
array is not a square wave — it is a square wave with bright rails. That
matters to us twice over: it puts most of the image energy at the *edges*
(good for the aperiodic residual, which is made of edge discontinuities), and
it makes the signal **polarity-dependent**, which is why `core/preprocess.py`
uses doubled-angle gradient features that are invariant to contrast polarity
rather than raw intensity.

---

## 4. Charging as a slowly-varying background
**Code:** `core/sem.py` — a wide random Gaussian blob field added as a
low-frequency multiplicative/additive background.

- **[Reimer1998]** (as above) — insulator charging and its effect on local
  SE collection efficiency.
- **[Postek2015]** M. T. Postek and A. E. Vladár, "Does Your SEM Really Tell
  the Truth?—How Would You Know? Part 4: Charging and its Mitigation,"
  *Proc. SPIE* 9636, 963605, 2015. DOI: 10.1117/12.2195344.

**What they justify.** Charging on dielectric layers drifts the local contrast
over length scales far larger than the feature pitch. Modelling it as a
*smooth, low-spatial-frequency* field (rather than as noise) is what makes the
flatten + local contrast normalisation step in `core/preprocess.py` both
necessary and sufficient: a background that lives well below the lattice
fundamental can be removed without touching the signal we localise on.

---

## 5. Beam PSF, sharpness, and raster distortion
**Code:** `core/sem.py` — Gaussian PSF with anisotropy
`sigma_y = sigma_x × U(0.9, 1.1)` (astigmatism / raster distortion);
`core/lattice.py: match_psf()` — closed-form estimate of the PSF sigma
difference between the two images.

- **[PostekVladar1996]** M. T. Postek and A. E. Vladár, "SEM performance
  evaluation using the sharpness criterion," *Proc. SPIE* 2725, *Metrology,
  Inspection, and Process Control for Microlithography X*, 1996, pp. 504–514.
  DOI: 10.1117/12.240107.
- **[Postek1998]** M. T. Postek and A. E. Vladár, "Image sharpness measurement
  in scanning electron microscopy — Part I," *Scanning* 20, 1998, pp. 1–9.
  (Part II: A. E. Vladár, M. T. Postek, M. P. Davidson, *Scanning* 20, 1998,
  pp. 24–34.)
- **[Postek2013]** M. T. Postek and A. E. Vladár, "Does your SEM really tell
  the truth? How would you know?," *Scanning* 35, 2013, pp. 355–361.

**What they justify.** Two things. First, that a Gaussian PSF is the right
first-order model — [PostekVladar1996] characterises SEM resolution *in the
frequency domain* ("the low frequency changes in the video signal contain
information about the larger features and the high frequency ones carry
information of finer details"), which is precisely the log-spectrum picture
`match_psf` works in. Second, that astigmatism is a routine, real, per-image
condition rather than an exotic fault — hence a *different, random* PSF
anisotropy per image, which the pipeline must estimate rather than assume.

**Why this matters to the algorithm, not just the dataset.** Because Gaussian
blur is *additive in the log power spectrum*, the PSF difference between the
reference and the search image reduces to a one-parameter linear least-squares
fit in σ². `match_psf` solves it in closed form; the earlier 61-step σ grid
search gave identical results at ~10× the cost (0.225 s → ~0.02 s).

---

## 6. Poisson–Gaussian (shot + readout) noise
**Code:** `core/sem.py` — signal-dependent Poisson term scaled by an electron
dose, plus additive Gaussian readout.

- **[Foi2008]** A. Foi, M. Trimeche, V. Katkovnik, K. Egiazarian, "Practical
  Poissonian-Gaussian noise modeling and fitting for single-image raw-data,"
  *IEEE Trans. Image Processing* 17(10), 2008, pp. 1737–1754.
  DOI: 10.1109/TIP.2008.2001399.

**What it justifies.** The exact two-parameter form we implement: variance
that is *affine in the signal*, `var = a·y + b`. In SEM the `a·y` term is beam
shot noise (dose-limited) and `b` is detector/readout noise. This is the model
the whole preprocessing stack is built against, and it is why "dose" is a
first-class manifest field the evaluation harness can bucket on. A single
citation is enough here because [Foi2008] *is* the canonical statement of the
model; §7 covers what to do about it.

---

## 7. Anscombe variance-stabilising transform
**Code:** `core/preprocess.py` — `2·sqrt(x + 3/8)` applied before any
filtering or correlation.

- **[Anscombe1948]** F. J. Anscombe, "The transformation of Poisson, binomial
  and negative-binomial data," *Biometrika* 35(3–4), 1948, pp. 246–254.
  DOI: 10.1093/biomet/35.3-4.246.
- **[Makitalo2011]** M. Mäkitalo and A. Foi, "Optimal inversion of the
  Anscombe transformation in low-count Poisson image denoising," *IEEE Trans.
  Image Processing* 20(1), January 2011, pp. 99–109.
  DOI: 10.1109/TIP.2010.2056693.

**What they justify.** [Anscombe1948] is the transform itself, including the
3/8 offset — it maps Poisson data to approximately unit-variance Gaussian.
That is the precondition every downstream step silently assumes: local
contrast normalisation, gradient features and normalised cross-correlation are
all built for *homoscedastic* noise, and applying them to raw Poisson counts
weights bright regions wrongly. [Makitalo2011] is cited for the honest
boundary: the forward transform degrades at very low counts, which is the
regime our low-dose pairs sit closest to. We do **not** need its optimal
*inverse* — we never invert, because we correlate in the transformed domain
and never display or measure intensities — but it is the reference that tells
you when the assumption starts to fail.

---

## 8. FFT-based scale and rotation estimation
**Code:** `core/lattice.py` — log-polar / reciprocal-lattice treatment of the
magnitude spectrum to recover scale and rotation *without* translation.

- **[Reddy1996]** B. S. Reddy and B. N. Chatterji, "An FFT-based technique for
  translation, rotation, and scale-invariant image registration," *IEEE Trans.
  Image Processing* 5(8), 1996, pp. 1266–1271. DOI: 10.1109/83.506761.

**What it justifies.** The core structural claim of the whole project: the
*magnitude* of the Fourier transform is translation-invariant, so rotation
becomes a shift in angle and scale a shift in log-radius. This is what lets
Drift-Sense **solve geometry before it ever thinks about position**, which is
the thesis in CLAUDE.md ("periodic component carries zero position information
but perfect scale/rotation information") expressed as an algorithm.

**Where we go beyond [Reddy1996], and why we had to.** Textbook log-polar
phase correlation fails on this data. Two measured failure modes: (a) on a
sub-array-broken lattice, the *superlattice* peaks are shorter in reciprocal
space than the fundamental, so independently choosing "the two shortest
reciprocal vectors" in each image picks non-corresponding bases; (b) FinFET
fins at ~2.5 px/period in the search image are erased entirely, so the two
images do not even share a peak set. Our fix — match reciprocal *vectors*
between images as complex numbers under the similarity constraint
`f_search = c · f_ref`, score each hypothesis by how many other peaks it
explains, refine by weighted complex least squares — is a departure from the
reference and is measured, not assumed: median rotation error 0.040°, median
scale error 0.054%, p95 rotation 0.160°.

---

## 9. Normalised cross-correlation over the score maps
**Code:** `core/correlate.py` — dual-channel (lattice-notched residual +
gradient) normalised score maps.

- **[Lewis1995]** J. P. Lewis, "Fast normalized cross-correlation,"
  *Vision Interface*, 1995, pp. 120–123.

**What it justifies.** Both the normalisation (correlation must be invariant
to local gain and offset, or the charging background of §4 dominates it) and
the running-sums/FFT implementation that makes a full-search-image score map
affordable inside the time budget. OpenCV's `TM_CCOEFF_NORMED`, which
`localize.py` falls back to when the main pipeline fails, is this method.

**The honest caveat.** [Lewis1995] applied naively to this problem is exactly
the baseline our thesis says must fail: NCC mixes periodic and aperiodic
energy, and the periodic energy is overwhelming, so the score surface is a
near-uniform lattice of near-equal peaks. We use it *after* notching the
lattice out, on the residual — the citation justifies the estimator, not the
naive application of it.

---

## 10. Sub-pixel refinement by ECC alignment
**Code:** `core/refine.py` — ECC alignment between the warped reference and
the search patch; `rescore_candidates()` re-ranks peaks by
`ECC_cc × (1 + ZNCC on the notched residual)`.

- **[Evangelidis2008]** G. D. Evangelidis and E. Z. Psarakis, "Parametric
  image alignment using enhanced correlation coefficient maximization,"
  *IEEE Trans. Pattern Analysis and Machine Intelligence* 30(10), October
  2008, pp. 1858–1865. DOI: 10.1109/TPAMI.2008.113.

**What it justifies.** ECC maximises a correlation coefficient that is
invariant to photometric gain and bias, which is the right objective when the
two images were taken at different magnification, different dose and different
charging state — an SSD-based refiner would chase the brightness difference
instead of the alignment. It is available as `cv2.findTransformECC`, so this
costs us no dependency beyond the allowed set.

**Measured, and beyond the brief.** On pairs where cell selection is correct,
ECC refinement reaches **median 0.064 px, p90 0.407 px** — about 8× better
than the 0.5 px target. Using the ECC coefficient as a *re-ranking* signal
(not just a refiner) moved median error over 40 pairs from 153 px to 0.6 px.
Rejected alternatives, measured: ECC alone (5.0 px), residual ZNCC alone
(15.0 px), additive combinations (2.0–3.2 px).

---

## 11. Confidence estimation and the reject option
**Code:** `core/confidence.py` (numpy-only inference), `train_confidence.py`
(fitting, scikit-learn, development dependency only). A six-feature logistic
regression predicts *correct vs incorrect* (error ≤ 5 px) and is used **only**
to accept or reject an answer — never to produce one.

- **[Chow1970]** C. K. Chow, "On optimum recognition error and reject
  tradeoff," *IEEE Trans. Information Theory* 16(1), 1970, pp. 41–46.
  DOI: 10.1109/TIT.1970.1054406.
- **[Cox1958]** D. R. Cox, "The regression analysis of binary sequences,"
  *Journal of the Royal Statistical Society, Series B* 20(2), 1958,
  pp. 215–232. DOI: 10.1111/j.2517-6161.1958.tb00292.x.
- **[Pedregosa2011]** F. Pedregosa, G. Varoquaux, A. Gramfort, et al.,
  "Scikit-learn: Machine Learning in Python," *Journal of Machine Learning
  Research* 12, 2011, pp. 2825–2830.

**What they justify.** [Chow1970] is the reason a reject option is the right
shape of answer for this problem at all: it establishes the optimum
error/reject tradeoff and shows that a recogniser which may abstain, thresholded
on its own posterior probability, dominates one forced to decide. That is
exactly the fab-tool framing — a navigation-recovery tool that says "I don't
know" on the ambiguous half is more useful than one that guesses silently.
[Cox1958] is the model itself. [Pedregosa2011] is the implementation used to
fit it.

**Why a linear model on six features, and not something larger.** With 40 pairs
and ~22 positives, anything with real capacity would memorise the draw. The
features are already physics-normalised by the earlier stages — ratios, counts
and correlation coefficients, never raw scores or pixel sizes — so the residual
job is a weighting, which is what logistic regression is. The fitted weights
are small and every sign is physically sensible (ambiguity down-weights,
aperiodic agreement up-weights), which is itself evidence the model is reading
the physics rather than the sample.

**The honest limitation, measured.** On the held-out seed-4242 set the model
reaches AUC 0.987 and 92.5% accuracy, and its `p ≥ 0.5` gate accepts 24/40
predictions at 95.8% precision. But as a *ranking* it is barely distinguishable
from the single `rescore_margin` feature that already existed
(Spearman ρ = +0.92 between the two on held-out data; it wins one reject tier
and ties two). The defensible claim is **calibration, not discrimination**: the
logistic regression converts an uncalibrated margin into a probability with a
threshold that needs no per-dataset tuning. `train_confidence.py` prints this
comparison on every run and states the verdict either way, so the claim cannot
quietly drift.

---

## Sources consulted for verification

- [IEEE TIP 5(8):1266–1271](https://ieeexplore.ieee.org/document/506761/) · [ADS record](https://ui.adsabs.harvard.edu/abs/1996ITIP....5.1266R/abstract) — Reddy & Chatterji 1996
- [IEEE TPAMI 30(10):1858–1865](https://ieeexplore.ieee.org/document/4515873/) · [PubMed 18703836](https://pubmed.ncbi.nlm.nih.gov/18703836/) — Evangelidis & Psarakis 2008
- [Lewis, Fast Normalized Cross-Correlation (author's copy)](https://scribblethink.org/Work/nvisionInterface/nip.pdf) — Lewis 1995
- [IEEE TIP 17(10):1737–1754](https://dl.acm.org/doi/10.1109/TIP.2008.2001399) · [ADS record](https://ui.adsabs.harvard.edu/abs/2008ITIP...17.1737F/abstract) — Foi et al. 2008
- [Biometrika 35(3–4):246–254](https://academic.oup.com/biomet/article-abstract/35/3-4/246/280278) — Anscombe 1948
- [IEEE TIP 20(1):99–109](https://pubmed.ncbi.nlm.nih.gov/20615809/) · [TUNI project page](https://webpages.tuni.fi/foi/invansc/) — Mäkitalo & Foi 2011
- [NIST record, SPIE 2725](https://www.nist.gov/publications/sem-performance-evaluation-using-sharpness-criterion) · [SPIE Digital Library](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/2725/0000/SEM-performance-evaluation-using-the-sharpness-criterion/10.1117/12.240107.short) — Postek & Vladár 1996
- [Scanning 20:1–9 (Part I)](https://onlinelibrary.wiley.com/doi/abs/10.1002/sca.1998.4950200101) · [Scanning 20:24–34 (Part II)](https://dx.doi.org/10.1002/sca.1998.4950200104) — Postek/Vladár sharpness series
- [NIST record, "Does Your SEM Really Tell the Truth?" Part 1](https://www.nist.gov/publications/does-your-sem-really-tell-truth-how-would-you-know-part-1) · [PMC5486231 (Part 4)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5486231/) — Postek & Vladár 2013 / 2015
- [IEEE Xplore, IEDM 2008](https://ieeexplore.ieee.org/abstract/document/4796820) — Schloesser et al. 2008
- [IEEE IEDM 2023 memory highlights](https://x.com/ieee_iedm/status/1720085185365168475) — Ha et al. 2023, paper 6.3
- [IEEE Xplore, VLSIT 2012](https://ieeexplore.ieee.org/document/6242496) · [paper PDF, pp. 131–132](https://docs.ampnuts.ru/eevblog.docs/_Metrology/auth2012.pdf) — Auth et al. 2012
- [IEEE Xplore, IEDM 2017](https://ieeexplore.ieee.org/document/8268472/) — Auth et al. 2017
- [IEEE TIT 16(1):41–46](https://ieeexplore.ieee.org/document/1054406/) · [dblp record](https://dblp.org/rec/journals/tit/Chow70.html) — Chow 1970
- [JRSS-B 20(2):215–232](https://academic.oup.com/jrsssb/article/20/2/215/7027376) · [Wiley DOI record](https://rss.onlinelibrary.wiley.com/doi/10.1111/j.2517-6161.1958.tb00292.x) — Cox 1958
- [JMLR 12:2825–2830](https://dblp.org/rec/journals/jmlr/PedregosaVGMTGBPWDVPCBPD11.html) · [scikit-learn citation page](https://scikit-learn.org/stable/about.html) — Pedregosa et al. 2011
