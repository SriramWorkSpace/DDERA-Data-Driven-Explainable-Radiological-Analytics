# Decision Log (ADRs)

Every significant decision in DDERA is recorded here. Per **Invariant 10**, no architectural or
methodological simplification may be introduced for convenience without documenting its effect on
the research question.

## ADR template

```markdown
## ADR-NNN — <Title>
**Date:** YYYY-MM-DD · **Status:** Proposed | Accepted | Superseded by ADR-NNN

**Context.** What forced a decision.
**Decision.** What we chose.
**Rationale.** Why, including options rejected.
**Effect on the research question.** REQUIRED. How this affects DDERA's ability to answer
"can an intrinsically explainable model retain competitive predictive performance?"
Write "None" only when genuinely none, and justify it.
**Consequences.** What this commits us to, and what it costs.
```

An ADR without the **Effect on the research question** field is incomplete and must not be merged.

---

## ADR-001 — Concept Bottleneck Model as the primary architecture
**Date:** 2026-08-30 · **Status:** Accepted

**Context.** DDERA needs an architecture whose explanation is part of the prediction pathway
(Invariants 3, 4), not attached afterwards.

**Decision.** DenseNet-121 vision encoder → concept head (12 clinical concepts) → linear
interpretable reasoner → target. ProtoPNet is a possible secondary ante-hoc family in Phase 7.

**Rationale.** CBMs make the bottleneck explicit and support *concept intervention*, DDERA's
signature experiment. A linear reasoner means the prediction decomposes exactly into `w_j × c_j`
contributions that can be displayed honestly. ProtoPNet explains by prototype similarity — 
interpretable, but it does not support concept-level intervention, hence secondary. Post-hoc
approaches (SHAP/LIME/Grad-CAM) were rejected as the core mechanism because they explain a decision
that has already been made.

**Effect on the research question.** This *is* the research question's subject. The CBM is the
"intrinsically explainable model" whose performance we measure against a black-box ceiling.

**Consequences.** Requires a dataset with concept-level labels. Commits us to reporting a
performance gap versus the black-box baseline, whatever that gap turns out to be.

---

## ADR-002 — CheXpert as the first case study; VinDr-CXR as the generalization study
**Date:** 2026-08-30 · **Status:** Accepted

**Context.** The methodology needs a first validation domain with concept-level labels
(Invariant 2), plus a second domain to support any generality claim (Invariant 9).

**Decision.** CheXpert v1.0-small (~11 GB, ~390×320, 224,316 images / 65,240 patients) is the
primary case study. VinDr-CXR (18,000 images, 17 radiologists, 22 local findings + 6 global
diagnoses) is the Phase 8 generalization study.

**Rationale.** CheXpert provides 14 observation labels usable as concepts, is large, is a standard
benchmark so our numbers are comparable, and its uncertain (`-1`) labels enable the Phase 6
uncertainty-aware extension. VinDr-CXR is a genuinely different validation domain because its
concept/diagnosis split is explicit and human-annotated rather than NLP-extracted from reports.
NIH ChestX-ray14 was rejected as primary: its 14 labels are all disease-level with no
finding/diagnosis hierarchy, which weakens the bottleneck story. MIMIC-CXR was not chosen as
primary due to credentialing latency.

**Effect on the research question.** CheXpert's labels are NLP-extracted from radiology reports, so
"concept" labels are report-derived rather than image-derived. This is a genuine limitation on
concept-quality claims and must be stated in every results discussion. VinDr-CXR in Phase 8
directly addresses it with human annotations.

**Consequences.** Requires a Stanford research-use agreement. PhysioNet CITI credentialing for
VinDr must be started in Phase 0 because it takes weeks.

---

## ADR-003 — Target = Pneumonia; concepts = the 12 radiographic observations
**Date:** 2026-08-30 · **Status:** Accepted (with a pre-committed escalation rule)

**Context.** CheXpert's 14 observations mix radiographic *findings* with diagnoses. A CBM needs a
defensible split between concepts and target, or the bottleneck becomes circular.

**Decision.**
- **Target:** `Pneumonia`.
- **Concepts (12):** Enlarged Cardiomediastinum, Cardiomegaly, Lung Opacity, Lung Lesion, Edema,
  Consolidation, Atelectasis, Pneumothorax, Pleural Effusion, Pleural Other, Fracture,
  Support Devices.
- **Excluded:** `No Finding` (definitionally derivable from the others → guaranteed leakage),
  `Pneumonia` (it is the target).

**Escalation rule, committed in advance:** if EDA shows fewer than ~250 Pneumonia positives in the
test split, or the bootstrap AUROC 95% CI is wider than 0.10, escalate to a leave-one-out target
sweep (Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion, each predicted from the
remaining 13 observations).

**Rationale.** Pneumonia is the only CheXpert label that is a true *diagnosis* inferred from
findings rather than a directly visible radiographic sign. That mirrors real clinical reasoning:
findings → diagnosis. The escalation rule is fixed in advance so the choice stays data-driven
rather than being made after seeing results.

**Effect on the research question.** Some concepts are strongly associated with the target by
clinical definition — Consolidation and Lung Opacity are radiographic evidence *of* pneumonia. This
risks a CBM that looks excellent trivially. That is precisely why the Phase 5 leakage
quantification is mandatory, and why the leave-one-out sweep runs regardless: it doubles as a
leakage stress test by including pairs contaminated by construction (e.g. Enlarged
Cardiomediastinum → Cardiomegaly).

**Consequences.** Pneumonia is sparse and heavily uncertain-labelled in CheXpert, so confidence
intervals will be wide. Bootstrap CIs are mandatory on every headline metric.

---

## ADR-004 — Uncertainty policy: U-Mask on concepts, U-Ignore on target
**Date:** 2026-08-30 · **Status:** Accepted

**Context.** CheXpert labels are `1` positive, `0` negative, `-1` uncertain, blank. A policy is
required and it materially affects results.

**Decision.** Primary: **U-Mask** on concepts (masked BCE ignoring `-1` entries) and **U-Ignore** on
the target. Blank → negative, per CheXpert convention. U-Zeros and U-Ones are run as a sensitivity
analysis and reported.

**Rationale.** Masking avoids inventing a label where the radiologist expressed genuine uncertainty.
U-Zeros/U-Ones inject a systematic bias into exactly the concepts we then claim to explain with.
Sensitivity analysis shows whether the choice materially changes conclusions.

**Effect on the research question.** Discarding uncertain concept labels reduces training signal and
may lower concept AUROC. It also *preserves the semantics* of the concepts, which matters more than
raw concept accuracy for an interpretability claim. Phase 6 revisits this by modelling the
uncertainty explicitly instead of masking it, turning this limitation into a contribution.

**Consequences.** Loss functions must support per-element masking. The sensitivity analysis is
mandatory, not optional.

---

## ADR-005 — Patient-level splits, frontal views only
**Date:** 2026-08-30 · **Status:** Accepted

**Context.** The same patient has multiple studies and views. Random image-level splitting leaks
patients across splits and inflates every metric.

**Decision.** Patient-level `GroupShuffleSplit`, stratified on the target, 70/10/20 from CheXpert
`train`. The official `valid.csv` (234 studies, radiologist consensus) is held separately as a small
external check. **Frontal views only (AP + PA)** for v1; lateral excluded. `tests/test_splits.py`
asserts zero patient-ID intersection and runs in CI.

**Rationale.** Patient-level splitting is the standard requirement in medical imaging, and its
absence is the single most common source of overstated results in this literature. Frontal-only
keeps the v1 encoder input distribution consistent; mixing views without a view-conditioned encoder
adds variance without adding to the methodological claim.

**Effect on the research question.** Excluding laterals reduces dataset size and means the model is
not evaluated on lateral radiographs. Since the research question concerns the concept bottleneck
rather than maximal coverage, this does not weaken the claim — but the scope limit must be stated in
the README and the report.

**Consequences.** Official CheXpert test labels are not public, so our primary test set is the
patient-disjoint 20% carved from `train`. Every number must be labelled with which set it came from.

---

## ADR-006 — No horizontal flip in augmentation
**Date:** 2026-08-30 · **Status:** Accepted

**Context.** Horizontal flip is a near-default augmentation in computer vision.

**Decision.** **No horizontal flip.** Augmentation is ShiftScaleRotate (±7°, ±5%) plus
brightness/contrast jitter only.

**Rationale.** Chest anatomy is not left-right symmetric. The cardiac silhouette sits left of
midline, and the heart's apparent size relative to the thorax is exactly what the Cardiomegaly and
Enlarged Cardiomediastinum concepts measure. Flipping produces anatomically impossible images
(effectively situs inversus) and teaches the encoder that laterality is irrelevant.

**Effect on the research question.** Directly protective. Flip augmentation would corrupt the
semantics of two of our twelve concepts, undermining concept-quality metrics and making the
interpretability claim unsound. Slightly less augmentation diversity is an acceptable price.

**Consequences.** Slightly higher overfitting risk, compensated with rotation/scale/intensity
augmentation and early stopping.

---

## ADR-007 — Python 3.11, not 3.13
**Date:** 2026-08-30 · **Status:** Accepted

**Context.** The machine has Python 3.13.7 (default) and 3.11 installed.

**Decision.** Python 3.11 for the project virtual environment.

**Rationale.** Better wheel coverage across the PyTorch-ROCm / albumentations / OpenCV stack; AMD's
current ROCm PyTorch documentation targets 3.12 and below. 3.11 is already installed, so no cost.

**Effect on the research question.** None. Runtime choice only.

**Consequences.** All environment docs and CI pin 3.11.

**Amendment (2026-08-30).** Ubuntu 24.04 ships Python 3.12, and AMD validates ROCm PyTorch
against 3.12 on that release. Forcing 3.11 onto the Linux training box would mean building a
Python outside AMD's validated combination for no benefit. `pyproject.toml` therefore accepts
`>=3.11,<3.13`: **3.11 on the Windows analysis/dashboard machine, 3.12 on the Linux training
box.** Both are covered by the same test suite. Still no effect on the research question.

---

## ADR-008 — Cached encoder features as the standard experiment substrate
**Date:** 2026-08-30 · **Status:** Accepted

**Context.** Phases 4–6 require many CBM variants (independent, sequential, joint λ sweep, hybrid-k
sweep, uncertainty-aware) plus a large XAI protocol. Re-running the CNN for each is wasteful on a
12 GB laptop GPU.

**Decision.** Run the frozen DenseNet-121 once over each split and cache the 1024-d feature vectors
(float16 memmap + `index.parquet`). All frozen-encoder variants train on cached features.
`cache.py` stores an encoder fingerprint (weights hash + resolution + normalization) and refuses to
serve a stale cache.

**Rationale.** Feature extraction is the only genuinely GPU-bound step for frozen-encoder variants.
Once cached, the concept head and reasoner train in seconds to minutes, which makes the full sweep
schedule feasible locally and the XAI protocol cheap to iterate on.

**Effect on the research question.** **None for frozen-encoder variants** — the computation is
mathematically identical, merely memoised. One important limitation: the cache is invalid for joint
CBM training and the black-box fine-tune, because those update the encoder. Those runs go full-GPU
and must not use the cache. This is enforced by the fingerprint check, not by discipline.

**Consequences.** Disk cost for cached features. A mandatory invalidation step whenever the encoder,
resolution, or normalization changes.

---

## ADR-009 — Local GPU (RX 6800M / gfx1031) via ROCm on native Linux
**Date:** 2026-08-30 · **Status:** Proposed — pending the Phase 0 verification gate

**Context.** Local GPU training is a hard requirement; cloud GPU is excluded. The machine is an ASUS
ROG Strix G15 Advantage Edition with a Radeon RX 6800M (Navi 22, **gfx1031**, 12 GB). Investigation
of current AMD documentation found:

| Path | Status for gfx1031 |
|---|---|
| ROCm on Windows | Not supported — AMD's Windows matrix lists RDNA3/RDNA4 only (RX 9070/9060 XT, R9700, 7900 XTX, W7900, RX 7700) |
| ROCm on WSL2 | Not supported — covers 7900-series / 9070 / W7900 / Strix Halo only |
| **ROCm on native Linux** | **Primary path.** Current ROCm 7.x officially packages the **gfx1030** target (the supported PRO W6800 and V620 are gfx1030 cards) and PyTorch's ROCm wheels build gfx1030. gfx1031 loads those code objects via `HSA_OVERRIDE_GFX_VERSION=10.3.0` |
| Windows + `torch-directml` | Approved fallback. DirectML is in maintenance mode on an older PyTorch with op-coverage gaps |

**Decision.** Primary path is native Linux (Ubuntu 24.04.x HWE) + current stable ROCm 7.x + matching
PyTorch ROCm wheel + `HSA_OVERRIDE_GFX_VERSION=10.3.0`. Ordered fallback ladder: (1) host install,
(2) `rocm/pytorch` Docker, (3) step ROCm minor versions down via Docker tags newest-first,
(4) Windows + `torch-directml`.

Explicitly **not** doing: installing an old ROCm version on the basis of forum reports. Current
supported combinations are tested first, and any step down must be justified by an observed failure
recorded here.

**Rationale.** gfx1030 is an officially compiled target in the current stack, which gives the
override a real basis rather than making it a hack. Known risk: community reports of gfx1031 SIGSEGV
on some ROCm ≥ 6.4.3 builds, largely from llama.cpp/Ollama contexts and unconfirmed for PyTorch conv
workloads — so it must be tested, not assumed.

**Effect on the research question.** **None permitted.** Per Invariants 7 and 8, the backend may
change but the methodology may not. If compute forces a change, the only permitted lever is *scale*
(subset size, resolution, epochs), and any such change requires its own ADR.

**Consequences.** `scripts/verify_gpu.py` is a hard gate: 8 checks including conv2d fwd/bwd
(MIOpen), DenseNet-121 fwd/bwd under AMP, a 200-step overfit test, and a 30-minute soak. No training
phase begins until it passes and its exact versions are recorded here.

**Verification result:** _pending — to be filled in with `verify_gpu.py` output and exact versions._

---

## ADR-010 — Local JSON run artifacts instead of MLflow or W&B
**Date:** 2026-08-30 · **Status:** Accepted

**Context.** The project needs experiment tracking across many variants and sweeps.

**Decision.** Every run writes `experiments/runs/<run_id>/` containing `config.yaml`, `metrics.json`,
`predictions.parquet`, `concept_weights.json`, and a gitignored `checkpoint.pt`, plus a row in the
committed `experiments/runs_index.csv`.

**Rationale.** Zero infrastructure, works offline, git-versioned, and the Streamlit dashboard reads
these files directly so results can never drift out of sync with the app. `predictions.parquet` is
the substrate the entire Phase 5 XAI protocol runs on, so it must exist per-run regardless of the
tracking choice.

**Effect on the research question.** None on results. Positive on reproducibility: the resolved
config is stored alongside the metrics it produced.

**Consequences.** No hosted live training curves. W&B can be added later as an optional mirror
without changing the artifact schema.

---

## ADR-011 - Split prevalence checked against sampling noise, not a fixed tolerance
**Date:** 2026-08-30 - **Status:** Accepted

**Context.** `check_split_integrity` originally flagged any split whose target prevalence
drifted more than 5% relative from the others. On a correct patient-level split of 300
synthetic patients this fired immediately: the val split holds ~10% of patients (~30), where
ordinary binomial noise moves prevalence by 10-15% relative. The check was failing on
correct data.

**Decision.** Convert each split's prevalence to a z-score against the pooled rate, using the
**patient count** as the effective sample size, and flag only when `|z| > 3`.

**Rationale.** A fixed percentage tolerance ignores split size, so it is simultaneously too
strict on small splits and too lenient on large ones. The patient count is the right
effective n because images from one patient are correlated; using the image count would
overstate the available information and make the test over-sensitive. Verified empirically:
a correct split scores |z| < 0.5, while a deliberately skewed one scores |z| ~ 11.

**Effect on the research question.** None on results; this is a data-quality guard, not a
modelling choice. Indirectly protective: a check that cries wolf on correct splits is a check
people learn to ignore, and patient-level split integrity is the guarantee every metric in
this project rests on.

**Consequences.** `check_split_integrity` now reports `prevalence_z_scores` and
`prevalence_pooled` alongside the raw prevalences. Covered by two tests asserting both
directions (no false alarm on a clean split, detection of a deliberate skew).

---

## ADR-012 - Leakage and incompleteness measured as separate quantities
**Date:** 2026-08-30 - **Status:** Accepted

**Context.** The CBM literature routinely uses "leakage" for two different failures. While
testing `soft_vs_hard_leakage` the distinction became concrete: on noisy synthetic concepts
the measure came out *negative*, because thresholding a noisy soft concept moves it back
toward its true binary value. Hardening denoised the bottleneck rather than destroying
smuggled information.

**Decision.** Measure and report the two failures separately, and treat the sign of the
soft-vs-hard measure as meaningful:

- **Leakage** (`soft_vs_hard_leakage`, positive value) - soft concept values encode
  sub-symbolic information a reasoner trained on them learns to exploit. A defect: the
  bottleneck looks intact while the explanation has stopped being true.
- **Denoising** (same measure, negative value) - hardening improves AUROC. A
  concept-calibration signal, not smuggled information.
- **Incompleteness** (`residual_probe_leakage`, and the Phase 4 hybrid-k sweep) - the
  concepts honestly do not carry all task-relevant information. A finding, not a defect.

`ddera.data.synthetic` gained a `soft_leak_strength` knob that injects target information
into concept *probabilities* while leaving their hard thresholds intact, so the positive
direction can be tested rather than assumed.

**Effect on the research question.** Materially sharpens it. Conflating the two would let an
incomplete-but-honest bottleneck be reported as a leaking one (or the reverse), which would
misattribute the interpretability cost. Since quantifying that cost *is* the research
question, the distinction is load-bearing rather than terminological.

**Consequences.** Three regimes are now asserted discriminatively in
`tests/test_xai_harness.py`: positive leakage with a reasoner fitted on leaked soft concepts,
negative on denoising, and near-zero for a confident predictor. Results reporting must quote
the sign and the interpretation string, never `abs(leakage)`.
