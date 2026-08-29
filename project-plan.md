# DDERA — Project Plan

A phased roadmap. Each phase has an explicit **gate** that must pass before the next begins.

Build order is incremental by design: a demonstrable ante-hoc pipeline early (Phase 3), then the
research-grade depth layered on top. Per Invariants 7, 8 and 10, **early milestones reduce scale,
never methodology.**

**Legend:** `[ ]` todo · `[~]` in progress · `[x]` done · 🔒 gate

---

## Phase 0 — Foundation & GPU bring-up · *Days 1–2*

### 0.1 Repository scaffold
- [x] Directory structure
- [x] `CLAUDE.md` — invariants, stack, workflows, conventions
- [x] `decisions.md` — ADR-001…010 seeded
- [x] `ARCHITECTURE.md` — pathway, module map, artifact schema
- [x] `project-plan.md` — this file
- [x] `README.md` — badges, pitch, quickstart, disclaimer
- [x] `pyproject.toml`, `requirements/`, `.gitignore`, `.pre-commit-config.yaml`
- [x] `src/ddera` package skeleton (data, eval, xai implemented; 195 tests passing)
- [ ] Virtual environment (3.11 on Windows analysis box, 3.12 on the Linux training box)

### 0.2 GPU bring-up 🔒
Ordered ladder — stop at the first rung that passes the full gate; record versions in `decisions.md`
(ADR-009). **Do not install an old ROCm version on the basis of forum reports.** Test current
supported combinations first; any step down must be justified by an observed, recorded failure.

- [ ] Rung 1 — Ubuntu 24.04.x + current stable ROCm + matching PyTorch ROCm wheel,
      `HSA_OVERRIDE_GFX_VERSION=10.3.0` — **step-by-step: [`docs/SETUP-LINUX-ROCM.md`](docs/SETUP-LINUX-ROCM.md)**
- [ ] Rung 2 — `rocm/pytorch` Docker image (isolates the stack; easy version sweeping)
- [ ] Rung 3 — step ROCm minor versions down via Docker tags, newest-first
- [ ] Rung 4 — Windows + `torch-directml` (approved fallback, flagged as degraded)

`scripts/verify_gpu.py` — 8 checks against the *actual* workload, not just device detection:

| # | Check | Pass criterion |
|---|---|---|
| 1 | Device visible, `torch.version.hip` | GPU detected, HIP build |
| 2 | Large matmul vs CPU | `allclose` within fp32 tolerance |
| 3 | **conv2d fwd + bwd** (MIOpen — the real risk area) | matches CPU, no crash |
| 4 | **DenseNet-121 fwd+bwd, 224², batch 32, AMP** | completes, finite grads |
| 5 | `BCEWithLogitsLoss` backward | finite, matches CPU |
| 6 | 200-step overfit on 8 fixed images | loss → ~0 |
| 7 | 30-minute soak | no SIGSEGV, no memory growth |
| 8 | VRAM probe → max batch at 224² / 320² | recorded into config |

> 🔒 **GATE 0.** No training phase begins until all 8 checks pass and ADR-009's *Verification
> result* field is filled in with exact versions. If the whole ladder fails, **stop and report** —
> do not silently degrade.

### 0.3 Version control
- [x] `git init`, `.gitignore`, first commit
- [x] `git remote add origin` → the GitHub repo
- [ ] Push (**needs explicit go-ahead**)

### 0.4 Long-lead items — start now, used much later
- [ ] Register for CheXpert (Stanford AIMI research-use agreement)
- [ ] **Start PhysioNet CITI credentialing for VinDr-CXR** — takes days to weeks; Phase 8 is blocked
      on it, so it must be requested in Phase 0

---

## Phase 1 — Data, EDA, splits · *Days 3–5*

- [ ] `01` — acquisition; `manifest.parquet`; integrity + corruption checks
- [ ] `02` — label EDA: positive/negative/uncertain/blank per observation, co-occurrence matrix,
      class imbalance, patient/study/image counts
- [ ] `03` — image EDA: dimensions, aspect ratios, intensity distributions, view mix, duplicates
- [ ] `04` — patient-level splits (70/10/20), stratified; `valid.csv` held as external check
- [ ] `tests/test_splits.py` — **zero patient-ID intersection**, non-negotiable
- [ ] `configs/concepts/chexpert_v1.yaml` — 12 concepts, target, uncertainty policy

**Decision point (rule pre-committed in ADR-003):** if Pneumonia positives in the test split
< ~250, or bootstrap AUROC 95% CI width > 0.10, escalate to the leave-one-out target sweep. The
sweep runs regardless in Phase 5 as the leakage stress test.

> 🔒 **GATE 1.** Splits are patient-disjoint (test passes), the concept/target split is finalised and
> recorded, and the uncertainty policy is implemented and unit-tested.

---

## Phase 2 — Preprocessing & feature cache · *Days 5–6*

- [ ] `05` — preprocessing pipeline; augmentation (**no horizontal flip**, ADR-006)
- [ ] `ddera/features/cache.py` — float16 memmap + `index.parquet` + `fingerprint.json`
- [ ] Extract and cache features for all splits
- [ ] Stale-cache rejection test

> 🔒 **GATE 2.** Cached features load correctly, the fingerprint check rejects a mismatched encoder,
> and a linear probe on cached features reaches sane concept AUROC (sanity, not a result).

---

## Phase 3 — 🎯 MILESTONE M1: professor demo · *Days 6–10*

**The deliverable: a working end-to-end ante-hoc pipeline with real trained numbers, demoable live.**

- [ ] `06` — B0 black-box baseline (DenseNet-121 → target)
- [ ] `07` — M2 sequential CBM: concept head + linear reasoner
- [ ] Concept-quality metrics + target metrics with bootstrap CIs
- [ ] First concept-intervention demonstration
- [ ] `tests/test_cbm_math.py` — contributions sum exactly to the logit
- [ ] Streamlit: Home, Data, Model, **Explainability Lab**

If full-scale training does not fit the timebox, M1 trains on a **stratified patient-disjoint subset**
(~20–30k frontal images). Permitted lever is **scale only** — subset size, resolution, epochs.
Architecture, bottleneck and metric set stay exactly as in the final system, and the decision gets
its own ADR with the *Effect on the research question* field completed.

> 🔒 **GATE M1.** Upload an X-ray → see concepts → see the contribution waterfall → move a slider →
> the prediction changes by exactly `wⱼ·Δcⱼ` in logit space. Live, from real trained weights.

---

## Phase 4 — CBM variants & the trade-off curve · *Weeks 3–4*

- [ ] `08` — M1 independent CBM, M3 joint CBM
- [ ] λ sweep {0.1, 0.25, 0.5, 1, 2, 5, 10} → **trade-off curve**
- [ ] `09` — M4 hybrid/residual, k ∈ {0, 4, 8, 16, 32, 64} → **completeness curve**
- [ ] Full-scale runs of B0 and M2 (if M1 used a subset)
- [ ] `12` — trade-off and leakage analysis
- [ ] Headline figure: predictive performance vs interpretability, B0 as the ceiling

> 🔒 **GATE 4.** The core research question has a numeric answer: *how much predictive performance
> is lost by forcing the decision through 12 clinical concepts?* — with confidence intervals.

---

## Phase 5 — The XAI evaluation protocol · *Weeks 4–5*

The reusable methodological contribution. All of `src/ddera/xai`, domain-agnostic by construction.

- [ ] `10` — intervention experiments: TTI curves under 4 orderings (random, by uncertainty,
      by `|wⱼ|`, oracle-worst-first)
- [ ] `11` — the protocol: concept quality · leakage (residual probe + permutation necessity) ·
      faithfulness (`∂p/∂cⱼ` vs `wⱼ`) · stability · calibration · bootstrap CIs
- [ ] Leave-one-out target sweep as the leakage stress test
- [ ] `13` — error analysis: the five failure buckets
- [ ] `reporting/runs.py` enforces all 8 metric families before marking a run complete

> 🔒 **GATE 5.** Every claim about interpretability is backed by a number with a CI. Any concept
> whose permutation test shows no effect is reported as decorative — findings are not filtered.

---

## Phase 6 — Uncertainty-aware CBM · *Week 6*

- [ ] `14` — M5: model concept uncertainty explicitly (rather than masking `-1`), propagate to the
      reasoner
- [ ] Concept-query policy: which concept, verified first, most reduces diagnostic uncertainty?
- [ ] Uncertainty-ordered curve added to the TTI plot
- [ ] U-Zeros / U-Ones sensitivity analysis (ADR-004 obligation)

---

## Phase 7 — Post-hoc baselines · *Weeks 6–7*

- [ ] `15` — Grad-CAM, SHAP, LIME on **B0 only** (Invariant 5)
- [ ] Comparison: post-hoc saliency cannot be intervened upon and cannot be validated the way
      Phase 5 validates the concept pathway — the argument for ante-hoc, made with evidence
- [ ] *Optional:* ProtoPNet as a second ante-hoc family, showing the harness generalises across
      interpretable architectures

---

## Phase 8 — VinDr-CXR generalization study · *Weeks 7+*

- [ ] `16` — VinDr-CXR: 22 local findings (concepts) → 6 global diagnoses (targets)
- [ ] `configs/concepts/vindr_v1.yaml`
- [ ] Re-run the full protocol unchanged

> 🔒 **GATE 8 (Invariant 9).** Success = a new domain requires **only a new config**. If it requires
> changes to `src/ddera/xai`, the generality claim has failed and we report that. **No generality
> claim appears in the README until this gate passes.**

---

## Phase 9 — Dashboard & deployment · *ongoing, finalised last*

- [ ] All five pages complete and reading live from run artifacts
- [ ] Custom theme + CSS — anti-slop: real type scale, one accent colour, generous spacing
- [ ] Permanent research-use-only disclaimer
- [ ] README results tables filled with real measured numbers
- [ ] Notebook HTML exports to `reports/`
- [ ] Deploy to Hugging Face Spaces

---

## Milestone summary

| Milestone | Phases | Deliverable |
|---|---|---|
| **M0** — Foundation | 0 | Repo, docs, verified GPU |
| **M1** — 🎯 Professor demo | 1–3 | Live end-to-end ante-hoc pipeline, real numbers |
| **M2** — Core result | 4 | The interpretability↔accuracy trade-off curve |
| **M3** — Methodology | 5 | The full, reusable XAI evaluation protocol |
| **M4** — Extension | 6–7 | Uncertainty-aware CBM + post-hoc comparison |
| **M5** — Generality | 8 | Cross-domain validation on VinDr-CXR |
| **M6** — Delivery | 9 | Polished dashboard, report, deployment |

---

## Risk register

| Risk | Impact | Mitigation |
|---|---|---|
| ROCm fails on gfx1031 | Blocks all training | Four-rung ladder (ADR-009); DirectML fallback approved; escalate rather than degrade silently |
| Pneumonia too sparse | Wide CIs on the headline metric | Escalation rule pre-committed in ADR-003; leave-one-out sweep runs regardless |
| Concept↔target leakage | CBM looks good trivially | Phase 5 leakage quantification is mandatory; contaminated pairs reported as findings |
| CBM underperforms B0 badly | Looks like failure | It is *the research question*, not a failure. Report the gap plainly (CLAUDE.md §8) |
| VinDr credentialing slow | Blocks Phase 8 | Started in Phase 0, weeks ahead of need |
| 12 GB VRAM ceiling | Limits batch/resolution | AMP, gradient accumulation, checkpointed DenseNet, cached features (ADR-008) |
| Notebook sprawl | Unreproducible results | Notebooks import from `src`; configs drive every experiment; artifacts are the source of truth |
