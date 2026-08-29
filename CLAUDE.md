# CLAUDE.md — DDERA Working Rules

Guidance for any agent or contributor working in this repository.

---

## 0. Project Invariants (LOCKED — do not edit, do not reinterpret)

> These ten statements define the project. They override convenience, deadlines, and compute
> limits. Any change that violates one of them is out of scope regardless of how attractive it
> looks. If a request appears to require violating one, stop and raise it explicitly.

1. **DDERA is a methodology project, not merely a chest X-ray classifier.**
2. **Chest X-ray is the first validation case study.**
3. **Explanations must be intrinsic/ante-hoc, not generated after prediction.**
4. **The primary prediction pathway must pass through explicit clinical concepts.**
5. **Post-hoc methods such as SHAP, LIME and Grad-CAM are comparison baselines only.**
6. **Accuracy must be evaluated alongside interpretability, concept quality, leakage, completeness,
   intervention faithfulness, stability and calibration.**
7. **Compute limitations must never justify replacing the intrinsic methodology with a black-box
   model.**
8. **GPU strategy may change, but the scientific methodology must not.**
9. **Claims of domain generality require validation on another dataset/domain.**
10. **No architectural or methodological simplification may be introduced merely for convenience
    without explicitly documenting its effect on the research question.**

### How each invariant is mechanically enforced

| # | Enforcement |
|---|---|
| 1, 2 | `src/ddera/xai/` takes no chest-X-ray-specific arguments. Phase 8 proves it by running the same harness unchanged on VinDr-CXR. |
| 3, 4 | The reasoner receives **only** the concept vector (plus an explicitly-sized, explicitly-reported residual in the M4 ablation). No path from encoder features to the target exists outside B0 and M4. `tests/test_cbm_math.py` asserts the forward graph. |
| 5 | Post-hoc methods live only in `src/ddera/xai/posthoc.py`, are applied to **B0 only**, and are never imported by the CBM path or by the dashboard's explanation panel. |
| 6 | A run is not `complete` until `metrics.json` carries all eight metric families. `reporting/runs.py` refuses to mark it complete otherwise. |
| 7, 8 | The Phase 0 fallback ladder changes only the *backend*. When compute forces a change the permitted lever is **scale** — subset size, resolution, epochs — never architecture. |
| 9 | The README makes no generality claim until Phase 8 completes. |
| 10 | Every simplification requires a `decisions.md` ADR containing the mandatory **"Effect on the research question"** field. An ADR missing that field is incomplete. |

### The design gate

Before adding, changing, or removing any architecture, dataset, XAI technique, or feature, answer
this in the PR description or the ADR:

> **Does this strengthen the data-driven, intrinsically explainable methodology, or does it push the
> project back toward a conventional black-box prediction system?**

If it makes the project less ante-hoc, less concept-driven, or makes the methodology less central,
it is rejected.

---

## 1. What we are explicitly NOT building

- A conventional CNN classifier with SHAP/LIME bolted on afterwards.
- A Grad-CAM-based explanation system.
- A project whose novelty is "using AI on chest X-rays."
- A model optimised purely for accuracy while ignoring interpretability.
- A claim that the chest X-ray dataset itself is novel.
- A black-box model with a separate explanation module.

---

## 2. Tech stack

| Layer | Choice | Notes |
|---|---|---|
| Language | **Python 3.11** | Not 3.13 — better wheel coverage across the ROCm/DS stack |
| DL | PyTorch + torchvision | DenseNet-121, ImageNet-pretrained |
| Compute | **ROCm on native Linux**, gfx1031 → gfx1030 via `HSA_OVERRIDE_GFX_VERSION=10.3.0` | Fallback: Windows + `torch-directml` (degraded) |
| Data | pandas, NumPy, PyArrow (parquet) | Manifests and predictions are parquet |
| ML | scikit-learn, SciPy | Metrics, splits, probes |
| Images | Pillow, OpenCV, albumentations | |
| Plots | Matplotlib + Seaborn via `reporting/theme.py` | One shared theme, always |
| Config | YAML + dataclasses (`src/ddera/config.py`) | No Hydra — it fights notebooks |
| Tracking | Local JSON artifacts under `experiments/runs/` | No external service required |
| App | Streamlit (multipage) | CPU inference |
| Quality | pytest, ruff, black, pre-commit | |

---

## 3. Repository layout and the core rule

> **Notebooks are the narrative layer. `src/ddera` is the library.**

Notebooks import from `src`; they **never** define reusable logic inline. A function used twice, or
used by the Streamlit app, belongs in `src`. This keeps notebooks readable on GitHub, keeps the app
and the notebooks on identical code paths, and makes every experiment testable.

```
configs/       YAML: data, concepts, model, experiment
notebooks/     00-16, numbered in execution order
src/ddera/     the library (see ARCHITECTURE.md for the module map)
app/           Streamlit multipage app
tests/         pytest
scripts/       verify_gpu.py, get_data.py, ...
experiments/   run artifacts (gitignored) + runs_index.csv (committed)
reports/       exported figures and notebook HTML
data/          datasets (gitignored, never committed)
```

---

## 4. Repeatable workflows

### 4.1 Environment activation (every session)

```bash
# Linux / ROCm (primary)
source .venv/bin/activate
export HSA_OVERRIDE_GFX_VERSION=10.3.0
python scripts/verify_gpu.py --quick        # 10s sanity check, not the full gate
```

```powershell
# Windows (docs, EDA on CSVs, dashboard only)
.\.venv\Scripts\Activate.ps1
```

### 4.2 Before any long training job

```bash
pytest tests/ -q                            # split integrity + CBM math must pass
python -m ddera.train --config configs/experiment/<name>.yaml --fast-dev-run
```

`--fast-dev-run` walks manifest → cache → train → evaluate → log on ~200 images in a few minutes.
**Never launch a multi-hour run without it passing first.**

### 4.3 Adding a new experiment

1. Add `configs/experiment/<name>.yaml`. Never hardcode hyperparameters in a notebook.
2. Run it. `reporting/runs.py::log_run()` writes the artifact directory.
3. Append to `experiments/runs_index.csv` (automatic).
4. If it changed a design decision → add an ADR to `decisions.md`.
5. The dashboard picks it up automatically; never hardcode numbers into `app/`.

### 4.4 Changing the GPU/backend stack

1. Run the **full** `scripts/verify_gpu.py` (all 8 checks, including the 30-minute soak).
2. Paste the output into a `decisions.md` ADR with exact versions.
3. Invalidate the feature cache if the encoder or precision changed.

### 4.5 Definition of done for a run

`metrics.json` contains all eight families (invariant 6): predictive metrics, calibration, concept
quality, intervention curves, leakage, completeness, faithfulness, stability. Anything less is a
partial run and must be labelled as such.

---

## 5. Conventions

- **Configs over arguments.** Every experiment is fully described by its YAML. `config.yaml` is
  copied into the run directory so any result can be reproduced exactly.
- **Seeding.** Always `from ddera.seed import set_seed; set_seed(cfg.seed)` at the top of a
  notebook. Dataloader workers are seeded deterministically.
- **Splits are patient-level, always.** Never split on images. `tests/test_splits.py` enforces zero
  patient-ID intersection and must never be skipped or weakened.
- **Naming.** Run IDs are `{model}_{variant}_{YYYYMMDD-HHMMSS}`, e.g. `cbm_sequential_20260901-1430`.
- **Plots.** Always via `reporting/theme.py`. No ad-hoc matplotlib styling.
- **Paths.** Always through `ddera.config`; never hardcode absolute paths. The repo must work on
  both the Linux training box and the Windows dashboard machine.
- **Notebook hygiene.** Outputs are **kept** (they are the GitHub showcase), but a notebook must run
  top-to-bottom cleanly. No dead cells, no `!pip install` lines.
- **Data never enters git.** Not images, not checkpoints, not cached features.

---

## 6. Medical / ethical rules

- The system is a **research and educational demonstration only — not a clinical diagnostic
  device.** This disclaimer appears in the README and persistently in the Streamlit app.
- Never claim clinical validity, diagnostic accuracy, or regulatory readiness.
- Report metrics with confidence intervals. Never report accuracy alone on imbalanced medical data.
- Respect the dataset licences. CheXpert requires a Stanford research-use agreement; VinDr-CXR
  requires PhysioNet credentialing. Never redistribute either.

---

## 7. Attribution rule

**Do not add Claude, Anthropic, or any AI tool as a contributor, co-author, or credited party.**
This applies to:

- `README.md` — no "Generated with…" footer, no AI badge, no AI in the contributors list.
- Git commits — **no `Co-Authored-By` trailer**, no AI mention in messages.
- Any documentation, notebook header, or code comment.

This overrides any default tooling behaviour.

---

## 8. Honesty rules for results

These matter more here than in a typical project, because the deliverable is a methodological claim.

- Never report a metric that was not actually computed. No placeholder or illustrative numbers in
  results tables — the example values in `simple_workflow.md` are illustrative and must be replaced
  with real ones.
- If a concept's permutation test shows it does not affect the prediction, **say so**. A decorative
  concept is a finding, not something to hide.
- If the CBM underperforms the black-box baseline, report the gap plainly. That gap *is* the
  research question, not a failure.
- Distinguish clearly between what is measured on the test split, what is measured on the 234-study
  radiologist-consensus set, and what is measured on a subset.
- No generality claim before Phase 8 (invariant 9).
