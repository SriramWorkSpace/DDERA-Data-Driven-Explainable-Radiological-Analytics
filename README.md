<div align="center">

# DDERA

### Data-Driven Explainable Radiological Analytics

**A methodology for building predictive models that are explainable *by design* — not explained
after the fact.**

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![ROCm](https://img.shields.io/badge/AMD%20ROCm-ED1C24?style=flat-square&logo=amd&logoColor=white)](https://rocm.docs.amd.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![pandas](https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square)](https://matplotlib.org/)
[![Ruff](https://img.shields.io/badge/Ruff-D7FF64?style=flat-square&logo=ruff&logoColor=black)](https://docs.astral.sh/ruff/)
[![pytest](https://img.shields.io/badge/pytest-0A9EDC?style=flat-square&logo=pytest&logoColor=white)](https://pytest.org/)

[![Status](https://img.shields.io/badge/status-in%20development-orange?style=flat-square)](project-plan.md)
[![XAI](https://img.shields.io/badge/XAI-ante--hoc%20%2F%20intrinsic-6E56CF?style=flat-square)](ARCHITECTURE.md)
[![Architecture](https://img.shields.io/badge/model-Concept%20Bottleneck-2D9E6E?style=flat-square)](ARCHITECTURE.md)
[![License](https://img.shields.io/badge/license-MIT-blue?style=flat-square)](LICENSE)

</div>

---

> [!IMPORTANT]
> **Research and educational demonstration only. This is not a clinical diagnostic device.**
> It must not be used for medical decision-making.

---

## The idea

Most "explainable AI" in medical imaging is explained *afterwards*. A black-box CNN makes a
prediction, and then SHAP, LIME, or Grad-CAM is asked to guess why. The explanation is a separate
artifact — it never influenced the decision, and it cannot be acted upon.

DDERA inverts that. The model is **constrained to reason through human-interpretable clinical
concepts**, so the explanation *is* the prediction pathway.

```
   POST-HOC XAI                          DDERA (ante-hoc)

   Image                                 Image
     ↓                                     ↓
   Black box                             Vision encoder
     ↓                                     ↓
   Prediction                            Clinical concepts   ← inspectable, and interveneable
     ↓                                     ↓
   Explanation  (guessed afterwards)     Interpretable reasoner
                                           ↓
                                         Prediction
```

Because the final reasoner is linear, the prediction decomposes **exactly**:

```
logit(p) = b + Σⱼ wⱼ · cⱼ
```

No attribution method. No approximation. Each concept's contribution is arithmetic identity — which
also means you can **intervene**: change a concept, and the prediction must move by `wⱼ·Δcⱼ`.

## What this project actually is

> **The uniqueness is in the methodology, not the dataset.**

DDERA is a **methodology project**. Chest X-ray is its first validation case study, chosen because
public chest radiography datasets carry meaningful clinical concept labels — not because chest
X-ray is the point.

The contribution is a reusable, domain-agnostic framework for **constructing and evaluating**
intrinsically explainable models, consisting of:

1. **A domain-agnostic XAI evaluation harness** — concept quality, leakage quantification,
   intervention curves, empirical concept-completeness, faithfulness, stability. No chest-X-ray
   assumptions anywhere in `src/ddera/xai`.
2. **An empirical interpretability↔accuracy trade-off curve** — a joint-CBM λ sweep and a
   hybrid-residual sweep that answer the core research question *with numbers*, against a black-box
   ceiling.
3. **Uncertainty-aware concept modelling** — exploiting CheXpert's uncertain (`-1`) labels rather
   than discarding them, yielding a concept-query policy: *which single concept should a radiologist
   verify first?*

### Explicitly not this

- ❌ A CNN classifier with SHAP/LIME bolted on afterwards
- ❌ A Grad-CAM explanation system
- ❌ "We used AI on chest X-rays"
- ❌ Accuracy optimised at the expense of interpretability
- ❌ Any claim that the dataset itself is novel

SHAP, LIME and Grad-CAM appear in this project **only as comparison baselines** against a black-box
model — never as DDERA's explanation mechanism.

## Core research question

> **Can a data-driven, intrinsically explainable model maintain competitive predictive performance
> while forcing its decisions through clinically meaningful concepts?**

Supporting questions: Are the learned concepts predictive? Does the prediction genuinely *depend* on
them? Are concept-based explanations stable? How close does the interpretable model get to the
black box? What is the trade-off? Does the methodology transfer to another domain?

## Architecture

```
Chest X-ray  (224 × 224)
      ↓
DenseNet-121 encoder          ImageNet-pretrained, classifier removed
      ↓
Visual representation         f ∈ ℝ¹⁰²⁴
      ↓
Concept head                  Linear(1024 → 12) + σ
      ↓
Concept vector  c ∈ [0,1]¹²   ← the bottleneck: everything downstream sees only this
      ↓
Interpretable reasoner        p = σ(w·c + b)
      ↓
Prediction
```

**Concepts (12):** Enlarged Cardiomediastinum · Cardiomegaly · Lung Opacity · Lung Lesion · Edema ·
Consolidation · Atelectasis · Pneumothorax · Pleural Effusion · Pleural Other · Fracture ·
Support Devices

**Target:** Pneumonia — the one CheXpert label that is a *diagnosis inferred from findings* rather
than a directly visible radiographic sign.

Full detail in [`ARCHITECTURE.md`](ARCHITECTURE.md).

## Model variants

| ID | Model | Purpose |
|---|---|---|
| `B0` | Black-box DenseNet-121 | Accuracy ceiling |
| `M1` | Independent CBM | Max interpretability, min leakage |
| `M2` | Sequential CBM | The practical default |
| `M3` | Joint CBM (λ sweep) | → the **trade-off curve** |
| `M4` | Hybrid/residual CBM (k sweep) | → the **completeness curve** |
| `M5` | Uncertainty-aware CBM | Concept-query policy |

## Evaluation protocol

A run is not complete until all eight metric families are computed:

| Family | Answers |
|---|---|
| Predictive performance | Is it accurate? |
| Calibration | Are its probabilities meaningful? |
| Concept quality | Can it identify its own concepts? |
| Intervention curves | Does correcting concepts help? |
| **Leakage** | Does information bypass the bottleneck? |
| **Completeness** | How much do the concepts fail to carry? |
| **Faithfulness** | Does `∂p/∂cⱼ` match the claimed `wⱼ`? |
| **Stability** | Do concepts survive irrelevant perturbation? |

The last four are what make the interpretability claim *falsifiable* rather than decorative.

## Results

> Results appear here once measured. **No placeholder or illustrative numbers** — see
> [`CLAUDE.md`](CLAUDE.md) §8. Any illustrative values in the original specification are
> illustrative only and will be replaced by real measurements.

## Dataset

**[CheXpert](https://stanfordaimi.azurewebsites.net/datasets/8cbd9ed4-2eb9-4565-affc-111cf4f7ebe2)**
— 224,316 chest radiographs from 65,240 patients, with 14 clinical observation labels. The
`v1.0-small` variant (~11 GB) is used. Requires a Stanford research-use agreement; the dataset is
**not** redistributed in this repository.

Phase 8 adds **[VinDr-CXR](https://physionet.org/content/vindr-cxr/1.0.0/)** (18,000 images,
17 radiologists, 22 local findings + 6 global diagnoses) as a cross-domain validation study. Its
concept/diagnosis split is explicit and human-annotated rather than NLP-extracted, which makes it a
genuinely different domain rather than a re-run.

Handling decisions — patient-level splits, frontal views only, the uncertainty-label policy — are
recorded in [`decisions.md`](decisions.md).

## Quickstart

```bash
git clone https://github.com/SriramWorkSpace/DDERA-Data-Driven-Explainable-Radiological-Analytics.git
cd DDERA-Data-Driven-Explainable-Radiological-Analytics

python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements/base.txt -r requirements/rocm.txt

# Verify the GPU before anything else (see notes below)
export HSA_OVERRIDE_GFX_VERSION=10.3.0
python scripts/verify_gpu.py

# Acquire data (after accepting the Stanford agreement), then run the notebooks in order
python scripts/get_data.py --dest data/chexpert

# Launch the dashboard
streamlit run app/Home.py
```

### Hardware notes

Developed on an AMD **Radeon RX 6800M** (Navi 22, `gfx1031`, 12 GB) under ROCm on native Linux.
gfx1031 is not on AMD's supported list, but `gfx1030` **is** an officially packaged target in
current ROCm and PyTorch's ROCm wheels, so `HSA_OVERRIDE_GFX_VERSION=10.3.0` maps onto shipped
kernels. `scripts/verify_gpu.py` proves the stack works — conv2d fwd/bwd, DenseNet under AMP, an
overfit test and a soak — before any training runs. Details in [`decisions.md`](decisions.md)
ADR-009.

ROCm does **not** support this card on Windows or WSL2 (both are RDNA3/RDNA4 only). NVIDIA CUDA and
CPU are supported transparently via `ddera/device.py`.

Full dual-boot and ROCm procedure: [`docs/SETUP-LINUX-ROCM.md`](docs/SETUP-LINUX-ROCM.md).
Starting a new Claude Code session on that machine? Paste [`docs/RESUME.md`](docs/RESUME.md)
as your first message to bring it up to speed.

## Repository layout

```
configs/       YAML configs — data, concepts, model, experiment
notebooks/     00–16, numbered in execution order (the narrative layer)
src/ddera/     the library (see ARCHITECTURE.md)
app/           Streamlit multipage dashboard
tests/         pytest — split integrity, label policy, CBM math, metrics
scripts/       verify_gpu.py, get_data.py, ...
experiments/   run artifacts + runs_index.csv
```

**Core convention:** notebooks are the narrative layer; `src/ddera` is the library. Notebooks import
from `src` and never define reusable logic inline, so the dashboard and the experiments run on
identical code paths.

## Documentation

| File | Contents |
|---|---|
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Pathway, module map, data flow, artifact schema |
| [`project-plan.md`](project-plan.md) | Phased roadmap with gates, milestones, risk register |
| [`decisions.md`](decisions.md) | ADR log — every design decision and its effect on the research question |
| [`CLAUDE.md`](CLAUDE.md) | Project invariants, conventions, repeatable workflows |

## Scope and limitations

- Chest X-ray is the **first** case study. Generality claims will not be made until validated on a
  second domain (Phase 8).
- CheXpert concept labels are NLP-extracted from radiology reports, not annotated on the images.
  This limits concept-quality claims; VinDr-CXR's human annotations address it directly.
- Frontal views (AP/PA) only in v1; lateral radiographs are out of scope.
- The official CheXpert test labels are not public, so the primary test set is a patient-disjoint
  split carved from the training set. Every reported number is labelled with its source set.
- **Not a clinical diagnostic device.** No claim of clinical validity or regulatory readiness.

## License

Code is released under the [MIT License](LICENSE). The CheXpert and VinDr-CXR datasets carry their
own licences and are **not** redistributed here.

## Citation

If this methodology is useful in your work:

```bibtex
@software{ddera2026,
  title  = {DDERA: Data-Driven Explainable Radiological Analytics},
  author = {Madala, Sriram},
  year   = {2026},
  url    = {https://github.com/SriramWorkSpace/DDERA-Data-Driven-Explainable-Radiological-Analytics}
}
```
