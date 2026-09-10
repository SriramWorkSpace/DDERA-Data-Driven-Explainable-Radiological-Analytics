# Data Acquisition

How to obtain the two datasets DDERA needs, and how to feed them into the pipeline that is
already built. Nothing here can be automated: both require a human to register, agree to a
licence, and (for VinDr) complete training. **Start both now** — the code is ready and
waiting, and the VinDr credentialing has weeks of lead time (ADR-002).

| Dataset | Used in | Access | Lead time |
|---|---|---|---|
| **CheXpert v1.0-small** (~11 GB) | Phases 1–7 (the primary case study) | Stanford AIMI account + Research Use Agreement | hours–days |
| **VinDr-CXR** (~18 k studies) | Phase 8 only (the generality claim) | PhysioNet credentialed account + CITI training | **days–weeks** |

---

## 1. CheXpert — the blocker for everything past Phase 5

### 1.1 What the pipeline expects

`scripts/get_data.py --dest <DIR>` and `ddera.data.acquire.inspect_download` look for either
of these layouts under `<DIR>`:

```
<DIR>/                              <DIR>/CheXpert-v1.0-small/
└── CheXpert-v1.0-small/            ├── train.csv
    ├── train.csv                   ├── valid.csv
    ├── valid.csv                   ├── train/
    ├── train/  patient00001/...    │   └── patientNNNNN/studyN/view1_frontal.jpg
    └── valid/  patient64541/...    └── valid/
```

- `train.csv` / `valid.csv` — the 14-observation label tables. `valid.csv` is the 234-study
  radiologist-consensus set; the pipeline holds it out as `external.parquet` (ADR-005).
- The `train/` image tree — needed for feature extraction and the stability family. The
  `Path` column in the CSV is `CheXpert-v1.0-small/train/patientNNNNN/studyN/viewN_*.jpg`,
  and `--dest` is whatever directory makes that path resolve.
- Use the **small** variant (`~390×320`, ~11 GB), not the full-resolution set (~440 GB).
  DenseNet-121 downsamples to 224² anyway (`configs/model/encoder_v1.yaml`).

Recommended location: `data/chexpert/` (gitignored via `/data/`).

### 1.2 Registration and download

The canonical source is **Stanford AIMI Shared Datasets**:
<https://stanfordaimi.azurewebsites.net/datasets/8cbd9ed4-2eb9-4565-affc-111cf4f7ebe2>
(linked from <https://aimi.stanford.edu/datasets/chexpert-chest-x-rays>).

Steps (follow whatever the site shows now — the UI changes):

1. **Create / sign in to a Stanford AIMI account** ("Web Login" on the dataset page). A
   research or institutional email is expected; a PI / supervisor name may be requested.
2. **Read and accept the Research Use Agreement.** Its substance: non-commercial,
   non-clinical research only; no redistribution; no attempt to re-identify patients;
   citation required. These map onto CLAUDE.md §6 and are already reflected in the README.
3. **Download.** Stanford AIMI serves via **Azure Blob Storage**. You will get either a
   time-limited download link or an `azcopy` command. The classic form:
   ```bash
   # azcopy is the fast path for the Azure blob; the site gives you the exact SAS URL
   azcopy copy "<SAS-URL-for-CheXpert-v1.0-small>" "data/chexpert/" --recursive
   ```
   A browser download of the `CheXpert-v1.0-small.zip` also works; then
   `unzip CheXpert-v1.0-small.zip -d data/chexpert/`.
4. **Verify the extraction:**
   ```bash
   ls data/chexpert/CheXpert-v1.0-small/           # train.csv valid.csv train/ valid/
   wc -l data/chexpert/CheXpert-v1.0-small/train.csv   # ~223k + header
   du -sh data/chexpert/                            # ~11 GB
   ```

> **Compliant mirrors.** Stanford also gates the data on Hugging Face
> (`StanfordAIMI/CheXpert-v1.0-512`, `StanfordAIMI/CheXpert-v1.0-small` if present) — same
> licence, same agreement click-through, sometimes easier tooling (`huggingface-cli
> download`). Third-party Kaggle copies exist but **do not** carry the agreement; per
> CLAUDE.md §6 and ADR-002, use a Stanford-controlled source.

### 1.3 Disk and time budget

| Item | Size |
|---|---|
| `data/chexpert/` (download + extracted) | ~11 GB (~22 GB transient if you keep the zip) |
| `data/processed/` (manifest + splits + sidecars) | < 50 MB |
| `data/features/` (fp16 memmap, N×1024, 3 splits) | < 1 GB |
| Checkpoints (`experiments/runs/*/checkpoint.pt`) | ~30 MB/run |

131 GB is free on `/home` — ample. Download is bandwidth-bound (~15–60 min typical).

---

## 2. VinDr-CXR — start the credentialing clock now (Phase 8)

VinDr-CXR is only needed for the Phase 8 generality study, but ADR-002 says to start it in
Phase 0 because PhysioNet credentialing takes weeks.

1. **PhysioNet account** — <https://physionet.org/register/>.
2. **CITI "Data or Specimens Only Research" training** — <https://physionet.org/about/citi-course/>.
   Complete the course on the CITI site, then submit the completion report to PhysioNet.
   Credentialing review is typically 1–3 weeks.
3. Once credentialed, accept the VinDr-CXR DUA:
   <https://physionet.org/content/vindr-cxr/1.0.0/> and download (`wget -r -N -c -np` with
   your PhysioNet credentials, or the `physionet` client).
4. Target location: `data/vindr/` (gitignored). Phase 8 adds
   `configs/concepts/vindr_v1.yaml`; no `src/ddera/xai` change is permitted (Invariant 9).

---

## 3. Once CheXpert is on disk — the runbook

Everything below is already built and tested on synthetic data. These are the real-data
invocations.

```bash
source .venv/bin/activate
export HSA_OVERRIDE_GFX_VERSION=10.3.0

# 1. Manifest + patient-level splits + integrity report  (Phase 1; ~1 min, CPU)
python scripts/get_data.py --dest data/chexpert --out data/processed
#   -> data/processed/{manifest.parquet, splits.parquet, external.parquet} + JSON sidecars
#   Check the printed ADR-003 escalation advisory: if test Pneumonia positives < 250,
#   the leave-one-out target sweep becomes mandatory.

# 2. EDA / progress figures from the real cohort  (Phase 1/2)
python scripts/visualize_dataset.py --processed data/processed --out reports/figures/dataset

# 3. Feature cache for the frozen-encoder variants  (Phase 2; GPU)
#    NOTE: scripts/extract_features.py does not exist yet — it is the one code gap between
#    "data downloaded" and "training runnable". It is ~30 lines: DenseNet121Encoder(
#    EncoderConfig()) over CheXpertImageDataset for train/val/test via FeatureCache.extract.
#    Build it (or extend get_data.py) as the first real-data task.

# 4. Fill the data locations in the experiment configs
#    configs/experiment/m2_sequential.yaml:
#      splits_parquet: data/processed/splits.parquet
#      data_root:      data/chexpert
#      feature_cache:  data/features

# 5. Smoke test, then train  (Phase 3; GPU)
pytest tests/ -q                                             # expect 302 passed
python -m ddera.train --config configs/experiment/m2_sequential.yaml --fast-dev-run
python -m ddera.train --config configs/experiment/b0_baseline.yaml   # the ceiling (GPU fine-tune)
python -m ddera.train --config configs/experiment/m2_sequential.yaml \
       --baseline-run experiments/runs/<b0-run-id>              # real M1/GATE-M1 numbers

# 6. The dashboard now shows real runs
streamlit run app/Home.py
```

GATE 4 (the λ / k sweeps → trade-off + completeness curves) and GATE 5 (the full protocol as
*results*) then become Phase 4 / Phase 5 execution — the machinery is in place.

---

## 4. Checklist

- [ ] Stanford AIMI account created
- [ ] CheXpert Research Use Agreement accepted
- [ ] `CheXpert-v1.0-small` downloaded and extracted to `data/chexpert/`
- [ ] `scripts/get_data.py --dest data/chexpert` runs clean (no patient leakage)
- [ ] `scripts/extract_features.py` written and features cached
- [ ] experiment configs point at the real `data/processed` / `data/features`
- [ ] PhysioNet account + CITI training started (VinDr, Phase 8 — do this in parallel, now)
