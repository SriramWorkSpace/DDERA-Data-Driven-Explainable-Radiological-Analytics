# Graph Report - DDERA  (2026-09-10)

## Corpus Check
- 64 files · ~295,945 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1119 nodes · 1938 edges · 67 communities (50 shown, 10 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 78 edges (avg confidence: 0.9)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `aa8f2ee8`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- safe_auroc
- patient_level_split
- manifest_from_frame
- apply_uncertainty_policy
- ConceptSpec
- TestCalibration
- verify_gpu.py
- TestCompleteness
- GradCAM
- Step 4: Concept Bottleneck Layer
- make_synthetic_cbm
- Final Flow Diagram: CNN-based Concept Bottleneck Model for Ante-hoc Explainable Chest X-ray Diagnosis
- Concept Bottleneck Layer
- stability_report
- tti_curve
- ArrayLike
- intervention.py
- get_device_info
- 10. Concept Intervention (What-If Analysis)
- test_xai_harness.py
- What You Must Do When Invoked
- intervention_order
- Phase 8: VinDr-CXR generalization study
- verify_decomposition
- Phase 3: Milestone M1 professor demo
- The eight-family evaluation protocol
- Concept vector bottleneck c in [0,1]^12
- dataset.py
- LinearReasoner
- test_cbm_math.py
- DDERA (Data-Driven Explainable Radiological Analytics)
- acquire.py
- graphify reference: extra exports and benchmark
- graphify reference: query, path, explain
- DDERA tech stack
- Interpretable linear reasoner p=sigma(w.c+b)
- graphify reference: add a URL and watch a folder
- Invariant 10: simplifications require a documented effect on the research question
- ddera/__init__.py
- Module: eval/bootstrap.py
- Module: eval/calibration.py
- Module: eval/metrics.py
- No AI attribution rule
- ddera
- EncoderConfig
- graphify reference: commit hook and native CLAUDE.md integration
- graphify reference: incremental update and cluster-only
- graphify reference: GitHub clone and cross-repo merge
- graphify reference: transcribe video and audio
- CLAUDE.md
- extraction-spec.md
- build_processed_dataset
- get_data.py
- ProcessedArtifacts
- test_chexpert.py
- concept_matrix
- load_yaml
- EscalationSpec
- parse_chexpert_path
- config.py

## God Nodes (most connected - your core abstractions)
1. `patient_level_split()` - 28 edges
2. `ConceptSpec` - 27 edges
3. `build_processed_dataset()` - 27 edges
4. `LinearReasoner` - 23 edges
5. `manifest_from_frame()` - 21 edges
6. `apply_uncertainty_policy()` - 21 edges
7. `EncoderConfig` - 19 edges
8. `safe_auroc()` - 19 edges
9. `encode_concept_matrix()` - 15 edges
10. `load_splits()` - 15 edges

## Surprising Connections (you probably didn't know these)
- `Linear logit decomposition logit(p)=b+sum(wj*cj)` --semantically_similar_to--> `Interpretable linear reasoner p=sigma(w.c+b)`  [INFERRED] [semantically similar]
  README.md → ARCHITECTURE.md
- `Eight-family evaluation protocol` --semantically_similar_to--> `The eight-family evaluation protocol`  [INFERRED] [semantically similar]
  README.md → ARCHITECTURE.md
- `Original simple_workflow specification document` --semantically_similar_to--> `The eight-family evaluation protocol`  [INFERRED] [semantically similar]
  simple_workflow.md → ARCHITECTURE.md
- `Final project definition (name, architecture, methodology)` --semantically_similar_to--> `DDERA (Data-Driven Explainable Radiological Analytics)`  [INFERRED] [semantically similar]
  simple_workflow.md → README.md
- `The 12 concept list (chexpert_v1)` --semantically_similar_to--> `The 12 clinical concepts (chest X-ray)`  [INFERRED] [semantically similar]
  configs/concepts/chexpert_v1.yaml → README.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Modules implementing the eight-family evaluation protocol** — claude_invariant_6, architecture_module_xai_intervention, architecture_module_xai_leakage, architecture_module_xai_stability, architecture_module_xai_completeness, architecture_module_eval_metrics, architecture_module_eval_calibration, architecture_module_eval_bootstrap [EXTRACTED 1.00]
- **The B0/M1-M5 model variant family** — architecture_b0, architecture_m1, architecture_m2, architecture_m3, architecture_m4, architecture_m5 [EXTRACTED 1.00]
- **The ten locked project invariants** — claude_invariant_1, claude_invariant_2, claude_invariant_3, claude_invariant_4, claude_invariant_5, claude_invariant_6, claude_invariant_7, claude_invariant_8, claude_invariant_9, claude_invariant_10 [EXTRACTED 1.00]
- **Clinical concepts predicted by the Concept Bottleneck Layer** — end_to_end_flow_concept_bottleneck_layer, end_to_end_flow_concept_lung_opacity, end_to_end_flow_concept_consolidation, end_to_end_flow_concept_edema, end_to_end_flow_concept_pleural_effusion, end_to_end_flow_concept_cardiomegaly, end_to_end_flow_concept_atelectasis, end_to_end_flow_concept_pneumothorax, end_to_end_flow_concept_nodule_mass [EXTRACTED 1.00]
- **Evaluation metric families making up the Evaluation stage** — end_to_end_flow_evaluation, end_to_end_flow_predictive_performance_metrics, end_to_end_flow_concept_quality_metrics, end_to_end_flow_overall_goals [EXTRACTED 1.00]
- **Two-stage training pipeline: CNN concept predictor then interpretable disease classifier** — end_to_end_flow_training_strategy, end_to_end_flow_concept_predictor_stage, end_to_end_flow_concept_to_disease_stage, end_to_end_flow_densenet121, end_to_end_flow_interpretable_reasoning_layer [EXTRACTED 1.00]
- **Concept Vector / Interpretable Bottleneck** — flow_diagram_concept_prediction_layer, flow_diagram_concept_representations, flow_diagram_opacity, flow_diagram_consolidation, flow_diagram_edema, flow_diagram_pleural_effusion, flow_diagram_cardiomegaly, flow_diagram_atelectasis [EXTRACTED 1.00]
- **Validation and Delivery Stage** — docs_images_flow_diagram_model_evaluation, docs_images_flow_diagram_explainability_evaluation, docs_images_flow_diagram_deployment_streamlit [INFERRED 0.75]
- **Data & Feature Extraction Pipeline** — flow_diagram_data_acquisition, flow_diagram_data_cleaning_eda, flow_diagram_preprocessing, flow_diagram_cnn_vision_encoder [INFERRED 0.75]
- **Evaluation and Deployment Phase** — flow_diagram_model_evaluation, flow_diagram_explainability_evaluation, flow_diagram_deployment_streamlit_app, flow_diagram_final_output [INFERRED 0.75]
- **Radiological Concepts Predicted by Concept Bottleneck Layer** — docs_images_end_to_end_flow_concept_bottleneck_layer, docs_images_end_to_end_flow_lung_opacity, docs_images_end_to_end_flow_consolidation, docs_images_end_to_end_flow_edema, docs_images_end_to_end_flow_pleural_effusion, docs_images_end_to_end_flow_cardiomegaly, docs_images_end_to_end_flow_atelectasis, docs_images_end_to_end_flow_pneumothorax, docs_images_end_to_end_flow_nodule_mass [INFERRED 0.85]
- **Evaluation Metric Families** — docs_images_end_to_end_flow_evaluation, docs_images_end_to_end_flow_predictive_performance, docs_images_end_to_end_flow_concept_quality_metrics, docs_images_end_to_end_flow_overall_goals [INFERRED 0.85]
- **Two-Stage Training Strategy** — docs_images_end_to_end_flow_training_strategy, docs_images_end_to_end_flow_concept_predictor_stage, docs_images_end_to_end_flow_concept_to_disease_stage [INFERRED 0.85]
- **Concept Bottleneck Model Core Pipeline** — docs_images_flow_diagram_concept_prediction_layer, docs_images_flow_diagram_concept_representations, docs_images_flow_diagram_interpretable_classifier, docs_images_flow_diagram_disease_prediction, docs_images_flow_diagram_ante_hoc_explanation, docs_images_flow_diagram_concept_intervention [INFERRED 0.85]

## Communities (67 total, 10 thin omitted)

### Community 0 - "safe_auroc"
Cohesion: 0.05
Nodes (37): MetricFn, bootstrap_ci(), BootstrapResult, is_significant(), paired_bootstrap_diff(), Any, ArrayLike, Bootstrap confidence intervals. Mandatory on every headline metric. Pneumonia… (+29 more)

### Community 1 - "patient_level_split"
Cohesion: 0.06
Nodes (34): Policy: notebook outputs deliberately kept (no nbstripout), Module: data/splits.py, Repository conventions (configs, seeding, splits, naming, plots, paths), Cohort definition: frontal views, patient-level split, ADR-005: Patient-level splits, frontal views only, ADR-011: Split prevalence checked via z-score, not fixed tolerance, assert_no_patient_leakage(), check_split_integrity() (+26 more)

### Community 2 - "manifest_from_frame"
Cohesion: 0.23
Nodes (4): manifest_from_frame(), Manifest construction from an already-loaded frame. Split out from…, Applying an uncertainty policy is a separate, explicit step -- not done here., TestManifest

### Community 3 - "apply_uncertainty_policy"
Cohesion: 0.05
Nodes (44): BlankPolicy, bool_, ConceptPolicy, _binarize_target(), Apply ADR-004's target policy: drop uncertain-target rows, then map to 0/1., apply_uncertainty_policy(), encode_concept_matrix(), label_distribution() (+36 more)

### Community 4 - "ConceptSpec"
Cohesion: 0.13
Nodes (7): ConceptSpec, The concept/target contract for one domain. This is the object the whole…, Path, Invariant 4: a target inside the bottleneck would be circular., It is derivable from the other observations, so it would leak by construction., Consolidation and Lung Opacity are radiographic evidence *of* pneumonia., TestConceptSpec

### Community 5 - "TestCalibration"
Cohesion: 0.08
Nodes (30): BinStrategy, apply_temperature(), _bin_edges(), brier_score(), calibration_report(), expected_calibration_error(), fit_temperature(), maximum_calibration_error() (+22 more)

### Community 6 - "verify_gpu.py"
Cohesion: 0.06
Nodes (54): Module: xai/leakage.py, GPU/backend environment workflow (4.1, 4.4), Invariant 7: compute limitations must never justify a black-box swap, Invariant 8: GPU strategy may change, methodology must not, The 12 concept list (chexpert_v1), Pre-committed escalation rule to leave-one-out sweep, Expected leakage watchlist (Consolidation, Lung Opacity), CheXpert v1 concept specification file (+46 more)

### Community 7 - "TestCompleteness"
Cohesion: 0.08
Nodes (20): completeness_curve(), completeness_ratio(), completeness_report(), CompletenessCurve, _describe(), Any, Concept completeness. How much of the task-relevant information does the…, Build a curve from a ``{residual_width: auroc}`` mapping (as produced by the… (+12 more)

### Community 8 - "GradCAM"
Cohesion: 0.10
Nodes (19): assert_baseline_only(), comparison_table(), GradCAM, lime_explanation(), Any, ArrayLike, float64, NDArray (+11 more)

### Community 9 - "Step 4: Concept Bottleneck Layer"
Cohesion: 0.09
Nodes (30): Ante-hoc explainability (built into the model, not post-hoc), Real medical data: CheXpert / ChestX-ray14, Step 3: CNN / Vision Encoder, Clinical concept: Atelectasis, Step 4: Concept Bottleneck Layer, Clinical concept: Cardiomegaly, Clinical concept: Consolidation, Clinical concept: Edema (+22 more)

### Community 10 - "make_synthetic_cbm"
Cohesion: 0.08
Nodes (29): make_synthetic_cbm(), ArrayLike, DataFrame, float64, NDArray, Path, Synthetic concept-bottleneck data with known ground truth. Used for testing the…, Write a miniature ``CheXpert-v1.0-small/`` tree: ``train.csv``, ``valid.csv``… (+21 more)

### Community 11 - "Final Flow Diagram: CNN-based Concept Bottleneck Model for Ante-hoc Explainable Chest X-ray Diagnosis"
Cohesion: 0.16
Nodes (29): 9. Ante-hoc Explanation, Atelectasis (clinical concept), Cardiomegaly (clinical concept), CheXpert Dataset, 4. CNN / Vision Encoder, 10. Concept Intervention (What-If Analysis), 5. Concept Prediction Layer, 6. Concept Representations (interpretable bottleneck) (+21 more)

### Community 12 - "Concept Bottleneck Layer"
Cohesion: 0.10
Nodes (26): Atelectasis (concept), Cardiomegaly (concept), CNN / Vision Encoder, Concept Bottleneck Layer, Concept Intervention (Optional, manual concept adjustment), Stage 1: Concept Predictor (CNN, multi-label BCE loss), Concept Quality metrics (concept accuracy, concept F1, human alignment, intervention sensitivity, sufficiency & necessity, stability/noise test), Stage 2: Concept -> Disease (interpretable classifier, cross-entropy loss) (+18 more)

### Community 13 - "stability_report"
Cohesion: 0.13
Nodes (15): concept_drift(), _describe(), prediction_flip_rate(), Any, ArrayLike, rank_stability(), Explanation stability under clinically irrelevant perturbation. An explanation…, Did the decision change? The clinically consequential view. (+7 more)

### Community 14 - "tti_curve"
Cohesion: 0.12
Nodes (14): Ordering, Any, AUROC as a function of how many concepts were corrected to ground truth., AUROC recovered by correcting every concept. Near zero means the reasoner…, Test-time intervention curve. Progressively replaces predicted concepts with…, Run all four orderings -- the standard Phase 5 intervention panel., tti_all_strategies(), tti_curve() (+6 more)

### Community 15 - "ArrayLike"
Cohesion: 0.23
Nodes (13): _as_concept_matrix(), empirical_sensitivity(), logit(), ArrayLike, float64, NDArray, Estimate ``d logit(p) / d c_j`` per concept by central finite differences.…, Numerically stable logistic function. (+5 more)

### Community 16 - "intervention.py"
Cohesion: 0.16
Nodes (11): expected_logit_shift(), intervene(), intervention_effect(), Concept contributions and intervention analysis. This module is the mechanical…, Return a copy of ``concepts`` with column ``index`` set to ``value``. Never…, Measure what one intervention actually does to the prediction. Returns the…, Closed-form logit shift for a linear reasoner: ``w_j * (new - old)``., Raising a positively-weighted concept must raise the prediction, and vice versa. (+3 more)

### Community 17 - "get_device_info"
Cohesion: 0.18
Nodes (13): Backend, DeviceInfo, get_device(), get_device_info(), _probe_amp(), Any, device, Backend resolution. This is the ONLY module in the codebase that branches on… (+5 more)

### Community 18 - "10. Concept Intervention (What-If Analysis)"
Cohesion: 0.16
Nodes (16): 9. Ante-hoc Explanation, CheXpert Dataset, 4. CNN / Vision Encoder, 10. Concept Intervention (What-If Analysis), 5. Concept Prediction Layer, 6. Concept Representations, 1. Data Acquisition, 2. Data Cleaning & EDA (+8 more)

### Community 19 - "test_xai_harness.py"
Cohesion: 0.06
Nodes (36): concept_permutation_necessity(), _describe_incompleteness(), _describe_leakage(), leakage_report(), Any, ArrayLike, float64, NDArray (+28 more)

### Community 20 - "What You Must Do When Invoked"
Cohesion: 0.08
Nodes (24): For /graphify add and --watch, For /graphify query, For the commit hook and native CLAUDE.md integration, For --update and --cluster-only, /graphify, Honesty Rules, Interpreter guard for subcommands, Part A - Structural extraction for code files (+16 more)

### Community 21 - "intervention_order"
Cohesion: 0.26
Nodes (6): int_, apply_intervention_order(), intervention_order(), Build an ``(n, k)`` matrix giving, per sample, the order to intervene on…, Replace the first ``n_intervened`` concepts (per sample, per ``order``) with…, TestInterventionOrdering

### Community 22 - "Phase 8: VinDr-CXR generalization study"
Cohesion: 0.25
Nodes (8): Why xai/ takes no domain arguments, Module: xai/posthoc.py (B0 baselines only), Invariant 1: DDERA is a methodology project, not merely a classifier, Invariant 2: Chest X-ray is the first validation case study, Invariant 5: post-hoc methods are comparison baselines only, Phase 7: Post-hoc baselines, Phase 8: VinDr-CXR generalization study, Phase 9: Dashboard and deployment

### Community 23 - "verify_decomposition"
Cohesion: 0.20
Nodes (7): Assert that contributions plus bias reproduce the logit exactly. Returns the…, verify_decomposition(), logit(p) == bias + sum_j w_j c_j, exactly., All-zeros and all-ones are the boundary cases the dashboard sliders can reach., With every concept at zero the logit must be exactly the bias., The guard must actually fire -- a test that can never fail is worthless., TestDecomposition

### Community 24 - "Phase 3: Milestone M1 professor demo"
Cohesion: 0.16
Nodes (14): B0: black-box DenseNet-121 baseline, Two execution profiles: training vs analysis/demo, Feature-cache invalidation via encoder fingerprint, M1: independent CBM (ground-truth concepts), M2: sequential CBM (predicted concepts, practical default), M3: joint CBM, lambda sweep -> trade-off curve, M4: hybrid/residual CBM, k sweep -> completeness curve, Module: features/cache.py (+6 more)

### Community 25 - "The eight-family evaluation protocol"
Cohesion: 0.25
Nodes (8): The eight-family evaluation protocol, Module: xai/completeness.py, Module: xai/stability.py, Definition of done for a run, Invariant 6: accuracy must be evaluated alongside interpretability metrics, ADR-012: Leakage and incompleteness measured as separate quantities, Eight-family evaluation protocol, Reusable domain-agnostic methodology contribution

### Community 26 - "Concept vector bottleneck c in [0,1]^12"
Cohesion: 0.33
Nodes (6): Concept head: Linear(1024->12)+sigma, Concept vector bottleneck c in [0,1]^12, DenseNet-121 vision encoder stage, Module: models/encoder.py, Invariant 3: Explanations must be intrinsic/ante-hoc, ADR-006: No horizontal flip in augmentation

### Community 27 - "dataset.py"
Cohesion: 0.08
Nodes (16): Dataset, CheXpertImageDataset, EncodedSplit, _LabelledDataset, ndarray, Torch datasets over the Phase-1 artifacts (``splits.parquet``).…, One split of ``splits.parquet`` as ``(image, concepts, concept_mask, target)``.…, Policy-encoded concept labels + target + provenance for one split. Shared by… (+8 more)

### Community 28 - "LinearReasoner"
Cohesion: 0.20
Nodes (7): faithfulness_report(), LinearReasoner, PredictFn, Adapt to the ``predict_fn`` interface the rest of this module consumes., Does the model behave the way its weights claim? Compares the measured…, ``p = sigmoid(w . c + b)`` -- DDERA's primary interpretable reasoner.…, A model whose declared weights are not the weights it uses must be caught.

### Community 29 - "test_cbm_math.py"
Cohesion: 0.18
Nodes (6): The CBM mathematics. These are the correctness-critical tests in the project.…, Does the model behave the way its weights claim?, For a linear reasoner, d logit / d c_j must equal w_j., A linear reasoner has the same derivative everywhere; that is why it is…, Concepts pinned at 0 or 1 must still yield the correct one-sided estimate., TestFaithfulness

### Community 30 - "DDERA (Data-Driven Explainable Radiological Analytics)"
Cohesion: 0.31
Nodes (10): The core ante-hoc prediction pathway, ProtoPNet: optional second ante-hoc family, Honesty rules for results, ADR-001: Concept Bottleneck Model as primary architecture, Ante-hoc vs post-hoc explanation inversion, Core research question, DDERA (Data-Driven Explainable Radiological Analytics), Linear logit decomposition logit(p)=b+sum(wj*cj) (+2 more)

### Community 31 - "acquire.py"
Cohesion: 0.14
Nodes (15): Phase-1 data acquisition orchestration (ADR-002/003/004/005/011). Ties the…, build_manifest(), ImageProbeReport, ManifestSummary, probe_image_dimensions(), Any, Path, CheXpert manifest construction. Turns the raw ``train.csv`` / ``valid.csv``… (+7 more)

### Community 32 - "graphify reference: extra exports and benchmark"
Cohesion: 0.22
Nodes (8): graphify reference: extra exports and benchmark, Step 6b - Wiki (only if --wiki flag), Step 7 - Neo4j export (only if --neo4j or --neo4j-push flag), Step 7a - FalkorDB export (only if --falkordb or --falkordb-push flag), Step 7b - SVG export (only if --svg flag), Step 7c - GraphML export (only if --graphml flag), Step 7d - MCP server (only if --mcp flag), Step 8 - Token reduction benchmark (only if total_words > 5000)

### Community 33 - "graphify reference: query, path, explain"
Cohesion: 0.33
Nodes (5): For /graphify explain, For /graphify path, graphify reference: query, path, explain, Step 0 — Constrained query expansion (REQUIRED before traversal), Step 1 — Traversal

### Community 34 - "DDERA tech stack"
Cohesion: 0.40
Nodes (6): pre-commit hook: ruff (--fix), Notebooks are narrative, src/ddera is the library, DDERA tech stack, ADR-007: Python 3.11 (Windows) / 3.12 (Linux training box), Backend-agnostic core dependencies, Dev/quality dependencies (pytest, ruff, black, pre-commit)

### Community 35 - "Interpretable linear reasoner p=sigma(w.c+b)"
Cohesion: 0.18
Nodes (13): Streamlit application architecture, Concept intervention experiment, Data flow and run artifacts pipeline, Explainability Lab (live concept intervention page), Interpretable linear reasoner p=sigma(w.c+b), M5: uncertainty-aware CBM, Module: models/reasoner.py, Module: reporting/runs.py (+5 more)

### Community 36 - "graphify reference: add a URL and watch a folder"
Cohesion: 0.50
Nodes (3): For /graphify add, For --watch, graphify reference: add a URL and watch a folder

### Community 37 - "Invariant 10: simplifications require a documented effect on the research question"
Cohesion: 0.67
Nodes (3): The design gate question, Invariant 10: simplifications require a documented effect on the research question, ADR template with mandatory Effect-on-research-question field

### Community 48 - "EncoderConfig"
Cohesion: 0.15
Nodes (18): Compose, EncoderConfig, Vision encoder specification (ADR-001). Also the source of the feature-cache…, Training augmentation knobs (ADR-006). There are deliberately **no**…, TransformConfig, assert_no_reflection(), build_eval_transform(), build_train_transform() (+10 more)

### Community 52 - "graphify reference: commit hook and native CLAUDE.md integration"
Cohesion: 0.50
Nodes (3): For git commit hook, For native CLAUDE.md integration, graphify reference: commit hook and native CLAUDE.md integration

### Community 53 - "graphify reference: incremental update and cluster-only"
Cohesion: 0.50
Nodes (3): For --cluster-only, For --update (incremental re-extraction), graphify reference: incremental update and cluster-only

### Community 58 - "build_processed_dataset"
Cohesion: 0.06
Nodes (38): main(), _build_synthetic(), main(), Path, Phase-1/2 progress figures: dataset, split and concept-label distributions.…, Synthetic CheXpert tree -> Phase-1 pipeline -> processed dir. Returns the dir., build_processed_dataset(), _first_existing() (+30 more)

### Community 59 - "get_data.py"
Cohesion: 0.22
Nodes (9): ArgumentParser, Invariant 9: domain generality claims require a second dataset, Medical/ethical rules, ADR-002: CheXpert first case study, VinDr-CXR generalization study, CheXpert dataset, Scope and limitations, VinDr-CXR dataset, build_arg_parser() (+1 more)

### Community 60 - "ProcessedArtifacts"
Cohesion: 0.18
Nodes (5): CheXpertLayout, ProcessedArtifacts, Any, Everything :func:`build_processed_dataset` produced, for the CLI and the EDA…, What :func:`inspect_download` found under a candidate CheXpert directory.

### Community 61 - "test_chexpert.py"
Cohesion: 0.33
Nodes (5): DataFrame, fixture, CheXpert manifest construction. Path parsing is strict on purpose: a mis-parsed…, A miniature CheXpert CSV covering positive/negative/uncertain/blank and both…, raw_csv()

### Community 62 - "concept_matrix"
Cohesion: 0.22
Nodes (9): concept_matrix(), cooccurrence_matrix(), DataFrame, ndarray, Extract the ``(n, k)`` raw concept matrix (values still 1/0/-1/NaN)., Extract the raw target vector (values still 1/0/-1/NaN)., Pairwise positive co-occurrence counts. A core EDA output: concepts that almost…, target_vector() (+1 more)

### Community 63 - "load_yaml"
Cohesion: 0.21
Nodes (5): load_yaml(), Any, Path, Load a YAML file, resolving bare names against ``configs/``., TestYamlLoading

### Community 64 - "EscalationSpec"
Cohesion: 0.24
Nodes (5): EscalationSpec, Pre-committed rule for switching target when the primary one is too sparse…, Apply the rule. Returns (escalate, human-readable reason)., ADR-003's rule is committed in advance so the choice stays data-driven., TestEscalationRule

### Community 65 - "parse_chexpert_path"
Cohesion: 0.31
Nodes (5): parametrize, parse_chexpert_path(), Extract patient, study and view-file identifiers from a CheXpert image path.…, study1' alone is ambiguous; the study_id must be patient-qualified., TestPathParsing

### Community 66 - "config.py"
Cohesion: 0.25
Nodes (6): CohortSpec, Configuration loading. Every experiment is fully described by YAML. Nothing is…, How to treat CheXpert-style uncertain (-1) and blank labels. See ADR-004., Which images are in scope, and how they are split. See ADR-005., UncertaintySpec, Configuration and the concept specification. ``ConceptSpec`` is the object that…

## Knowledge Gaps
- **86 isolated node(s):** `ddera`, `graphify`, `Usage`, `What graphify is for`, `Step 0 - GitHub repos and multi-path merge (only if a URL or several paths)` (+81 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 400 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **10 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `synthetic_processed()` connect `make_synthetic_cbm` to `build_processed_dataset`, `ConceptSpec`?**
  _High betweenness centrality (0.105) - this node is a cross-community bridge._
- **Why does `ConceptSpec` connect `ConceptSpec` to `EscalationSpec`, `config.py`, `apply_uncertainty_policy`, `make_synthetic_cbm`, `build_processed_dataset`, `dataset.py`, `load_yaml`, `acquire.py`?**
  _High betweenness centrality (0.066) - this node is a cross-community bridge._
- **Why does `build_processed_dataset()` connect `build_processed_dataset` to `patient_level_split`, `apply_uncertainty_policy`, `ConceptSpec`, `make_synthetic_cbm`, `ProcessedArtifacts`, `acquire.py`?**
  _High betweenness centrality (0.056) - this node is a cross-community bridge._
- **Are the 12 inferred relationships involving `ConceptSpec` (e.g. with `main()` and `_build_synthetic()`) actually correct?**
  _`ConceptSpec` has 12 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `build_processed_dataset()` (e.g. with `_build_synthetic()` and `ConceptSpec`) actually correct?**
  _`build_processed_dataset()` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 7 inferred relationships involving `LinearReasoner` (e.g. with `reasoner()` and `TestDecomposition`) actually correct?**
  _`LinearReasoner` has 7 INFERRED edges - model-reasoned connections that need verification._
- **What connects `ddera`, `graphify`, `Usage` to the rest of the system?**
  _86 weakly-connected nodes found - possible documentation gaps or missing edges._