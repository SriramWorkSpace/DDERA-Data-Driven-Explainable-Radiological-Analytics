# Graph Report - DDERA  (2026-09-10)

## Corpus Check
- 97 files · ~310,583 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1533 nodes · 2968 edges · 105 communities (85 shown, 13 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 174 edges (avg confidence: 0.92)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `70de2966`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- safe_auroc
- patient_level_split
- manifest_from_frame
- labels.py
- .from_yaml
- TestCalibration
- verify_gpu.py
- TestCompleteness
- GradCAM
- Step 4: Concept Bottleneck Layer
- predict_fn
- Final Flow Diagram: CNN-based Concept Bottleneck Model for Ante-hoc Explainable Chest X-ray Diagnosis
- Concept Bottleneck Layer
- stability_report
- intervention.py
- ArrayLike
- intervention_effect
- train_model
- 10. Concept Intervention (What-If Analysis)
- soft_vs_hard_leakage
- What You Must Do When Invoked
- intervention_order
- reporting/runs.py
- verify_decomposition
- Phase 3: Milestone M1 professor demo
- Interpretable linear reasoner p=sigma(w.c+b)
- Concept vector bottleneck c in [0,1]^12
- dataset.py
- FeatureCache
- LinearReasoner
- DDERA (Data-Driven Explainable Radiological Analytics)
- acquire.py
- graphify reference: extra exports and benchmark
- graphify reference: query, path, explain
- DDERA tech stack
- set_seed
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
- Phase 8: VinDr-CXR generalization study
- ProcessedArtifacts
- parse_chexpert_path
- plots.py
- load_yaml
- DenseNet121Encoder
- test_reporting.py
- components/runs.py
- apply_uncertainty_policy
- main
- binary_metrics
- run_protocol
- concept_permutation_necessity
- make_synthetic_cbm
- ConceptHead
- CheXpert v1 concept specification file
- 2_Model.py
- concept_metrics
- test_feature_cache.py
- test_acquire.py
- test_train.py
- test_probe.py
- __main__.py
- LinearReasoner
- evaluate_model
- _LabelledDataset
- EscalationSpec
- figures.py
- empirical_sensitivity
- 3_Explainability_Lab.py
- cbm.py
- ADR-009: Local GPU (RX 6800M / gfx1031) via ROCm on native Linux
- ConceptSpec
- rows_to_keep
- label_distribution
- logit
- TestSoftVsHardLeakage
- test_app_components.py
- inspect_download
- main
- test_xai_harness.py
- The eight-family evaluation protocol
- TestVisualizeCLIs
- components/__init__.py
- Any
- ConceptBottleneckModel

## God Nodes (most connected - your core abstractions)
1. `EncoderConfig` - 46 edges
2. `ConceptSpec` - 38 edges
3. `set_seed()` - 36 edges
4. `LinearReasoner` - 35 edges
5. `build_processed_dataset()` - 31 edges
6. `ModelConfig` - 29 edges
7. `DenseNet121Encoder` - 29 edges
8. `patient_level_split()` - 28 edges
9. `build_model()` - 27 edges
10. `safe_auroc()` - 25 edges

## Surprising Connections (you probably didn't know these)
- `Linear logit decomposition logit(p)=b+sum(wj*cj)` --semantically_similar_to--> `Interpretable linear reasoner p=sigma(w.c+b)`  [INFERRED] [semantically similar]
  README.md → ARCHITECTURE.md
- `Eight-family evaluation protocol` --semantically_similar_to--> `The eight-family evaluation protocol`  [INFERRED] [semantically similar]
  README.md → ARCHITECTURE.md
- `Original simple_workflow specification document` --semantically_similar_to--> `The eight-family evaluation protocol`  [INFERRED] [semantically similar]
  simple_workflow.md → ARCHITECTURE.md
- `Final project definition (name, architecture, methodology)` --semantically_similar_to--> `DDERA (Data-Driven Explainable Radiological Analytics)`  [INFERRED] [semantically similar]
  simple_workflow.md → README.md
- `test_config_rejects_unknown_variant()` --uses--> `ModelConfig`  [INFERRED]
  tests/test_cbm_model.py → src/ddera/config.py

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

## Communities (105 total, 13 thin omitted)

### Community 0 - "safe_auroc"
Cohesion: 0.12
Nodes (18): MetricFn, bootstrap_ci(), BootstrapResult, is_significant(), paired_bootstrap_diff(), Any, ArrayLike, Bootstrap confidence intervals. Mandatory on every headline metric. Pneumonia… (+10 more)

### Community 1 - "patient_level_split"
Cohesion: 0.06
Nodes (33): Policy: notebook outputs deliberately kept (no nbstripout), Module: data/splits.py, Repository conventions (configs, seeding, splits, naming, plots, paths), ADR-005: Patient-level splits, frontal views only, ADR-011: Split prevalence checked via z-score, not fixed tolerance, assert_no_patient_leakage(), check_split_integrity(), patient_level_split() (+25 more)

### Community 2 - "manifest_from_frame"
Cohesion: 0.14
Nodes (8): manifest_from_frame(), ndarray, Manifest construction from an already-loaded frame. Split out from…, Extract the raw target vector (values still 1/0/-1/NaN)., target_vector(), Applying an uncertainty policy is a separate, explicit step -- not done here., TestExtraction, TestManifest

### Community 3 - "labels.py"
Cohesion: 0.19
Nodes (10): encode_concept_matrix(), mask_coverage(), float64, NDArray, Label encoding and uncertainty policies (ADR-004). CheXpert-style label…, Apply the policy to an ``(n_samples, n_concepts)`` matrix. Returns ``(labels,…, Fraction of usable labels overall and per concept. Low coverage on a concept is…, Guard against a policy bug silently leaving -1 or NaN in the label array. (+2 more)

### Community 4 - ".from_yaml"
Cohesion: 0.13
Nodes (9): CohortSpec, How to treat CheXpert-style uncertain (-1) and blank labels. See ADR-004., Which images are in scope, and how they are split. See ADR-005., UncertaintySpec, Phase-1 artifacts from a synthetic CheXpert tree *with* generated JPEGs.…, synthetic_processed(), Invariant 4: a target inside the bottleneck would be circular., It is derivable from the other observations, so it would leak by construction. (+1 more)

### Community 5 - "TestCalibration"
Cohesion: 0.08
Nodes (30): BinStrategy, apply_temperature(), _bin_edges(), brier_score(), calibration_report(), expected_calibration_error(), fit_temperature(), maximum_calibration_error() (+22 more)

### Community 6 - "verify_gpu.py"
Cohesion: 0.18
Nodes (23): check_1_device(), check_2_matmul(), check_3_conv2d(), check_4_densenet_amp(), check_5_bce(), check_6_overfit(), check_7_soak(), check_8_vram() (+15 more)

### Community 7 - "TestCompleteness"
Cohesion: 0.08
Nodes (20): completeness_curve(), completeness_ratio(), completeness_report(), CompletenessCurve, _describe(), Any, Concept completeness. How much of the task-relevant information does the…, Build a curve from a ``{residual_width: auroc}`` mapping (as produced by the… (+12 more)

### Community 8 - "GradCAM"
Cohesion: 0.10
Nodes (19): assert_baseline_only(), comparison_table(), GradCAM, lime_explanation(), Any, ArrayLike, float64, NDArray (+11 more)

### Community 9 - "Step 4: Concept Bottleneck Layer"
Cohesion: 0.09
Nodes (30): Ante-hoc explainability (built into the model, not post-hoc), Real medical data: CheXpert / ChestX-ray14, Step 3: CNN / Vision Encoder, Clinical concept: Atelectasis, Step 4: Concept Bottleneck Layer, Clinical concept: Cardiomegaly, Clinical concept: Consolidation, Clinical concept: Edema (+22 more)

### Community 10 - "predict_fn"
Cohesion: 0.18
Nodes (14): predict_fn(), fixture, Shared pytest fixtures. All fixtures are built on :mod:`ddera.data.synthetic`,…, The TRUE linear reasoner that generated the synthetic targets., The ``concepts -> probabilities`` callable the XAI harness consumes., Default dataset: noisy concept predictor, no leakage past the bottleneck., A near-perfect concept predictor: interventions should have almost nothing to…, Target information injected into features that bypasses the concepts entirely. (+6 more)

### Community 11 - "Final Flow Diagram: CNN-based Concept Bottleneck Model for Ante-hoc Explainable Chest X-ray Diagnosis"
Cohesion: 0.16
Nodes (29): 9. Ante-hoc Explanation, Atelectasis (clinical concept), Cardiomegaly (clinical concept), CheXpert Dataset, 4. CNN / Vision Encoder, 10. Concept Intervention (What-If Analysis), 5. Concept Prediction Layer, 6. Concept Representations (interpretable bottleneck) (+21 more)

### Community 12 - "Concept Bottleneck Layer"
Cohesion: 0.10
Nodes (26): Atelectasis (concept), Cardiomegaly (concept), CNN / Vision Encoder, Concept Bottleneck Layer, Concept Intervention (Optional, manual concept adjustment), Stage 1: Concept Predictor (CNN, multi-label BCE loss), Concept Quality metrics (concept accuracy, concept F1, human alignment, intervention sensitivity, sufficiency & necessity, stability/noise test), Stage 2: Concept -> Disease (interpretable classifier, cross-entropy loss) (+18 more)

### Community 13 - "stability_report"
Cohesion: 0.13
Nodes (15): concept_drift(), _describe(), prediction_flip_rate(), Any, ArrayLike, rank_stability(), Explanation stability under clinically irrelevant perturbation. An explanation…, Did the decision change? The clinically consequential view. (+7 more)

### Community 14 - "intervention.py"
Cohesion: 0.13
Nodes (14): Ordering, Concept contributions and intervention analysis. This module is the mechanical…, AUROC as a function of how many concepts were corrected to ground truth., AUROC recovered by correcting every concept. Near zero means the reasoner…, Test-time intervention curve. Progressively replaces predicted concepts with…, Run all four orderings -- the standard Phase 5 intervention panel., tti_all_strategies(), tti_curve() (+6 more)

### Community 15 - "ArrayLike"
Cohesion: 0.33
Nodes (9): _as_concept_matrix(), expected_logit_shift(), ArrayLike, float64, NDArray, Closed-form logit shift for a linear reasoner: ``w_j * (new - old)``., Numerically stable logistic function., ``(n, k)`` matrix of signed contributions ``w_j * c_j``. These terms plus… (+1 more)

### Community 16 - "intervention_effect"
Cohesion: 0.17
Nodes (9): intervene(), intervention_effect(), Any, Return a copy of ``concepts`` with column ``index`` set to ``value``. Never…, Measure what one intervention actually does to the prediction. Returns the…, Raising a positively-weighted concept must raise the prediction, and vice versa., Setting c_j := v must shift the logit by exactly w_j * (v - c_j)., The decorative concept must not move the prediction, whatever it is set to. (+1 more)

### Community 17 - "train_model"
Cohesion: 0.07
Nodes (36): Backend, DeviceInfo, get_device(), get_device_info(), _probe_amp(), Any, device, Backend resolution. This is the ONLY module in the codebase that branches on… (+28 more)

### Community 18 - "10. Concept Intervention (What-If Analysis)"
Cohesion: 0.16
Nodes (16): 9. Ante-hoc Explanation, CheXpert Dataset, 4. CNN / Vision Encoder, 10. Concept Intervention (What-If Analysis), 5. Concept Prediction Layer, 6. Concept Representations, 1. Data Acquisition, 2. Data Cleaning & EDA (+8 more)

### Community 19 - "soft_vs_hard_leakage"
Cohesion: 0.19
Nodes (16): _describe_incompleteness(), _describe_leakage(), leakage_report(), Any, ArrayLike, float64, NDArray, PredictFn (+8 more)

### Community 20 - "What You Must Do When Invoked"
Cohesion: 0.08
Nodes (24): For /graphify add and --watch, For /graphify query, For the commit hook and native CLAUDE.md integration, For --update and --cluster-only, /graphify, Honesty Rules, Interpreter guard for subcommands, Part A - Structural extraction for code files (+16 more)

### Community 21 - "intervention_order"
Cohesion: 0.26
Nodes (6): int_, apply_intervention_order(), intervention_order(), Build an ``(n, k)`` matrix giving, per sample, the order to intervene on…, Replace the first ``n_intervened`` concepts (per sample, per ``order``) with…, TestInterventionOrdering

### Community 22 - "reporting/runs.py"
Cohesion: 0.09
Nodes (28): datetime, _append_index(), compare_runs(), IncompleteRunError, list_runs(), load_run(), log_run(), missing_families() (+20 more)

### Community 23 - "verify_decomposition"
Cohesion: 0.20
Nodes (7): Assert that contributions plus bias reproduce the logit exactly. Returns the…, verify_decomposition(), logit(p) == bias + sum_j w_j c_j, exactly., All-zeros and all-ones are the boundary cases the dashboard sliders can reach., With every concept at zero the logit must be exactly the bias., The guard must actually fire -- a test that can never fail is worthless., TestDecomposition

### Community 24 - "Phase 3: Milestone M1 professor demo"
Cohesion: 0.16
Nodes (14): B0: black-box DenseNet-121 baseline, Two execution profiles: training vs analysis/demo, Feature-cache invalidation via encoder fingerprint, M1: independent CBM (ground-truth concepts), M2: sequential CBM (predicted concepts, practical default), M3: joint CBM, lambda sweep -> trade-off curve, M4: hybrid/residual CBM, k sweep -> completeness curve, Module: features/cache.py (+6 more)

### Community 25 - "Interpretable linear reasoner p=sigma(w.c+b)"
Cohesion: 0.12
Nodes (18): Streamlit application architecture, Concept intervention experiment, Data flow and run artifacts pipeline, Explainability Lab (live concept intervention page), Interpretable linear reasoner p=sigma(w.c+b), M5: uncertainty-aware CBM, Module: models/reasoner.py, Module: reporting/runs.py (+10 more)

### Community 26 - "Concept vector bottleneck c in [0,1]^12"
Cohesion: 0.33
Nodes (6): Concept head: Linear(1024->12)+sigma, Concept vector bottleneck c in [0,1]^12, DenseNet-121 vision encoder stage, Module: models/encoder.py, Invariant 3: Explanations must be intrinsic/ante-hoc, ADR-006: No horizontal flip in augmentation

### Community 27 - "dataset.py"
Cohesion: 0.10
Nodes (15): concept_matrix(), Extract the ``(n, k)`` raw concept matrix (values still 1/0/-1/NaN)., CachedFeatureDataset, EncodedSplit, Path, Tensor, Torch datasets over the Phase-1 artifacts (``splits.parquet``).…, One split as ``(features, concepts, concept_mask, target)`` from the ADR-008… (+7 more)

### Community 28 - "FeatureCache"
Cohesion: 0.11
Nodes (15): memmap, FeatureCache, Fingerprint, Any, DataFrame, device, Path, RuntimeError (+7 more)

### Community 29 - "LinearReasoner"
Cohesion: 0.16
Nodes (6): LinearReasoner, Adapt to the ``predict_fn`` interface the rest of this module consumes., ``p = sigmoid(w . c + b)`` -- DDERA's primary interpretable reasoner.…, The discriminative test: leaky features must probe higher than clean ones., (y - p) / (p(1-p)): positive when under-predicting, negative when over., TestResidualProbe

### Community 30 - "DDERA (Data-Driven Explainable Radiological Analytics)"
Cohesion: 0.31
Nodes (10): The core ante-hoc prediction pathway, ProtoPNet: optional second ante-hoc family, Honesty rules for results, ADR-001: Concept Bottleneck Model as primary architecture, Ante-hoc vs post-hoc explanation inversion, Core research question, DDERA (Data-Driven Explainable Radiological Analytics), Linear logit decomposition logit(p)=b+sum(wj*cj) (+2 more)

### Community 31 - "acquire.py"
Cohesion: 0.14
Nodes (17): Phase-1 data acquisition orchestration (ADR-002/003/004/005/011). Ties the…, build_manifest(), cooccurrence_matrix(), ImageProbeReport, ManifestSummary, probe_image_dimensions(), Any, DataFrame (+9 more)

### Community 32 - "graphify reference: extra exports and benchmark"
Cohesion: 0.22
Nodes (8): graphify reference: extra exports and benchmark, Step 6b - Wiki (only if --wiki flag), Step 7 - Neo4j export (only if --neo4j or --neo4j-push flag), Step 7a - FalkorDB export (only if --falkordb or --falkordb-push flag), Step 7b - SVG export (only if --svg flag), Step 7c - GraphML export (only if --graphml flag), Step 7d - MCP server (only if --mcp flag), Step 8 - Token reduction benchmark (only if total_words > 5000)

### Community 33 - "graphify reference: query, path, explain"
Cohesion: 0.33
Nodes (5): For /graphify explain, For /graphify path, graphify reference: query, path, explain, Step 0 — Constrained query expansion (REQUIRED before traversal), Step 1 — Traversal

### Community 34 - "DDERA tech stack"
Cohesion: 0.40
Nodes (6): pre-commit hook: ruff (--fix), Notebooks are narrative, src/ddera is the library, DDERA tech stack, ADR-007: Python 3.11 (Windows) / 3.12 (Linux training box), Backend-agnostic core dependencies, Dev/quality dependencies (pytest, ruff, black, pre-commit)

### Community 35 - "set_seed"
Cohesion: 0.21
Nodes (13): ModelConfig, Which model to build. See ARCHITECTURE section 5 and ADR-001., build_model(), LinearReasoner, Construct the model for ``cfg.variant``. ``encoder=None`` gives an…, Seed Python, NumPy and PyTorch. Args: seed: the seed. deterministic: request…, set_seed(), _feature_batch() (+5 more)

### Community 36 - "graphify reference: add a URL and watch a folder"
Cohesion: 0.50
Nodes (3): For /graphify add, For --watch, graphify reference: add a URL and watch a folder

### Community 37 - "Invariant 10: simplifications require a documented effect on the research question"
Cohesion: 0.67
Nodes (3): The design gate question, Invariant 10: simplifications require a documented effect on the research question, ADR template with mandatory Effect-on-research-question field

### Community 48 - "EncoderConfig"
Cohesion: 0.12
Nodes (22): EncoderConfig, Configuration loading. Every experiment is fully described by YAML. Nothing is…, Vision encoder specification (ADR-001). Also the source of the feature-cache…, Training augmentation knobs (ADR-006). There are deliberately **no**…, TransformConfig, assert_no_reflection(), build_eval_transform(), build_train_transform() (+14 more)

### Community 52 - "graphify reference: commit hook and native CLAUDE.md integration"
Cohesion: 0.50
Nodes (3): For git commit hook, For native CLAUDE.md integration, graphify reference: commit hook and native CLAUDE.md integration

### Community 53 - "graphify reference: incremental update and cluster-only"
Cohesion: 0.50
Nodes (3): For --cluster-only, For --update (incremental re-extraction), graphify reference: incremental update and cluster-only

### Community 58 - "build_processed_dataset"
Cohesion: 0.19
Nodes (10): build_processed_dataset(), Run the Phase-1 pipeline end to end and write the artifacts. Args: dest: an…, load_manifest(), Read a manifest written by :func:`write_manifest`, restoring…, load_splits(), Read a split table written by :func:`write_splits`, returning ``(df,…, _spec(), TestBuildProcessedDataset (+2 more)

### Community 59 - "Phase 8: VinDr-CXR generalization study"
Cohesion: 0.22
Nodes (11): Why xai/ takes no domain arguments, Invariant 1: DDERA is a methodology project, not merely a classifier, Invariant 2: Chest X-ray is the first validation case study, Invariant 9: domain generality claims require a second dataset, Medical/ethical rules, ADR-002: CheXpert first case study, VinDr-CXR generalization study, Phase 8: VinDr-CXR generalization study, Phase 9: Dashboard and deployment (+3 more)

### Community 60 - "ProcessedArtifacts"
Cohesion: 0.18
Nodes (5): CheXpertLayout, ProcessedArtifacts, Any, Everything :func:`build_processed_dataset` produced, for the CLI and the EDA…, What :func:`inspect_download` found under a candidate CheXpert directory.

### Community 61 - "parse_chexpert_path"
Cohesion: 0.16
Nodes (10): parametrize, parse_chexpert_path(), Extract patient, study and view-file identifiers from a CheXpert image path.…, DataFrame, fixture, CheXpert manifest construction. Path parsing is strict on purpose: a mis-parsed…, A miniature CheXpert CSV covering positive/negative/uncertain/blank and both…, study1' alone is ambiguous; the study_id must be patient-qualified. (+2 more)

### Community 62 - "plots.py"
Cohesion: 0.12
Nodes (28): _denormalise(), plot_augmentation_grid(), plot_concept_cooccurrence(), plot_concept_probe_auroc(), plot_feature_cache_sanity(), plot_feature_space(), plot_label_distribution(), plot_mask_coverage() (+20 more)

### Community 63 - "load_yaml"
Cohesion: 0.31
Nodes (4): load_yaml(), Path, Load a YAML file, resolving bare names against ``configs/``., TestYamlLoading

### Community 64 - "DenseNet121Encoder"
Cohesion: 0.10
Nodes (14): DenseNet121Encoder, Any, no_grad, Tensor, DenseNet-121 feature encoder (ADR-001). ImageNet-pretrained DenseNet-121 with…, ``image -> 1024-d`` DenseNet-121 features. See module docstring for the frozen…, Stable hash of the encoder's parameters -- the identity in the cache…, Everything that must match for a cached feature set to be valid (ADR-008). (+6 more)

### Community 65 - "test_reporting.py"
Cohesion: 0.15
Nodes (12): mark_synthetic(), Figure, Path, One shared plotting theme (CLAUDE.md section 5: no ad-hoc matplotlib styling).…, Stamp a figure as synthetic/demo. Use on every figure not built from real data., Save ``fig`` to ``path`` (creating parents), optionally stamping it synthetic,…, save_figure(), _clean_theme() (+4 more)

### Community 66 - "components/runs.py"
Cohesion: 0.19
Nodes (14): available_runs(), latest_run(), load_reasoner(), Any, LinearReasoner, ndarray, Path, Load run artifacts for the app. Thin wrapper over :mod:`ddera.reporting.runs`. (+6 more)

### Community 67 - "apply_uncertainty_policy"
Cohesion: 0.20
Nodes (6): apply_uncertainty_policy(), Map raw CheXpert label values to ``(labels, mask)``. Args: raw: array of raw…, Truth table: (policy, blank_policy) -> (labels, mask)., No -1 or NaN may survive; either would poison the loss silently., Fail-safe: an accidental unmasked loss degrades to 'negative', never to NaN., TestUncertaintyPolicies

### Community 68 - "main"
Cohesion: 0.19
Nodes (10): _build_synthetic(), _first_rgb_image(), main(), ndarray, Path, Phase-2 progress figures: augmentation, preprocessed samples, encoder feature…, CheXpertImageDataset, One split of ``splits.parquet`` as ``(image, concepts, concept_mask, target)``.… (+2 more)

### Community 69 - "binary_metrics"
Cohesion: 0.16
Nodes (8): binary_metrics(), Average precision, or ``nan`` when only one class is present. More informative…, The standard binary panel at a fixed operating point. Returns AUROC, AUPRC,…, safe_auprc(), y = [1,1,0,0], predictions at 0.5 -> [1,0,1,0]: tp=1 fn=1 fp=1 tn=1., For a random scorer, average precision tends to prevalence -- unlike AUROC's…, A rare concept with no positives must not abort a whole sweep., TestBinaryMetrics

### Community 70 - "run_protocol"
Cohesion: 0.18
Nodes (16): _concept_columns(), _perturbation_names(), Any, DataFrame, ndarray, The eight-family evaluation protocol, assembled from run artifacts…, Assemble every computable metric family. Families that genuinely need more than…, run_protocol() (+8 more)

### Community 71 - "concept_permutation_necessity"
Cohesion: 0.29
Nodes (5): concept_permutation_necessity(), Per-concept necessity by permutation. Shuffles one concept's column across…, The synthetic reasoner has one concept with weight ~0.01. It must be flagged., Concepts the model weights heavily should be the ones it cannot do without., TestConceptNecessity

### Community 72 - "make_synthetic_cbm"
Cohesion: 0.11
Nodes (16): make_synthetic_cbm(), ArrayLike, DataFrame, float64, NDArray, Synthetic concept-bottleneck data with known ground truth. Used for testing the…, A synthetic dataset plus the ground truth that generated it., Logits under the TRUE weights and TRUE concepts -- the achievable ceiling. (+8 more)

### Community 73 - "ConceptHead"
Cohesion: 0.22
Nodes (5): Module, ConceptHead, Tensor, Concept prediction layer (ARCHITECTURE stage 5). ``Linear(1024 ->…, ``features -> concept logits``. Trained with masked BCE against the concept…

### Community 74 - "CheXpert v1 concept specification file"
Cohesion: 0.19
Nodes (13): Cohort definition: frontal views, patient-level split, The 12 concept list (chexpert_v1), Pre-committed escalation rule to leave-one-out sweep, Expected leakage watchlist (Consolidation, Lung Opacity), CheXpert v1 concept specification file, Target definition: Pneumonia, Uncertainty policy: U-Mask concepts, U-Ignore target, ADR-003: Target=Pneumonia, concepts=12 radiographic observations (+5 more)

### Community 75 - "2_Model.py"
Cohesion: 0.23
Nodes (7): The persistent medical disclaimer (CLAUDE.md section 6) and the data-state…, render_disclaimer(), render_state_banner(), DDERA dashboard -- Home. streamlit run app/Home.py Reads run artifacts from…, Data page -- cohort, label and concept distributions from the Phase-1 artifacts., Model page -- predictive metrics, calibration, per-concept quality, from run…, Methodology page -- the pathway, the evaluation protocol, the limitations.

### Community 76 - "concept_metrics"
Cohesion: 0.15
Nodes (11): concept_metrics(), prediction_agreement(), Any, ArrayLike, Predictive and concept-quality metrics. Accuracy alone is meaningless on…, Per-concept quality plus macro averages. Args: c_true: ``(n, k)`` binary…, Fraction of samples on which two probability vectors give the same hard label., Metrics, calibration and bootstrap. Where possible these check against **hand-… (+3 more)

### Community 77 - "test_feature_cache.py"
Cohesion: 0.24
Nodes (6): built_cache(), encoder(), image_ds(), fixture, Feature cache + fingerprint / stale-cache protection (ADR-008). Uses a random-…, TestExtractAndLoad

### Community 78 - "test_acquire.py"
Cohesion: 0.31
Nodes (5): ArgumentParser, build_arg_parser(), main(), Phase-1 acquisition pipeline: local-download inspection, manifest/splits…, TestCLI

### Community 79 - "test_train.py"
Cohesion: 0.20
Nodes (8): _cfg(), _loaders(), Training loop + evaluator + the `python -m ddera.train` entrypoint. Built on…, Wraps make_synthetic_cbm as CBM batches: features linearly encode the concepts,…, _signal_model(), _SignalDataset, TestEvaluator, TestTrainLoop

### Community 80 - "test_probe.py"
Cohesion: 0.18
Nodes (13): linear_probe_concept_auroc(), Any, ArrayLike, Linear concept probe -- the GATE 2 sanity check (project-plan Phase 2 /…, Per-concept logistic-regression AUROC from features to concept labels.…, fixture, Linear concept probe -- the GATE 2 wiring check. Uses ``make_synthetic_cbm``:…, _run() (+5 more)

### Community 81 - "__main__.py"
Cohesion: 0.11
Nodes (27): Phase-1 CheXpert acquisition. Verify a *local* CheXpert download, build the…, ExperimentConfig, A full experiment, resolved from ``configs/experiment/<name>.yaml``. The…, Path, Write a miniature ``CheXpert-v1.0-small/`` tree: ``train.csv``, ``valid.csv``…, write_synthetic_chexpert_tree(), Frozen-encoder feature cache (ADR-008). Run the frozen DenseNet-121 once over…, EncoderWrapped (+19 more)

### Community 82 - "LinearReasoner"
Cohesion: 0.17
Nodes (7): LinearReasoner, Tensor, ``concepts in [0, 1]^k -> target logit``. Deliberately transparent., The ``(k,)`` weight vector -- the explanation., ``(n, k)`` signed contributions ``w_j * c_j``; these + bias sum to the logit., A :class:`ddera.xai.intervention.LinearReasoner` with the trained ``w`` and…, TestExactDecomposition

### Community 83 - "evaluate_model"
Cohesion: 0.30
Nodes (13): _collect(), _concept_probs(), evaluate_model(), _predictions_frame(), _present(), Any, DataFrame, DataLoader (+5 more)

### Community 84 - "_LabelledDataset"
Cohesion: 0.22
Nodes (4): Dataset, _LabelledDataset, ndarray, Common plumbing: an :class:`EncodedSplit` plus the passthrough accessors.

### Community 85 - "EscalationSpec"
Cohesion: 0.22
Nodes (6): EscalationSpec, Pre-committed rule for switching target when the primary one is too sparse…, Apply the rule. Returns (escalate, human-readable reason)., Configuration and the concept specification. ``ConceptSpec`` is the object that…, ADR-003's rule is committed in advance so the choice stays data-driven., TestEscalationRule

### Community 86 - "figures.py"
Cohesion: 0.24
Nodes (11): concept_probability_bars(), contribution_waterfall(), Figure, ndarray, App figures -- built through :mod:`ddera.reporting.theme` so they match the…, Predicted concept probabilities; if ``edited`` is given, overlay the edited…, bias + each signed ``w_j * c_j``, cumulative, landing exactly on the logit., roc_curve_figure() (+3 more)

### Community 87 - "empirical_sensitivity"
Cohesion: 0.17
Nodes (11): empirical_sensitivity(), faithfulness_report(), PredictFn, Estimate ``d logit(p) / d c_j`` per concept by central finite differences.…, Does the model behave the way its weights claim? Compares the measured…, Does the model behave the way its weights claim?, For a linear reasoner, d logit / d c_j must equal w_j., A linear reasoner has the same derivative everywhere; that is why it is… (+3 more)

### Community 88 - "3_Explainability_Lab.py"
Cohesion: 0.25
Nodes (8): contribution_table(), intervene(), Any, ndarray, The concept-intervention identity -- the demonstration of the whole thesis.…, Apply ``edits`` (concept index -> new value) and return the before/after…, Per-concept ``(name, c_j, w_j, w_j * c_j)`` plus a bias row; the…, Explainability Lab -- the demonstration of the whole thesis. Pick a run and a…

### Community 89 - "cbm.py"
Cohesion: 0.17
Nodes (9): BlackBoxModel, Module, Tensor, B0 black-box baseline (ARCHITECTURE section 5). ``encoder -> Linear(1024 ->…, ``features -> target logit`` with no concept bottleneck., Concept-bottleneck composition (ARCHITECTURE stage 6/8) and the model factory.…, The interpretable reasoner (ARCHITECTURE stage 7). ``Linear(n_concepts -> 1)``:…, The trainable concept-bottleneck model -- forward-graph contract (Invariants… (+1 more)

### Community 90 - "ADR-009: Local GPU (RX 6800M / gfx1031) via ROCm on native Linux"
Cohesion: 0.23
Nodes (10): GPU/backend environment workflow (4.1, 4.4), Invariant 7: compute limitations must never justify a black-box swap, Invariant 8: GPU strategy may change, methodology must not, ADR-009: Local GPU (RX 6800M / gfx1031) via ROCm on native Linux, HSA_OVERRIDE_GFX_VERSION=10.3.0 gfx1031->gfx1030 mechanism, Ubuntu dual-boot + ROCm setup procedure, Part 7 verification gate (scripts/verify_gpu.py), Hardware notes: RX 6800M / gfx1031 override (+2 more)

### Community 91 - "ConceptSpec"
Cohesion: 0.23
Nodes (3): ConceptSpec, The concept/target contract for one domain. This is the object the whole…, TestConceptSpec

### Community 92 - "rows_to_keep"
Cohesion: 0.22
Nodes (8): BlankPolicy, bool_, ConceptPolicy, _binarize_target(), Apply ADR-004's target policy: drop uncertain-target rows, then map to 0/1., Boolean selector over samples for a single target column. ``u_ignore`` drops…, rows_to_keep(), TestRowsToKeep

### Community 93 - "label_distribution"
Cohesion: 0.22
Nodes (7): label_distribution(), ArrayLike, Count positive / negative / uncertain / blank. The first thing EDA should…, Uncertainty policies (ADR-004). A bug here silently changes what the model is…, Guards against a typo silently dropping a concept from the bottleneck., test_chexpert_observation_list_is_correct(), TestLabelDistribution

### Community 94 - "logit"
Cohesion: 0.28
Nodes (5): logit(), Inverse of :func:`sigmoid`, clipped away from 0 and 1 to stay finite., The CBM mathematics. These are the correctness-critical tests in the project.…, Saturation must not produce inf/nan: an intervention can push a logit far out., TestSigmoidLogit

### Community 95 - "TestSoftVsHardLeakage"
Cohesion: 0.25
Nodes (4): If concepts are already 0/1, hardening is a no-op and leakage must be exactly 0., The negative direction, which is a different finding and must not be conflated.…, Concepts already near 0/1 barely move when hardened, either way., TestSoftVsHardLeakage

### Community 96 - "test_app_components.py"
Cohesion: 0.22
Nodes (5): fixture, LinearReasoner, App component logic (pure functions). The Streamlit pages are thin glue over…, reasoner(), TestInterventionIdentity

### Community 97 - "inspect_download"
Cohesion: 0.36
Nodes (5): _first_existing(), inspect_download(), Path, Locate ``train.csv`` / ``valid.csv`` and the image tree under ``dest``. Accepts…, TestInspectDownload

### Community 98 - "main"
Cohesion: 0.47
Nodes (5): _build_synthetic(), main(), Path, Phase-1/2 progress figures: dataset, split and concept-label distributions.…, Synthetic CheXpert tree -> Phase-1 pipeline -> processed dir. Returns the dir.

### Community 99 - "test_xai_harness.py"
Cohesion: 0.22
Nodes (5): Leakage, completeness, stability and the post-hoc guard. The recurring pattern…, Invariant 1/9: the harness must not assume chest X-ray, or any particular…, A 22-concept, 6-target domain (VinDr's shape) must need no code change., TestDomainAgnosticism, TestLeakageReport

### Community 100 - "The eight-family evaluation protocol"
Cohesion: 0.29
Nodes (8): The eight-family evaluation protocol, Module: xai/completeness.py, Module: xai/leakage.py, Module: xai/stability.py, ADR-012: Leakage and incompleteness measured as separate quantities, Risk register, Eight-family evaluation protocol, Reusable domain-agnostic methodology contribution

### Community 104 - "ConceptBottleneckModel"
Cohesion: 0.32
Nodes (4): ConceptBottleneckModel, Tensor, ``encoder -> concept_head -> reasoner``, forward over a dataloader batch dict., Cached-feature path uses ``batch['features']``; image path runs the encoder.

## Knowledge Gaps
- **86 isolated node(s):** `ddera`, `graphify`, `Usage`, `What graphify is for`, `Step 0 - GitHub repos and multi-path merge (only if a URL or several paths)` (+81 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 535 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **13 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `ConceptSpec` connect `ConceptSpec` to `main`, `main`, `.from_yaml`, `Any`, `test_acquire.py`, `EncoderConfig`, `__main__.py`, `evaluate_model`, `EscalationSpec`, `build_processed_dataset`, `dataset.py`, `rows_to_keep`, `plots.py`, `acquire.py`?**
  _High betweenness centrality (0.079) - this node is a cross-community bridge._
- **Why does `DDERA (Data-Driven Explainable Radiological Analytics)` connect `DDERA (Data-Driven Explainable Radiological Analytics)` to `patient_level_split`, `The eight-family evaluation protocol`, `CheXpert v1 concept specification file`, `__main__.py`, `ADR-009: Local GPU (RX 6800M / gfx1031) via ROCm on native Linux`, `Phase 8: VinDr-CXR generalization study`?**
  _High betweenness centrality (0.060) - this node is a cross-community bridge._
- **Why does `LinearReasoner` connect `LinearReasoner` to `test_app_components.py`, `components/runs.py`, `test_xai_harness.py`, `run_protocol`, `predict_fn`, `intervention.py`, `ArrayLike`, `LinearReasoner`, `figures.py`, `verify_decomposition`, `3_Explainability_Lab.py`, `cbm.py`, `empirical_sensitivity`, `TestSoftVsHardLeakage`?**
  _High betweenness centrality (0.058) - this node is a cross-community bridge._
- **Are the 17 inferred relationships involving `EncoderConfig` (e.g. with `main()` and `DenseNet121Encoder`) actually correct?**
  _`EncoderConfig` has 17 INFERRED edges - model-reasoned connections that need verification._
- **Are the 20 inferred relationships involving `ConceptSpec` (e.g. with `main()` and `_build_synthetic()`) actually correct?**
  _`ConceptSpec` has 20 INFERRED edges - model-reasoned connections that need verification._
- **Are the 16 inferred relationships involving `LinearReasoner` (e.g. with `contribution_table()` and `intervene()`) actually correct?**
  _`LinearReasoner` has 16 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `build_processed_dataset()` (e.g. with `_build_synthetic()` and `_build_synthetic()`) actually correct?**
  _`build_processed_dataset()` has 5 INFERRED edges - model-reasoned connections that need verification._