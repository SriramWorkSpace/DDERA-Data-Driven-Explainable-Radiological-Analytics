# Graph Report - DDERA  (2026-09-10)

## Corpus Check
- 84 files · ~305,838 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1433 nodes · 2709 edges · 93 communities (75 shown, 11 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 165 edges (avg confidence: 0.91)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `6e89109a`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- safe_auroc
- patient_level_split
- manifest_from_frame
- labels.py
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
- predict_fn
- train_model
- 10. Concept Intervention (What-If Analysis)
- test_xai_harness.py
- What You Must Do When Invoked
- intervention_order
- runs.py
- verify_decomposition
- Phase 3: Milestone M1 professor demo
- The eight-family evaluation protocol
- Concept vector bottleneck c in [0,1]^12
- CheXpertImageDataset
- FeatureCache
- LinearReasoner
- DDERA (Data-Driven Explainable Radiological Analytics)
- acquire.py
- graphify reference: extra exports and benchmark
- graphify reference: query, path, explain
- DDERA tech stack
- ModelConfig
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
- _binarize_target
- parse_chexpert_path
- plots.py
- load_yaml
- DenseNet121Encoder
- test_reporting.py
- synthetic_cached
- apply_uncertainty_policy
- config.py
- binary_metrics
- splits.py
- ADR-009: Local GPU (RX 6800M / gfx1031) via ROCm on native Linux
- synthetic.py
- cbm.py
- CheXpert v1 concept specification file
- BootstrapResult
- concept_metrics
- test_feature_cache.py
- test_acquire.py
- set_seed
- test_probe.py
- __main__.py
- LinearReasoner
- evaluate_model
- _LabelledDataset
- EscalationSpec
- concept_matrix
- intervention.py
- rows_to_keep
- BlackBoxModel
- assert_no_patient_leakage
- ADR-005: Patient-level splits, frontal views only
- _SignalDataset

## God Nodes (most connected - your core abstractions)
1. `EncoderConfig` - 45 edges
2. `ConceptSpec` - 39 edges
3. `set_seed()` - 34 edges
4. `build_processed_dataset()` - 31 edges
5. `DenseNet121Encoder` - 29 edges
6. `patient_level_split()` - 28 edges
7. `LinearReasoner` - 25 edges
8. `ModelConfig` - 24 edges
9. `build_model()` - 24 edges
10. `safe_auroc()` - 23 edges

## Surprising Connections (you probably didn't know these)
- `Linear logit decomposition logit(p)=b+sum(wj*cj)` --semantically_similar_to--> `Interpretable linear reasoner p=sigma(w.c+b)`  [INFERRED] [semantically similar]
  README.md → ARCHITECTURE.md
- `Eight-family evaluation protocol` --semantically_similar_to--> `The eight-family evaluation protocol`  [INFERRED] [semantically similar]
  README.md → ARCHITECTURE.md
- `Original simple_workflow specification document` --semantically_similar_to--> `The eight-family evaluation protocol`  [INFERRED] [semantically similar]
  simple_workflow.md → ARCHITECTURE.md
- `Final project definition (name, architecture, methodology)` --semantically_similar_to--> `DDERA (Data-Driven Explainable Radiological Analytics)`  [INFERRED] [semantically similar]
  simple_workflow.md → README.md
- `Ante-hoc vs post-hoc explanation inversion` --semantically_similar_to--> `The core ante-hoc prediction pathway`  [INFERRED] [semantically similar]
  README.md → ARCHITECTURE.md

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

## Communities (93 total, 11 thin omitted)

### Community 0 - "safe_auroc"
Cohesion: 0.16
Nodes (14): MetricFn, bootstrap_ci(), is_significant(), paired_bootstrap_diff(), ArrayLike, Bootstrap confidence intervals. Mandatory on every headline metric. Pneumonia…, CI on ``metric(A) - metric(B)`` using shared resample indices. This is the…, True when a difference interval excludes zero. (+6 more)

### Community 1 - "patient_level_split"
Cohesion: 0.10
Nodes (15): check_split_integrity(), patient_level_split(), Full integrity report: leakage, coverage, and prevalence drift. Prevalence…, Split a manifest into train/val/test with **no patient appearing in two…, manifest(), fixture, Patient-level split integrity (ADR-005).…, A correct split must not trip the prevalence check. The val split holds ~10% of… (+7 more)

### Community 2 - "manifest_from_frame"
Cohesion: 0.23
Nodes (4): manifest_from_frame(), Manifest construction from an already-loaded frame. Split out from…, Applying an uncertainty policy is a separate, explicit step -- not done here., TestManifest

### Community 3 - "labels.py"
Cohesion: 0.12
Nodes (17): encode_concept_matrix(), label_distribution(), mask_coverage(), ArrayLike, float64, NDArray, Label encoding and uncertainty policies (ADR-004). CheXpert-style label…, Count positive / negative / uncertain / blank. The first thing EDA should… (+9 more)

### Community 4 - "ConceptSpec"
Cohesion: 0.12
Nodes (10): CohortSpec, ConceptSpec, How to treat CheXpert-style uncertain (-1) and blank labels. See ADR-004., Which images are in scope, and how they are split. See ADR-005., The concept/target contract for one domain. This is the object the whole…, UncertaintySpec, Invariant 4: a target inside the bottleneck would be circular., It is derivable from the other observations, so it would leak by construction. (+2 more)

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

### Community 10 - "make_synthetic_cbm"
Cohesion: 0.16
Nodes (17): make_synthetic_cbm(), ArrayLike, Generate a synthetic concept-bottleneck dataset. Args: n_patients: number of…, fixture, Shared pytest fixtures. All fixtures are built on :mod:`ddera.data.synthetic`,…, The TRUE linear reasoner that generated the synthetic targets., Default dataset: noisy concept predictor, no leakage past the bottleneck., A near-perfect concept predictor: interventions should have almost nothing to… (+9 more)

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
Cohesion: 0.38
Nodes (9): _as_concept_matrix(), expected_logit_shift(), ArrayLike, float64, NDArray, Closed-form logit shift for a linear reasoner: ``w_j * (new - old)``., Numerically stable logistic function., ``(n, k)`` matrix of signed contributions ``w_j * c_j``. These terms plus… (+1 more)

### Community 16 - "predict_fn"
Cohesion: 0.17
Nodes (10): intervene(), intervention_effect(), Return a copy of ``concepts`` with column ``index`` set to ``value``. Never…, Measure what one intervention actually does to the prediction. Returns the…, predict_fn(), The ``concepts -> probabilities`` callable the XAI harness consumes., Raising a positively-weighted concept must raise the prediction, and vice versa., Setting c_j := v must shift the logit by exactly w_j * (v - c_j). (+2 more)

### Community 17 - "train_model"
Cohesion: 0.07
Nodes (36): Backend, DeviceInfo, get_device(), get_device_info(), _probe_amp(), Any, device, Backend resolution. This is the ONLY module in the codebase that branches on… (+28 more)

### Community 18 - "10. Concept Intervention (What-If Analysis)"
Cohesion: 0.16
Nodes (16): 9. Ante-hoc Explanation, CheXpert Dataset, 4. CNN / Vision Encoder, 10. Concept Intervention (What-If Analysis), 5. Concept Prediction Layer, 6. Concept Representations, 1. Data Acquisition, 2. Data Cleaning & EDA (+8 more)

### Community 19 - "test_xai_harness.py"
Cohesion: 0.06
Nodes (34): concept_permutation_necessity(), _describe_incompleteness(), _describe_leakage(), leakage_report(), Any, ArrayLike, float64, NDArray (+26 more)

### Community 20 - "What You Must Do When Invoked"
Cohesion: 0.08
Nodes (24): For /graphify add and --watch, For /graphify query, For the commit hook and native CLAUDE.md integration, For --update and --cluster-only, /graphify, Honesty Rules, Interpreter guard for subcommands, Part A - Structural extraction for code files (+16 more)

### Community 21 - "intervention_order"
Cohesion: 0.26
Nodes (6): int_, apply_intervention_order(), intervention_order(), Build an ``(n, k)`` matrix giving, per sample, the order to intervene on…, Replace the first ``n_intervened`` concepts (per sample, per ``order``) with…, TestInterventionOrdering

### Community 22 - "runs.py"
Cohesion: 0.09
Nodes (28): datetime, _append_index(), compare_runs(), IncompleteRunError, list_runs(), load_run(), log_run(), missing_families() (+20 more)

### Community 23 - "verify_decomposition"
Cohesion: 0.20
Nodes (7): Assert that contributions plus bias reproduce the logit exactly. Returns the…, verify_decomposition(), logit(p) == bias + sum_j w_j c_j, exactly., All-zeros and all-ones are the boundary cases the dashboard sliders can reach., With every concept at zero the logit must be exactly the bias., The guard must actually fire -- a test that can never fail is worthless., TestDecomposition

### Community 24 - "Phase 3: Milestone M1 professor demo"
Cohesion: 0.15
Nodes (15): B0: black-box DenseNet-121 baseline, Two execution profiles: training vs analysis/demo, Feature-cache invalidation via encoder fingerprint, M1: independent CBM (ground-truth concepts), M2: sequential CBM (predicted concepts, practical default), M3: joint CBM, lambda sweep -> trade-off curve, M4: hybrid/residual CBM, k sweep -> completeness curve, Module: features/cache.py (+7 more)

### Community 25 - "The eight-family evaluation protocol"
Cohesion: 0.12
Nodes (19): Streamlit application architecture, Concept intervention experiment, Data flow and run artifacts pipeline, The eight-family evaluation protocol, Explainability Lab (live concept intervention page), Interpretable linear reasoner p=sigma(w.c+b), M5: uncertainty-aware CBM, Module: models/reasoner.py (+11 more)

### Community 26 - "Concept vector bottleneck c in [0,1]^12"
Cohesion: 0.33
Nodes (6): Concept head: Linear(1024->12)+sigma, Concept vector bottleneck c in [0,1]^12, DenseNet-121 vision encoder stage, Module: models/encoder.py, Invariant 3: Explanations must be intrinsic/ante-hoc, ADR-006: No horizontal flip in augmentation

### Community 27 - "CheXpertImageDataset"
Cohesion: 0.10
Nodes (16): CachedFeatureDataset, CheXpertImageDataset, EncodedSplit, Path, Tensor, Torch datasets over the Phase-1 artifacts (``splits.parquet``).…, One split of ``splits.parquet`` as ``(image, concepts, concept_mask, target)``.…, One split as ``(features, concepts, concept_mask, target)`` from the ADR-008… (+8 more)

### Community 28 - "FeatureCache"
Cohesion: 0.11
Nodes (15): memmap, FeatureCache, Fingerprint, Any, DataFrame, device, Path, RuntimeError (+7 more)

### Community 29 - "LinearReasoner"
Cohesion: 0.12
Nodes (14): empirical_sensitivity(), faithfulness_report(), LinearReasoner, PredictFn, Adapt to the ``predict_fn`` interface the rest of this module consumes., Estimate ``d logit(p) / d c_j`` per concept by central finite differences.…, Does the model behave the way its weights claim? Compares the measured…, ``p = sigmoid(w . c + b)`` -- DDERA's primary interpretable reasoner.… (+6 more)

### Community 30 - "DDERA (Data-Driven Explainable Radiological Analytics)"
Cohesion: 0.24
Nodes (12): The core ante-hoc prediction pathway, ProtoPNet: optional second ante-hoc family, Honesty rules for results, Uncertainty policy: U-Mask concepts, U-Ignore target, ADR-001: Concept Bottleneck Model as primary architecture, ADR-004: Uncertainty policy U-Mask/U-Ignore, Ante-hoc vs post-hoc explanation inversion, Core research question (+4 more)

### Community 31 - "acquire.py"
Cohesion: 0.14
Nodes (17): ProcessedArtifacts, Phase-1 data acquisition orchestration (ADR-002/003/004/005/011). Ties the…, Everything :func:`build_processed_dataset` produced, for the CLI and the EDA…, build_manifest(), ImageProbeReport, ManifestSummary, probe_image_dimensions(), Any (+9 more)

### Community 32 - "graphify reference: extra exports and benchmark"
Cohesion: 0.22
Nodes (8): graphify reference: extra exports and benchmark, Step 6b - Wiki (only if --wiki flag), Step 7 - Neo4j export (only if --neo4j or --neo4j-push flag), Step 7a - FalkorDB export (only if --falkordb or --falkordb-push flag), Step 7b - SVG export (only if --svg flag), Step 7c - GraphML export (only if --graphml flag), Step 7d - MCP server (only if --mcp flag), Step 8 - Token reduction benchmark (only if total_words > 5000)

### Community 33 - "graphify reference: query, path, explain"
Cohesion: 0.33
Nodes (5): For /graphify explain, For /graphify path, graphify reference: query, path, explain, Step 0 — Constrained query expansion (REQUIRED before traversal), Step 1 — Traversal

### Community 34 - "DDERA tech stack"
Cohesion: 0.40
Nodes (6): pre-commit hook: ruff (--fix), Notebooks are narrative, src/ddera is the library, DDERA tech stack, ADR-007: Python 3.11 (Windows) / 3.12 (Linux training box), Backend-agnostic core dependencies, Dev/quality dependencies (pytest, ruff, black, pre-commit)

### Community 35 - "ModelConfig"
Cohesion: 0.19
Nodes (13): LinearReasoner, ModelConfig, Which model to build. See ARCHITECTURE section 5 and ADR-001., build_model(), Construct the model for ``cfg.variant``. ``encoder=None`` gives an…, _feature_batch(), The trainable concept-bottleneck model -- forward-graph contract (Invariants…, The ante-hoc guarantee: move the representation within the concept head's null… (+5 more)

### Community 36 - "graphify reference: add a URL and watch a folder"
Cohesion: 0.50
Nodes (3): For /graphify add, For --watch, graphify reference: add a URL and watch a folder

### Community 37 - "Invariant 10: simplifications require a documented effect on the research question"
Cohesion: 0.67
Nodes (3): The design gate question, Invariant 10: simplifications require a documented effect on the research question, ADR template with mandatory Effect-on-research-question field

### Community 48 - "EncoderConfig"
Cohesion: 0.14
Nodes (19): Compose, EncoderConfig, Vision encoder specification (ADR-001). Also the source of the feature-cache…, Training augmentation knobs (ADR-006). There are deliberately **no**…, TransformConfig, assert_no_reflection(), build_eval_transform(), build_train_transform() (+11 more)

### Community 52 - "graphify reference: commit hook and native CLAUDE.md integration"
Cohesion: 0.50
Nodes (3): For git commit hook, For native CLAUDE.md integration, graphify reference: commit hook and native CLAUDE.md integration

### Community 53 - "graphify reference: incremental update and cluster-only"
Cohesion: 0.50
Nodes (3): For --cluster-only, For --update (incremental re-extraction), graphify reference: incremental update and cluster-only

### Community 58 - "build_processed_dataset"
Cohesion: 0.22
Nodes (8): build_processed_dataset(), Run the Phase-1 pipeline end to end and write the artifacts. Args: dest: an…, load_manifest(), Read a manifest written by :func:`write_manifest`, restoring…, _spec(), TestBuildProcessedDataset, TestImageProbing, TestManifestPersistence

### Community 59 - "Phase 8: VinDr-CXR generalization study"
Cohesion: 0.16
Nodes (14): Why xai/ takes no domain arguments, Module: xai/posthoc.py (B0 baselines only), Invariant 1: DDERA is a methodology project, not merely a classifier, Invariant 2: Chest X-ray is the first validation case study, Invariant 5: post-hoc methods are comparison baselines only, Invariant 9: domain generality claims require a second dataset, Medical/ethical rules, ADR-002: CheXpert first case study, VinDr-CXR generalization study (+6 more)

### Community 60 - "_binarize_target"
Cohesion: 0.20
Nodes (5): _binarize_target(), CheXpertLayout, Any, Apply ADR-004's target policy: drop uncertain-target rows, then map to 0/1., What :func:`inspect_download` found under a candidate CheXpert directory.

### Community 61 - "parse_chexpert_path"
Cohesion: 0.16
Nodes (10): parametrize, parse_chexpert_path(), Extract patient, study and view-file identifiers from a CheXpert image path.…, DataFrame, fixture, CheXpert manifest construction. Path parsing is strict on purpose: a mis-parsed…, A miniature CheXpert CSV covering positive/negative/uncertain/blank and both…, study1' alone is ambiguous; the study_id must be patient-qualified. (+2 more)

### Community 62 - "plots.py"
Cohesion: 0.12
Nodes (28): _denormalise(), plot_augmentation_grid(), plot_concept_cooccurrence(), plot_concept_probe_auroc(), plot_feature_cache_sanity(), plot_feature_space(), plot_label_distribution(), plot_mask_coverage() (+20 more)

### Community 63 - "load_yaml"
Cohesion: 0.14
Nodes (7): _encoder_kwargs(), load_yaml(), Any, Path, Load a YAML file, resolving bare names against ``configs/``., Configuration and the concept specification. ``ConceptSpec`` is the object that…, TestYamlLoading

### Community 64 - "DenseNet121Encoder"
Cohesion: 0.10
Nodes (14): DenseNet121Encoder, Any, no_grad, Tensor, ``image -> 1024-d`` DenseNet-121 features. See module docstring for the frozen…, Stable hash of the encoder's parameters -- the identity in the cache…, Everything that must match for a cached feature set to be valid (ADR-008)., frozen_encoder() (+6 more)

### Community 65 - "test_reporting.py"
Cohesion: 0.09
Nodes (20): _build_synthetic(), main(), Path, Phase-1/2 progress figures: dataset, split and concept-label distributions.…, Synthetic CheXpert tree -> Phase-1 pipeline -> processed dir. Returns the dir., apply_theme(), mark_synthetic(), Figure (+12 more)

### Community 66 - "synthetic_cached"
Cohesion: 0.29
Nodes (7): _build_synthetic(), Path, Path, Write a miniature ``CheXpert-v1.0-small/`` tree: ``train.csv``, ``valid.csv``…, write_synthetic_chexpert_tree(), A synthetic feature cache + splits.parquet for the Phase-3 training/eval tests.…, synthetic_cached()

### Community 67 - "apply_uncertainty_policy"
Cohesion: 0.20
Nodes (6): apply_uncertainty_policy(), Map raw CheXpert label values to ``(labels, mask)``. Args: raw: array of raw…, Truth table: (policy, blank_policy) -> (labels, mask)., No -1 or NaN may survive; either would poison the loss silently., Fail-safe: an accidental unmasked loss degrades to 'negative', never to NaN., TestUncertaintyPolicies

### Community 68 - "config.py"
Cohesion: 0.22
Nodes (7): Phase-1 CheXpert acquisition. Verify a *local* CheXpert download, build the…, _first_rgb_image(), main(), ndarray, Phase-2 progress figures: augmentation, preprocessed samples, encoder feature…, Configuration loading. Every experiment is fully described by YAML. Nothing is…, DenseNet-121 feature encoder (ADR-001). ImageNet-pretrained DenseNet-121 with…

### Community 69 - "binary_metrics"
Cohesion: 0.12
Nodes (14): binary_metrics(), prediction_agreement(), ArrayLike, Predictive and concept-quality metrics. Accuracy alone is meaningless on…, Fraction of samples on which two probability vectors give the same hard label., Average precision, or ``nan`` when only one class is present. More informative…, The standard binary panel at a fixed operating point. Returns AUROC, AUPRC,…, safe_auprc() (+6 more)

### Community 70 - "splits.py"
Cohesion: 0.20
Nodes (11): load_splits(), Any, DataFrame, Path, Patient-level dataset splitting (ADR-005). The single most common source of…, Write ``splits.parquet`` (the manifest plus its ``split`` column) and a JSON…, Read a split table written by :func:`write_splits`, returning ``(df,…, Summary of a split, for logging into the run config and the dashboard. (+3 more)

### Community 71 - "ADR-009: Local GPU (RX 6800M / gfx1031) via ROCm on native Linux"
Cohesion: 0.23
Nodes (10): GPU/backend environment workflow (4.1, 4.4), Invariant 7: compute limitations must never justify a black-box swap, Invariant 8: GPU strategy may change, methodology must not, ADR-009: Local GPU (RX 6800M / gfx1031) via ROCm on native Linux, HSA_OVERRIDE_GFX_VERSION=10.3.0 gfx1031->gfx1030 mechanism, Ubuntu dual-boot + ROCm setup procedure, Part 7 verification gate (scripts/verify_gpu.py), Hardware notes: RX 6800M / gfx1031 override (+2 more)

### Community 72 - "synthetic.py"
Cohesion: 0.16
Nodes (9): DataFrame, float64, NDArray, Synthetic concept-bottleneck data with known ground truth. Used for testing the…, A synthetic dataset plus the ground truth that generated it., Logits under the TRUE weights and TRUE concepts -- the achievable ceiling., An image-level manifest shaped like the real CheXpert one, for split testing., _sigmoid() (+1 more)

### Community 73 - "cbm.py"
Cohesion: 0.14
Nodes (10): ConceptBottleneckModel, Module, Tensor, Concept-bottleneck composition (ARCHITECTURE stage 6/8) and the model factory.…, ``encoder -> concept_head -> reasoner``, forward over a dataloader batch dict., Cached-feature path uses ``batch['features']``; image path runs the encoder., ConceptHead, Tensor (+2 more)

### Community 74 - "CheXpert v1 concept specification file"
Cohesion: 0.21
Nodes (13): Module: xai/leakage.py, The 12 concept list (chexpert_v1), Pre-committed escalation rule to leave-one-out sweep, Expected leakage watchlist (Consolidation, Lung Opacity), CheXpert v1 concept specification file, Target definition: Pneumonia, ADR-003: Target=Pneumonia, concepts=12 radiographic observations, ADR-012: Leakage and incompleteness measured as separate quantities (+5 more)

### Community 75 - "BootstrapResult"
Cohesion: 0.29
Nodes (4): BootstrapResult, Any, A point estimate with a percentile confidence interval., Interval width -- feeds the ADR-003 escalation rule.

### Community 76 - "concept_metrics"
Cohesion: 0.27
Nodes (5): concept_metrics(), Any, Per-concept quality plus macro averages. Args: c_true: ``(n, k)`` binary…, This is what makes the u_mask policy evaluable rather than merely trainable., TestConceptMetrics

### Community 77 - "test_feature_cache.py"
Cohesion: 0.24
Nodes (6): built_cache(), encoder(), image_ds(), fixture, Feature cache + fingerprint / stale-cache protection (ADR-008). Uses a random-…, TestExtractAndLoad

### Community 78 - "test_acquire.py"
Cohesion: 0.17
Nodes (10): ArgumentParser, build_arg_parser(), main(), _first_existing(), inspect_download(), Path, Locate ``train.csv`` / ``valid.csv`` and the image tree under ``dest``. Accepts…, Phase-1 acquisition pipeline: local-download inspection, manifest/splits… (+2 more)

### Community 79 - "set_seed"
Cohesion: 0.23
Nodes (9): Seed Python, NumPy and PyTorch. Args: seed: the seed. deterministic: request…, set_seed(), TestExactDecomposition, _cfg(), _loaders(), Training loop + evaluator + the `python -m ddera.train` entrypoint. Built on…, _signal_model(), TestEvaluator (+1 more)

### Community 80 - "test_probe.py"
Cohesion: 0.18
Nodes (13): linear_probe_concept_auroc(), Any, ArrayLike, Linear concept probe -- the GATE 2 sanity check (project-plan Phase 2 /…, Per-concept logistic-regression AUROC from features to concept labels.…, fixture, Linear concept probe -- the GATE 2 wiring check. Uses ``make_synthetic_cbm``:…, _run() (+5 more)

### Community 81 - "__main__.py"
Cohesion: 0.19
Nodes (16): ExperimentConfig, A full experiment, resolved from ``configs/experiment/<name>.yaml``. The…, Frozen-encoder feature cache (ADR-008). Run the frozen DenseNet-121 once over…, make_generator(), Deterministic seeding. Every notebook and script calls ``set_seed(cfg.seed)``…, DataLoader ``worker_init_fn``: keeps augmentation reproducible across workers., Generator for DataLoader shuffling, so batch order is reproducible., seed_worker() (+8 more)

### Community 82 - "LinearReasoner"
Cohesion: 0.16
Nodes (7): LinearReasoner, Tensor, The interpretable reasoner (ARCHITECTURE stage 7). ``Linear(n_concepts -> 1)``:…, ``concepts in [0, 1]^k -> target logit``. Deliberately transparent., The ``(k,)`` weight vector -- the explanation., ``(n, k)`` signed contributions ``w_j * c_j``; these + bias sum to the logit., A :class:`ddera.xai.intervention.LinearReasoner` with the trained ``w`` and…

### Community 83 - "evaluate_model"
Cohesion: 0.22
Nodes (13): _collect(), evaluate_model(), _predictions_frame(), Any, DataFrame, DataLoader, device, ndarray (+5 more)

### Community 84 - "_LabelledDataset"
Cohesion: 0.20
Nodes (4): Dataset, _LabelledDataset, ndarray, Common plumbing: an :class:`EncodedSplit` plus the passthrough accessors.

### Community 85 - "EscalationSpec"
Cohesion: 0.24
Nodes (5): EscalationSpec, Pre-committed rule for switching target when the primary one is too sparse…, Apply the rule. Returns (escalate, human-readable reason)., ADR-003's rule is committed in advance so the choice stays data-driven., TestEscalationRule

### Community 86 - "concept_matrix"
Cohesion: 0.21
Nodes (8): concept_matrix(), cooccurrence_matrix(), ndarray, Extract the ``(n, k)`` raw concept matrix (values still 1/0/-1/NaN)., Extract the raw target vector (values still 1/0/-1/NaN)., Pairwise positive co-occurrence counts. A core EDA output: concepts that almost…, target_vector(), TestExtraction

### Community 87 - "intervention.py"
Cohesion: 0.24
Nodes (6): logit(), Concept contributions and intervention analysis. This module is the mechanical…, Inverse of :func:`sigmoid`, clipped away from 0 and 1 to stay finite., The CBM mathematics. These are the correctness-critical tests in the project.…, Saturation must not produce inf/nan: an intervention can push a logit far out., TestSigmoidLogit

### Community 88 - "rows_to_keep"
Cohesion: 0.28
Nodes (6): BlankPolicy, bool_, ConceptPolicy, Boolean selector over samples for a single target column. ``u_ignore`` drops…, rows_to_keep(), TestRowsToKeep

### Community 89 - "BlackBoxModel"
Cohesion: 0.28
Nodes (5): BlackBoxModel, Module, Tensor, B0 black-box baseline (ARCHITECTURE section 5). ``encoder -> Linear(1024 ->…, ``features -> target logit`` with no concept bottleneck.

### Community 90 - "assert_no_patient_leakage"
Cohesion: 0.38
Nodes (4): assert_no_patient_leakage(), Raise if any patient appears in more than one split. This is the guarantee the…, The guard must actually fire. A leakage test that cannot fail is worthless., TestLeakageDetection

### Community 91 - "ADR-005: Patient-level splits, frontal views only"
Cohesion: 0.33
Nodes (6): Policy: notebook outputs deliberately kept (no nbstripout), Module: data/splits.py, Repository conventions (configs, seeding, splits, naming, plots, paths), Cohort definition: frontal views, patient-level split, ADR-005: Patient-level splits, frontal views only, ADR-011: Split prevalence checked via z-score, not fixed tolerance

## Knowledge Gaps
- **86 isolated node(s):** `ddera`, `graphify`, `Usage`, `What graphify is for`, `Step 0 - GitHub repos and multi-path merge (only if a URL or several paths)` (+81 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 502 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **11 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `ConceptSpec` connect `ConceptSpec` to `test_reporting.py`, `synthetic_cached`, `config.py`, `make_synthetic_cbm`, `test_acquire.py`, `_binarize_target`, `__main__.py`, `evaluate_model`, `EscalationSpec`, `build_processed_dataset`, `CheXpertImageDataset`, `acquire.py`, `plots.py`, `load_yaml`?**
  _High betweenness centrality (0.092) - this node is a cross-community bridge._
- **Why does `DDERA (Data-Driven Explainable Radiological Analytics)` connect `DDERA (Data-Driven Explainable Radiological Analytics)` to `config.py`, `ADR-009: Local GPU (RX 6800M / gfx1031) via ROCm on native Linux`, `CheXpert v1 concept specification file`, `ADR-005: Patient-level splits, frontal views only`, `The eight-family evaluation protocol`, `Phase 8: VinDr-CXR generalization study`?**
  _High betweenness centrality (0.051) - this node is a cross-community bridge._
- **Why does `LinearReasoner` connect `LinearReasoner` to `make_synthetic_cbm`, `tti_curve`, `ArrayLike`, `LinearReasoner`, `test_xai_harness.py`, `intervention.py`, `verify_decomposition`?**
  _High betweenness centrality (0.048) - this node is a cross-community bridge._
- **Are the 16 inferred relationships involving `EncoderConfig` (e.g. with `main()` and `DenseNet121Encoder`) actually correct?**
  _`EncoderConfig` has 16 INFERRED edges - model-reasoned connections that need verification._
- **Are the 21 inferred relationships involving `ConceptSpec` (e.g. with `main()` and `_build_synthetic()`) actually correct?**
  _`ConceptSpec` has 21 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `build_processed_dataset()` (e.g. with `_build_synthetic()` and `_build_synthetic()`) actually correct?**
  _`build_processed_dataset()` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 10 inferred relationships involving `DenseNet121Encoder` (e.g. with `main()` and `FeatureCache`) actually correct?**
  _`DenseNet121Encoder` has 10 INFERRED edges - model-reasoned connections that need verification._