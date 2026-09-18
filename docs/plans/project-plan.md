# Project Plan: Interactive Open-Vocabulary 3D Affordance Detection

## 1. Project in One Sentence

Build and evaluate an interactive system that takes a 3D object point cloud and a natural-language affordance query, such as `grasp`, `sit`, or `where should I hold this?`, then highlights the object points that support that action.

## 2. Project Positioning

This is one merged Pattern Recognition and HCI project.

- **Pattern Recognition contribution:** reproduce pretrained OpenAD inference, evaluate it with a documented protocol, study how prompt phrasing affects predictions, and analyze successes and failures.
- **HCI contribution:** provide an interactive interface that lets a user select an object, enter or refine a natural-language query, inspect the predicted affordance heatmap and confidence, and understand model limitations.

The project is an **implementation, evaluation, and interaction study**, not a claim that we invented OpenAD or trained a new state-of-the-art architecture.

## 3. Core Research Question

> How reliably does a pretrained open-vocabulary 3D affordance detector localize functional regions when users express the same intended action using different natural-language prompts, and how can those predictions be presented in an understandable interactive interface?

Supporting questions:

1. Does the pretrained model produce useful point-level predictions on the selected test objects?
2. How much do predictions change when the same affordance is phrased as a label, a question, or a descriptive instruction?
3. Where does the system fail, particularly for unseen affordances, ambiguous language, or visually similar regions?
4. Does an interface with heatmaps, confidence information, and query refinement make the output easier to inspect?

## 4. Scope Decision

### Required core

- Use the official pretrained OpenAD PointNet++ full-shape checkpoint.
- Run inference on the official OpenAD/3D AffordanceNet data.
- Produce point-level colored predictions.
- Evaluate predictions with the official or repository-compatible protocol.
- Run a controlled prompt-phrasing experiment.
- Show representative successes and failures.
- Build one usable interactive demonstration.
- Publish a reproducible public GitHub repository.
- Submit an IEEE-style report and a 5-minute presentation.

### Explicitly out of scope

- Training OpenAD from scratch.
- Claiming a new neural architecture.
- Reproducing every number in the OpenAD paper.
- Training DGCNN or Point Transformer alternatives.
- Large human-subject experiments.
- Presenting simulated HCI data as real user-study evidence.
- Spending the deadline window fixing CUDA after a working inference path exists.

### Optional stretch work

Attempt only after the required core is complete:

1. Per-class IoU and a confusion matrix.
2. Partial-view inference.
3. CLIP versus a sentence-transformer text encoder.
4. t-SNE or UMAP visualization of embeddings.
5. A very small informal usability walkthrough, reported as formative feedback rather than a formal study.

## 5. What Already Exists and What We Add

### Existing work used

- The OpenAD architecture and official source code.
- Pretrained OpenAD model weights.
- PointNet++ point-cloud feature extraction.
- CLIP text embeddings.
- The OpenAD/3D AffordanceNet dataset and annotations.

### Our concrete contribution

1. A reproducible pretrained-inference pipeline in this repository.
2. A documented evaluation performed by us on a fixed test subset or official test split.
3. A prompt-sensitivity experiment using semantically equivalent queries.
4. Quantitative comparison of prompt variants.
5. Qualitative analysis of correct, ambiguous, and failed predictions.
6. An interactive point-cloud interface with query refinement and model feedback.
7. A combined PR/HCI analysis explaining accuracy, behavior, usability, and limitations.

The report and presentation must clearly distinguish our results from values copied from the OpenAD paper.

## 6. Intended User Experience

The demo should support this flow:

1. The user selects a sample 3D object.
2. The user enters an affordance query.
3. The system runs pretrained OpenAD inference.
4. The 3D object is displayed with a color heatmap over its points.
5. The interface displays the query, predicted affordance, and a confidence summary if the model exposes meaningful scores.
6. The user rotates and zooms the object.
7. The user changes the wording and compares the new prediction.
8. The interface communicates uncertainty or unsupported cases without claiming certainty.

Minimum demo fallback: a prepared gallery of recorded predictions with prompt selection. The presentation must not depend on live GPU access.

## 7. Technical Pipeline

```text
3D point cloud (N x 3, plus optional features)
                 |
                 v
        PointNet++ visual encoder
                 |
                 v
       Per-point visual embeddings
                 |
                 |        Natural-language query
                 |                  |
                 |                  v
                 |          Frozen CLIP encoder
                 |                  |
                 +---------> similarity/alignment
                                    |
                                    v
                         per-point affordance score
                                    |
                                    v
                         threshold/color mapping
                                    |
                                    v
                      interactive 3D heatmap output
```

Technical details to explain in the PR presentation:

- **Input:** sampled 3D point cloud and a text affordance query.
- **Visual backbone:** PointNet++ builds local and global point features through hierarchical sampling and grouping.
- **Language encoder:** frozen CLIP converts the query into a semantic embedding.
- **Alignment:** point features and the language embedding are compared in a shared representation space.
- **Output:** a score or class prediction for each point, rendered as an affordance heatmap.
- **Training status:** official pretrained weights are used; no new end-to-end training is claimed.
- **Loss and optimizer:** explain those used to train the published checkpoint from the paper/code, while stating that we did not perform that training. Keep this separate from our inference-time experiment.

## 8. Dataset and Evaluation Protocol

### Dataset facts to verify before publication

- Official dataset name and citation.
- Number of objects, object categories, affordance labels, and points per object.
- Exact downloaded version and directory structure.
- Full-shape versus partial-view setting.
- Official seen/unseen or open-vocabulary split definition.
- Any normalization, point sampling, augmentation, or label remapping performed by the reference code.

Do not place unverified dataset counts in the report or slides.

### Primary evaluation

Use the full-shape setting because it is the shortest and safest route to a complete demonstration.

Report metrics that can be correctly reproduced by the official evaluator:

- **mIoU:** overlap between predicted and ground-truth affordance regions.
- **mAP:** ranking/detection quality across affordance classes or queries.
- **AUC:** threshold-independent quality of point-level predictions.

If time prevents reliable reproduction of all three, report the verified metric or metrics only. Never fabricate missing values.

Report seen and unseen results separately when supported by the selected protocol.

### Evaluation subset fallback

If evaluating the entire official split is too slow:

1. Select a fixed, diverse subset before looking at results.
2. Record every sample identifier and selection rule.
3. Include both seen and unseen cases where possible.
4. Use the same subset for every prompt condition.
5. Call it a subset evaluation everywhere; do not imply full-dataset performance.

### Baselines

The minimum baseline is the canonical affordance-label prompt, for example `grasp`.

If quickly implementable, add:

- Random per-point scores.
- Majority or most-frequent affordance prediction.
- Published OpenAD results as a cited reference value, clearly marked **not reproduced by us**.

Because the project is framed as an implementation/application, elaborate model baselines are secondary to a correct and transparent evaluation.

## 9. Prompt-Phrasing Experiment

### Independent variable

Prompt form for the same intended affordance.

Use three prompt templates per affordance:

1. **Label:** `grasp`
2. **Question:** `Where can I grasp this object?`
3. **Description:** `The part of the object used for holding`

Use three to five affordances supported by the chosen data, such as grasp, sit, pour, open, or contain. Final terms must match the verified label vocabulary.

### Controlled conditions

- Same model checkpoint.
- Same point clouds.
- Same preprocessing and point sampling.
- Same thresholds and evaluator.
- Same hardware and software environment where practical.
- Only prompt wording changes.

### Measurements

- mIoU, mAP, or AUC for each prompt form where ground truth permits.
- Change in predicted positive-point proportion.
- Pairwise overlap between heatmaps for equivalent prompts.
- Representative visual comparison for at least one stable and one unstable case.

### Interpretation

Answer these questions rather than merely showing a table:

- Which prompt form performs best overall?
- Are concise labels more stable than natural questions?
- Which affordances are sensitive to wording?
- Does a visually convincing heatmap always agree with ground truth?
- What behavior should the interface expose to the user?

## 10. Qualitative Analysis

Prepare at least six examples:

- Two clear successes.
- Two partial or ambiguous successes.
- Two failures.
- At least one comparison where prompt wording changes the result.
- At least one unseen/open-vocabulary example if the protocol supports it.

For every example, record:

- Object/sample identifier.
- Ground-truth affordance region.
- Exact prompt.
- Predicted heatmap.
- Metric or confidence value.
- One-sentence interpretation.
- Likely reason for failure, without overstating certainty.

Likely failure categories include ambiguous labels, context-dependent actions, spatial adjacency, small functional regions, semantically similar affordances, and domain shift.

## 11. HCI Design and Evaluation

### User goal

Allow a user to ask how an unfamiliar 3D object can be used and inspect where the model believes that action is possible.

### Design requirements

- The query box accepts natural language.
- The visualization makes high- and low-scoring regions distinguishable.
- The original object geometry remains visible.
- Users can rotate and zoom the point cloud.
- Users can revise a query without restarting the application.
- Loading, success, no-result, and error states are visible.
- The interface does not imply that model output is guaranteed correct.
- Equivalent prompts can be compared with minimal effort.

### Three scenarios

1. **Easy:** A user asks where to grasp a familiar mug-like object.
2. **Moderate:** A user asks where to sit on an unfamiliar chair-like object and inspects the highlighted region.
3. **Difficult:** A user enters an ambiguous or unsupported instruction, receives a weak/inconsistent result, and reformulates the query.

### Human-centered analysis

Document:

- A requirements-to-goals traceability matrix.
- A hierarchical task analysis of selecting an object, entering a query, inspecting output, and refining the query.
- Low-fidelity wireframes followed by the implemented interface.
- Explainability through per-point heatmaps and comparative prompts.
- Human-in-the-loop control through query revision and result inspection.
- Trust considerations: uncertainty, error disclosure, and failure examples.
- Accessibility considerations for color mapping and readable controls.

### HCI evidence

Prefer an honest expert walkthrough and scenario-based evaluation if there is no time or approval for real participants. If simulated data from earlier coursework is reused, label it prominently as simulated in the abstract, methods, results, and limitations. Never call simulated responses a real user study.

## 12. Repository Target Structure

```text
.
|-- README.md
|-- docs/
|   |-- plans/
|   |   `-- project-plan.md
|   |-- report/
|   `-- slides/
|-- src/
|   |-- data/
|   |-- inference/
|   |-- evaluation/
|   |-- visualization/
|   `-- app/
|-- scripts/
|-- configs/
|-- tests/
|-- results/
|   |-- metrics/
|   |-- figures/
|   `-- predictions/
|-- demo/
|-- requirements.txt or environment.yml
`-- _ref_openad/
```

Large datasets and checkpoints must not be committed. Provide download instructions, expected paths, and checksums or filenames when available.

## 13. Step-by-Step Implementation Plan

### Phase 0: Freeze the claim

- [ ] Copy the one-sentence project description into the README.
- [ ] State that official pretrained OpenAD weights are used.
- [ ] State that our contribution is evaluation, prompt analysis, and interaction design.
- [ ] Create a list of claims that require experimental evidence.

**Exit condition:** every team member can explain the project and contribution in under 30 seconds.

### Phase 1: Environment and assets

- [ ] Record available Python, PyTorch, CUDA, GPU, and driver versions.
- [ ] Create the project environment from pinned dependencies.
- [ ] Download the official full-shape dataset.
- [ ] Download the official PointNet++ full-shape checkpoint.
- [ ] Configure data and checkpoint paths without hard-coded personal paths.
- [ ] Run one reference command without modifying model behavior.
- [ ] Save the exact command and logs.

**Time limit:** 4 hours. If CUDA remains broken, move to a known Colab/Kaggle environment or CPU-compatible inference. Do not begin training.

**Exit condition:** one sample produces a nonempty prediction.

### Phase 2: Reproducible inference

- [ ] Wrap the reference inference entry point in a simple repository script.
- [ ] Accept checkpoint, config, sample, prompt, and output path as arguments.
- [ ] Set random seeds where applicable.
- [ ] Export raw point coordinates, ground truth, and prediction scores.
- [ ] Export a colored `.ply` or equivalent viewable result.
- [ ] Verify repeated runs produce equivalent outputs.
- [ ] Add clear failure messages for missing data or checkpoints.

**Exit condition:** a clean command generates a prediction artifact for a chosen object and query.

### Phase 3: Evaluation harness

- [ ] Read the official evaluation implementation before changing it.
- [ ] Confirm label indexing, ignored labels, thresholds, and aggregation.
- [ ] Implement or wrap mIoU first.
- [ ] Add mAP and AUC only if their definitions are verified.
- [ ] Save per-sample, per-affordance, and aggregate results to CSV/JSON.
- [ ] Separate seen and unseen results where the official split supports it.
- [ ] Add a fixed subset manifest if full evaluation is too slow.
- [ ] Run a small sanity test on one or two samples.
- [ ] Run the final selected evaluation once the sanity test passes.

**Exit condition:** at least one trustworthy aggregate metric and its exact evaluation protocol are available.

### Phase 4: Prompt experiment

- [ ] Select three to five verified affordances.
- [ ] Define the three prompt templates before running experiments.
- [ ] Store prompt definitions in a config file.
- [ ] Run every prompt on the exact same sample set.
- [ ] Store raw predictions for later comparison.
- [ ] Produce a summary table by prompt form and affordance.
- [ ] Calculate heatmap overlap or prediction-change statistics.
- [ ] Identify stable and unstable examples.

**Exit condition:** one complete table answers whether wording affected performance or prediction behavior.

### Phase 5: Visualization and demo

- [ ] Choose the simplest compatible viewer, preferably Gradio with Plotly or the existing Gradio Model3D component.
- [ ] Add object/sample selection.
- [ ] Add a text query input.
- [ ] Add example prompts.
- [ ] Render point colors from prediction scores.
- [ ] Show the exact submitted query and a concise result summary.
- [ ] Add loading, failure, and unsupported-query states.
- [ ] Add a reset or compare-prompt workflow.
- [ ] Test the demo locally from a clean start.
- [ ] Record the demo as a presentation fallback.

**Exit condition:** another person can select an object, enter a query, and inspect a result without editing code.

### Phase 6: Qualitative analysis

- [ ] Generate a pool of candidate figures.
- [ ] Select examples according to the predefined success/failure categories.
- [ ] Keep the same color scale across comparable figures.
- [ ] Pair predictions with ground truth where available.
- [ ] Write one concise interpretation for every selected figure.
- [ ] Avoid selecting only attractive outputs.

**Exit condition:** the report has balanced visual evidence and at least two explained failures.

### Phase 7: HCI artifacts

- [ ] Write user goals and system requirements.
- [ ] Complete the traceability matrix.
- [ ] Complete the three scenarios.
- [ ] Complete the hierarchical task analysis.
- [ ] Save initial wireframes.
- [ ] Map the implemented controls to the requirements.
- [ ] Perform scenario walkthroughs and document issues found.
- [ ] Describe explainability, trust, uncertainty, and accessibility decisions.
- [ ] Add a visible simulated-data disclosure wherever applicable.

**Exit condition:** HCI analysis describes and evaluates the same application used in the PR demo.

### Phase 8: Report

- [ ] Use the official IEEE conference template.
- [ ] Write the abstract last, after results are fixed.
- [ ] Include the pipeline/architecture diagram.
- [ ] Include dataset and preprocessing details.
- [ ] Explain PointNet++, CLIP, input/output formulation, published training loss, and published optimizer accurately.
- [ ] Clearly state that our experiments use pretrained weights.
- [ ] Include experimental setup, prompt conditions, hardware, and software.
- [ ] Include quantitative results and interpretation.
- [ ] Include qualitative successes and failures.
- [ ] Include the interaction design and HCI analysis.
- [ ] Include limitations, ethical considerations, and future work.
- [ ] Cite OpenAD, PointNet++, CLIP, the dataset, and every borrowed method or number.
- [ ] Compile from scratch and inspect the final PDF.

Recommended report sections:

1. Introduction and problem formulation
2. Related work
3. Dataset and preprocessing
4. Method and system design
5. Experimental setup
6. Results and prompt ablation
7. Qualitative and failure analysis
8. Interactive system and HCI analysis
9. Limitations and ethical considerations
10. Conclusion

**Exit condition:** the PDF builds cleanly and every major claim points to a result, figure, citation, or explicit limitation.

### Phase 9: Five-minute presentation

Use six slides:

1. **Problem and contribution:** task, relevance, and what we actually did.
2. **Dataset and evaluation:** split, preprocessing, prompts, and metrics.
3. **Method:** one pipeline diagram with PointNet++, CLIP, and point-level output.
4. **Experimental setup:** pretrained checkpoint, controlled prompt comparison, and baseline/reference.
5. **Results:** one compact table plus the meaning of the result.
6. **Qualitative analysis and demo:** successes, failures, interface, limitations, and takeaway.

Presentation rules:

- [ ] Rehearse to 5:00, never beyond 5:30.
- [ ] Assign one speaker per section if using multiple speakers.
- [ ] Do not spend time on generic background.
- [ ] State `pretrained` clearly.
- [ ] Explain results rather than reading values.
- [ ] Use the recorded demo during the presentation unless live inference is known to be reliable and fast.
- [ ] Keep backup slides for architecture detail, metric definitions, and extra examples.

**Exit condition:** the full talk and recorded demo finish within five minutes.

### Phase 10: Repository and submission

- [ ] Replace the placeholder README with setup, data, checkpoint, inference, evaluation, demo, and citation instructions.
- [ ] Pin dependency versions.
- [ ] Remove secrets, personal paths, temporary data, and large files.
- [ ] Include licenses and attribution for reused code.
- [ ] Verify commands from a clean clone or fresh environment.
- [ ] Set the GitHub repository to public.
- [ ] Verify every repository and Drive link in an incognito window.
- [ ] Submit the report before **20 September 2026, 11:59 PM**.
- [ ] Preserve time for the presentation on **21 September 2026**.

**Exit condition:** a reviewer can access the public repository, understand what is ours, reproduce the documented path, and open the final report.

## 14. Forty-Eight-Hour Schedule

### Hours 0-4: Establish a working result

- Freeze scope and claims.
- Configure environment, data, and checkpoint.
- Run pretrained inference on one sample.
- Capture exact commands and blockers.

### Hours 4-10: Build the core pipeline

- Make inference reproducible.
- Export predictions and visualizations.
- Begin the evaluation harness.
- Create report and slide skeletons.

### Hours 10-18: Generate results

- Complete metric validation.
- Run canonical-prompt evaluation.
- Run the prompt-phrasing experiment.
- Save results immediately in machine-readable files.

### Hours 18-26: Build and analyze the interface

- Implement the minimum interactive viewer.
- Select success and failure cases.
- Complete HCI scenarios, task analysis, and traceability.
- Record a first demo as soon as the interface works.

### Hours 26-36: Write and integrate

- Finish the report around actual results.
- Create the architecture diagram and result figures.
- Build the six presentation slides.
- Complete README and reproducibility instructions.

### Hours 36-42: Verify and freeze

- Run the documented workflow from a clean start.
- Compile and proofread the report.
- Verify citations, numbers, figures, repository visibility, and links.
- Record the final demo.
- Freeze result-changing code.

### Hours 42-48: Submit and rehearse

- Submit early.
- Rehearse the five-minute presentation repeatedly.
- Prepare local copies of slides, report, video, and key outputs.
- Use remaining time only for presentation-blocking fixes.

## 15. Go/No-Go Gates

### Gate 1: Hour 4

**Required:** one nonempty pretrained prediction.

If missing, stop all interface and writing refinements. Switch environment or use the simplest known compatible inference path.

### Gate 2: Hour 14

**Required:** a verified evaluation path on at least one sample.

If missing, narrow to a documented subset and one trustworthy metric. Do not add new metrics or models.

### Gate 3: Hour 24

**Required:** a results table, qualitative figures, and a minimal demo path.

If missing, remove all stretch work. Use a prepared result gallery if interactive inference is unreliable.

### Gate 4: Hour 36

**Required:** complete report draft, slides, README, and recorded demo.

After this gate, make only correctness, reproducibility, and presentation fixes.

## 16. Risks and Fallbacks

| Risk | Early signal | Response |
|---|---|---|
| CUDA/package incompatibility | Reference inference fails in the first hours | Move to a known Colab/Kaggle image or CPU-compatible inference; do not train |
| Dataset/checkpoint download delay | Assets are unavailable by hour 2 | Download the minimum full-shape assets and checkpoint; work on report/demo scaffolding in parallel |
| Official metrics are unclear | Results disagree with reference behavior | Read evaluator code, report only verified metrics, and document protocol precisely |
| Full evaluation is too slow | Estimated run exceeds available compute | Freeze a predetermined subset and label it as subset evaluation |
| Prompt text is constrained by code | Arbitrary sentences cannot be passed cleanly | Implement the smallest text-encoding change consistent with the reference pipeline; otherwise compare verified label templates and disclose the constraint |
| Interactive GPU inference is slow | Each query takes too long for a demo | Cache predictions for selected samples/prompts and use a recorded demonstration |
| Results are weaker than published values | Reproduced score is low | Report it honestly, check split/protocol compatibility, and focus analysis on causes and limitations |
| No time for real HCI participants | Recruitment has not begun | Use scenario walkthroughs and heuristic/formative analysis; label simulated evidence explicitly |
| Team integration fails | Outputs use different formats or paths | Agree on CSV/JSON/PLY contracts early and appoint one integration owner |

## 17. Team Work Split

Assign names based on actual team size. One person may own multiple roles.

- **Inference owner:** environment, checkpoint, reference inference, reproducible commands.
- **Evaluation owner:** split validation, metrics, prompt experiment, result tables.
- **Interface owner:** point-cloud rendering, query flow, comparison states, demo recording.
- **PR writing owner:** method, experimental setup, results, architecture diagram, slides.
- **HCI owner:** requirements, scenarios, task analysis, traceability, trust and usability analysis.
- **Integration owner:** repository structure, README, citations, final builds, submission checklist.

Coordination rules:

- Use one shared result schema and sample naming convention.
- Never overwrite raw predictions; generate derived tables and figures separately.
- Commit small working changes with descriptive messages.
- Communicate blockers immediately.
- Only the integration owner changes final report and slide structure during the final six hours.

## 18. Definition of Done

The project is complete when all of the following are true:

- [ ] A documented command runs pretrained OpenAD inference.
- [ ] At least one verified metric is reported on a clearly defined evaluation set.
- [ ] Canonical and alternative prompt conditions are compared fairly.
- [ ] Raw and summarized results are saved.
- [ ] Successful, ambiguous, and failed predictions are shown.
- [ ] The interactive demo works or a recorded fallback proves the workflow.
- [ ] The report accurately explains the architecture and our use of pretrained weights.
- [ ] The HCI analysis concerns the same implemented system.
- [ ] All simulated or non-human evidence is labeled honestly.
- [ ] The public repository contains setup and reproduction instructions.
- [ ] The IEEE PDF is submitted by the deadline.
- [ ] The presentation fits within 5 minutes and 30 seconds.

## 19. Final Claim Template

Use language close to this in the report and presentation:

> We implemented and evaluated a pretrained open-vocabulary 3D affordance detection pipeline based on OpenAD. We examined how semantically equivalent natural-language prompts affect point-level affordance predictions, analyzed representative successes and failures, and developed an interactive interface that supports query refinement and inspection of model outputs. We do not claim a new model architecture or newly trained state-of-the-art weights.

## 20. Immediate Next Actions

Do these in order:

1. Obtain the official dataset and pretrained PointNet++ full-shape checkpoint.
2. Confirm one inference result using `_ref_openad/test_open_vocab.py`.
3. Record the working environment and exact command.
4. Inspect the official evaluator and reproduce one metric on a tiny sample.
5. Freeze the evaluation subset and prompt list.
6. Run canonical and alternative prompts.
7. Build the interactive viewer around saved predictions first, then add live inference if reliable.
8. Write the report and slides around the results that actually exist.

